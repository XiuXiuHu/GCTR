import math

import torch
from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, tokens):
        value, weights = self.cross_attn(
            query, tokens, tokens, need_weights=False, average_attn_weights=False
        )
        query = self.norm1(query + value)
        return self.norm2(query + self.ffn(query)), weights


class MaskedContext(CrossAttention):
    def forward(self, query, tokens, valid):
        if valid.dtype != torch.bool or valid.shape != tokens.shape[:2]:
            raise ValueError("Context masks must be boolean [batch, candidates].")
        empty = ~valid.any(dim=1)
        safe = valid.clone()
        safe[empty, 0] = True
        tokens = tokens.masked_fill(~valid[:, :, None], 0)
        value, _ = self.cross_attn(
            query, tokens, tokens, key_padding_mask=~safe, need_weights=False
        )
        value = self.norm1(query + value)
        value = self.norm2(value + self.ffn(value))
        return value.masked_fill(empty[:, None, None], 0)


class GCTR(nn.Module):
    def __init__(self, queries, heads=8, layers=2, dropout=0.1, target_state=None):
        super().__init__()
        if queries.ndim != 2 or queries.shape[1] % heads:
            raise ValueError(
                "Gene queries must have shape [genes, dim], with dim divisible by heads."
            )
        dim = queries.shape[1]
        self.register_buffer("queries", queries.clone().float())
        self.target = nn.ModuleList([CrossAttention(dim, heads, dropout) for _ in range(layers)])
        if target_state is not None:
            self.target.load_state_dict(target_state, strict=True)
        self.head = nn.Linear(dim, 1)
        self.context = MaskedContext(dim, heads, dropout)
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.context_query = nn.Sequential(nn.Linear(dim, 384), nn.GELU(), nn.Linear(384, dim))
        self.far_coefficient = nn.Parameter(torch.zeros(()))

    def forward(self, target, context, valid):
        if target.ndim != 3 or context.shape != (len(target), 80, self.queries.shape[1]):
            raise ValueError("Expected target [B,P,D] and context [B,80,D].")
        query = self.queries[None].expand(len(target), -1, -1)
        feature = query
        for layer in self.target:
            feature, _ = layer(feature, target)
        query = self.context_query(query)
        local = self.context(query, context[:, :8], valid[:, :8])
        global_context = self.context(query, context[:, 8:], valid[:, 8:])
        feature = feature + self.alpha_logit.sigmoid() * local
        feature = feature + self.far_coefficient.tanh() * global_context
        return self.head(feature).squeeze(-1)


def load_target_state(path):
    from safetensors import safe_open

    prefix = "gene_decoder.layers."
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        state = {
            key[len(prefix) :]: checkpoint.get_tensor(key)
            for key in checkpoint.keys()
            if key.startswith(prefix)
        }
    if not state:
        raise ValueError("No pretrained gene_decoder.layers tensors found.")
    return state
