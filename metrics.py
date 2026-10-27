import numpy as np
import torch


def masked_mse(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("Predictions, targets and boolean masks must have identical shapes.")
    if not bool(mask.any()):
        raise ValueError("The batch has no valid supervision.")
    prediction, target = prediction[mask], target[mask]
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Non-finite supervised values.")
    return ((prediction - target) ** 2).mean()


def metrics(target, prediction, mask):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if target.shape != prediction.shape or target.shape != mask.shape or not mask.any():
        raise ValueError("Invalid evaluation shapes or empty supervision.")
    if not np.isfinite(target[mask]).all() or not np.isfinite(prediction).all():
        raise ValueError("Non-finite evaluation values.")

    def correlation(a, b):
        if len(a) < 2:
            return np.nan
        a, b = a - a.mean(), b - b.mean()
        aa, bb = (a * a).sum(), (b * b).sum()
        return float((a * b).sum() / np.sqrt(aa * bb)) if aa > 0 and bb > 0 else np.nan

    gene = np.array(
        [
            correlation(target[mask[:, g], g], prediction[mask[:, g], g])
            for g in range(target.shape[1])
        ]
    )
    spot = np.array(
        [correlation(target[i, mask[i]], prediction[i, mask[i]]) for i in range(len(target))]
    )
    defined_gene, defined_spot = gene[np.isfinite(gene)], spot[np.isfinite(spot)]
    error = prediction[mask] - target[mask]
    summary = dict(
        gene_pcc_mean=float(defined_gene.mean()) if len(defined_gene) else None,
        gene_pcc_median=float(np.median(defined_gene)) if len(defined_gene) else None,
        spot_pcc_mean=float(defined_spot.mean()) if len(defined_spot) else None,
        spot_pcc_median=float(np.median(defined_spot)) if len(defined_spot) else None,
        mse=float((error * error).mean()),
        mae=float(np.abs(error).mean()),
        valid_genes=int(len(defined_gene)),
        undefined_genes=int(len(gene) - len(defined_gene)),
        valid_spots=int(len(defined_spot)),
        undefined_spots=int(len(spot) - len(defined_spot)),
        valid_entries=int(mask.sum()),
        masked_entries=int((~mask).sum()),
        total_entries=int(mask.size),
    )
    return summary, gene, spot
