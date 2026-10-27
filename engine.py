import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import Dataset
from metrics import masked_mse, metrics
from model import GCTR, load_target_state


def seed_everything(seed, threads=2):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def read_genes(path):
    genes = pd.read_csv(path, dtype=str)["gene_symbol"].tolist()
    if (
        not genes
        or len(set(genes)) != len(genes)
        or any(not isinstance(g, str) or not g for g in genes)
    ):
        raise ValueError("genes.csv must contain unique gene_symbol values in frozen order.")
    return genes


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def predict_batches(model, dataset, ids, batch_size, device):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            target, context, valid = dataset.batch(ids[start : start + batch_size], device)[:3]
            value = model(target, context, valid)
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite predictions.")
            predictions.append(value.cpu().numpy())
    if not predictions:
        raise ValueError("The requested split is empty.")
    return np.concatenate(predictions)


def train(options):
    seed_everything(options.seed, options.threads)
    device = torch.device(options.device)
    genes = read_genes(options.genes)
    queries = torch.from_numpy(np.load(options.queries)).float()
    if (
        queries.shape != (options.num_genes, options.dim)
        or len(genes) != options.num_genes
        or not torch.isfinite(queries).all()
    ):
        raise ValueError("Gene panel, queries and configured dimensions do not agree.")
    if not options.pretrained:
        raise ValueError(
            "Training requires --pretrained model.safetensors; demo uses random initialization."
        )
    dataset = Dataset(options.manifest, ["train", "val"], len(genes), options.dim, options.tokens)
    if not len(dataset.splits["train"]) or not len(dataset.splits["val"]):
        raise ValueError("Training and validation sections are both required.")
    output = Path(options.output)
    if not options.resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError("Resume requires the original output directory.")
    model = GCTR(
        queries,
        options.heads,
        options.layers,
        options.dropout,
        load_target_state(options.pretrained),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.lr, weight_decay=options.weight_decay
    )
    history, best, best_mse, stale, first_epoch = [], -float("inf"), float("inf"), 0, 1
    if options.resume:
        state = torch.load(options.resume, map_location="cpu", weights_only=False)
        if state["genes"] != genes:
            raise ValueError("Resume gene order mismatch.")
        for name in [
            "seed",
            "dim",
            "heads",
            "layers",
            "dropout",
            "tokens",
            "batch_size",
            "lr",
            "weight_decay",
            "patience",
        ]:
            if state["options"][name] != getattr(options, name):
                raise ValueError(f"Resume option changed: {name}.")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        history, best, best_mse, stale = (
            state["history"],
            state["best"],
            state["best_mse"],
            state["stale"],
        )
        first_epoch = state["epoch"] + 1
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    write_json(output / "options.json", vars(options))
    for epoch in range(first_epoch, options.epochs + 1):
        model.train()
        order = np.random.default_rng(
            np.random.SeedSequence([options.seed, epoch - 1, 100])
        ).permutation(dataset.splits["train"])
        total_error, total_entries = 0.0, 0
        for start in range(0, len(order), options.batch_size):
            target, context, valid, y, mask = dataset.batch(
                order[start : start + options.batch_size], device
            )
            if not mask.any():
                continue
            prediction = model(target, context, valid)
            if not torch.isfinite(prediction).all():
                raise ValueError("Non-finite training predictions.")
            loss = masked_mse(prediction, y, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            ):
                raise ValueError("Non-finite gradients.")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Non-finite parameters.")
            entries = int(mask.sum())
            total_error += float(loss.detach()) * entries
            total_entries += entries
        if not total_entries:
            raise ValueError("No valid training supervision.")
        ids = dataset.splits["val"]
        prediction = predict_batches(model, dataset, ids, options.batch_size, device)
        validation, _, _ = metrics(dataset.y[ids], prediction, dataset.mask[ids])
        score = validation["gene_pcc_mean"]
        if score is None:
            raise ValueError("Validation Gene PCC is undefined.")
        improved, improved_mse = score > best, validation["mse"] < best_mse
        best, best_mse = max(best, score), min(best_mse, validation["mse"])
        stale = 0 if improved else stale + 1
        history.append(dict(epoch=epoch, train_mse=total_error / total_entries, **validation))
        state = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            validation=validation,
            genes=genes,
            options=vars(options),
            history=history,
            best=best,
            best_mse=best_mse,
            stale=stale,
            rng_python=random.getstate(),
            rng_numpy=np.random.get_state(),
            rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        )
        if improved:
            torch.save(state, output / "best_pcc.pt")
        if improved_mse:
            torch.save(state, output / "best_mse.pt")
        torch.save(state, output / "last.pt")
        np.save(output / f"validation_{epoch:03d}.npy", prediction)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(json.dumps(dict(epoch=epoch, **validation)), flush=True)
        if options.patience and stale >= options.patience:
            break


def evaluate(options, labels=True):
    if not options.checkpoint:
        raise ValueError("--checkpoint is required.")
    seed_everything(options.seed, options.threads)
    state = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
    genes = read_genes(options.genes)
    if genes != state["genes"]:
        raise ValueError("Evaluation gene order differs from the checkpoint.")
    config = state["options"]
    model = GCTR(state["model"]["queries"], config["heads"], config["layers"], config["dropout"])
    model.load_state_dict(state["model"], strict=True)
    model.to(options.device)
    dataset = Dataset(
        options.manifest,
        [options.split],
        len(genes),
        config["dim"],
        config["tokens"],
        labels=labels,
    )
    ids = dataset.splits[options.split]
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "selection.json",
        dict(
            checkpoint=str(options.checkpoint), epoch=state["epoch"], validation=state["validation"]
        ),
    )
    prediction = predict_batches(model, dataset, ids, config["batch_size"], options.device)
    np.save(output / "predictions.npy", prediction)
    index = dataset.index.iloc[ids].reset_index(drop=True)
    index.to_csv(output / "spot_index.csv", index=False)
    pd.DataFrame(dict(gene_symbol=genes)).to_csv(output / "genes.csv", index=False)
    if labels:
        y, mask = dataset.y[ids], dataset.mask[ids]
        summary, gene, spot = metrics(y, prediction, mask)
        np.save(output / "targets.npy", y)
        np.save(output / "supervision_mask.npy", mask)
        pd.DataFrame(dict(gene_symbol=genes, pcc=gene)).to_csv(
            output / "gene_metrics.csv", index=False
        )
        pd.DataFrame(dict(spot_key=index.spot_key, pcc=spot)).to_csv(
            output / "spot_metrics.csv", index=False
        )
        sections = []
        for sample in index.sample_id.unique():
            take = index.sample_id.eq(sample).to_numpy()
            values, _, _ = metrics(y[take], prediction[take], mask[take])
            sections.append(
                dict(sample_id=sample, patient_id=index.loc[take, "patient_id"].iloc[0], **values)
            )
        frame = pd.DataFrame(sections)
        frame.to_csv(output / "section_metrics.csv", index=False)
        write_json(output / "metrics.json", summary)
        write_json(
            output / "section_mean_sd.json",
            {
                key: dict(
                    mean=float(frame[key].mean()),
                    sd=float(frame[key].std(ddof=1)) if len(frame) > 1 else None,
                    n_sections=len(frame),
                )
                for key in ["gene_pcc_mean", "spot_pcc_mean", "mse", "mae"]
            },
        )
        print(json.dumps(summary), flush=True)
