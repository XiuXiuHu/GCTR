import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def manifest_rows(path):
    path = Path(path).resolve()
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"sample_id", "patient_id", "split", "path"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest columns must include {sorted(required)}.")
    if frame[list(required)].eq("").any().any() or frame.sample_id.duplicated().any():
        raise ValueError("Empty manifest fields or duplicate samples.")
    if not frame.split.isin(["train", "val", "test"]).all():
        raise ValueError("Split labels must be train, val or test.")
    if not frame.groupby("patient_id").split.nunique().eq(1).all():
        raise ValueError("A patient occurs in more than one split.")
    frame["path"] = [str((path.parent / value).resolve()) for value in frame.path]
    if "image" in frame:
        frame["image"] = [
            str((path.parent / value).resolve()) if value else "" for value in frame.image
        ]
    return frame


def sample_global(coordinates, keys, eligible, target):
    pool = np.flatnonzero(eligible & (np.arange(len(coordinates)) != target))
    distance = np.linalg.norm(coordinates[pool] - coordinates[target], axis=1)
    order = np.lexsort((keys[pool], distance))
    rest = pool[order][18:]
    digest = hashlib.sha256(str(keys[target]).encode()).digest()
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [0, int.from_bytes(digest[:4], "little"), int.from_bytes(digest[4:8], "little"), 1872]
        )
    )
    chosen = []
    for stratum in np.array_split(rest, 4):
        if len(stratum):
            chosen.extend(rng.choice(stratum, min(18, len(stratum)), replace=False).tolist())
    if len(chosen) < 72:
        remaining = np.array([index for index in rest if index not in set(chosen)], dtype=np.int64)
        if len(remaining):
            chosen.extend(
                rng.choice(remaining, min(72 - len(chosen), len(remaining)), replace=False).tolist()
            )
    result = np.full(72, -1, dtype=np.int64)
    result[: len(chosen)] = chosen
    return result


def build_context(manifest):
    for row in manifest_rows(manifest).itertuples():
        folder = Path(row.path)
        index = pd.read_csv(folder / "spot_index.csv", dtype={"spot_key": str})
        keys = index.spot_key.to_numpy(str)
        coordinates = index[["x_um", "y_um"]].to_numpy(np.float64)
        eligible = np.load(folder / "candidate_valid.npy")
        if (
            eligible.dtype != bool
            or eligible.shape != (len(index),)
            or not np.isfinite(coordinates).all()
        ):
            raise ValueError(f"Invalid geometry for {row.sample_id}.")
        for name in ["global_indices.npy", "global_mask.npy"]:
            if (folder / name).exists():
                raise FileExistsError(folder / name)
        indices = np.stack(
            [sample_global(coordinates, keys, eligible, i) for i in range(len(index))]
        )
        np.save(folder / "global_indices.npy", indices)
        np.save(folder / "global_mask.npy", indices >= 0)
        print(f"{row.sample_id}: {len(index)} spots, global context ready", flush=True)


class Dataset:
    def __init__(self, manifest, splits, genes, dim=256, tokens=256, labels=True):
        frame = manifest_rows(manifest)
        self.parts = []
        self.locations = []
        indexes, targets, masks = [], [], []
        for row in frame[frame.split.isin(splits)].itertuples():
            folder = Path(row.path)
            index = pd.read_csv(folder / "spot_index.csv", dtype=str)
            if "spot_key" not in index or index.spot_key.duplicated().any():
                raise ValueError(f"Unique spot_key values required for {row.sample_id}.")
            if "sample_id" in index and not index.sample_id.eq(row.sample_id).all():
                raise ValueError("Sample identity mismatch.")
            if "patient_id" in index and not index.patient_id.eq(row.patient_id).all():
                raise ValueError("Patient identity mismatch.")
            index["sample_id"], index["patient_id"], index["split"] = (
                row.sample_id,
                row.patient_id,
                row.split,
            )
            target = np.load(folder / "target.npy", mmap_mode="r")
            local = np.load(folder / "local.npy", mmap_mode="r")
            local_mask = np.load(folder / "local_mask.npy")
            global_indices = np.load(folder / "global_indices.npy")
            global_mask = np.load(folder / "global_mask.npy")
            n = len(index)
            if target.shape != (n, tokens, dim) or local.shape != (n, 8, dim):
                raise ValueError(f"Feature dimensions do not match options for {row.sample_id}.")
            if target.dtype != np.float32 or local.dtype != np.float32:
                raise ValueError("Features must be float32.")
            if local_mask.shape != (n, 8) or local_mask.dtype != bool:
                raise ValueError("local_mask must be boolean [spots,8].")
            if global_indices.shape != (n, 72) or global_mask.shape != (n, 72):
                raise ValueError("Global context must have 72 padded positions.")
            if global_mask.dtype != bool or not np.issubdtype(global_indices.dtype, np.integer):
                raise ValueError("Invalid global index or mask dtype.")
            if (
                not np.array_equal(global_indices >= 0, global_mask)
                or (global_indices[global_mask] >= n).any()
            ):
                raise ValueError("Global candidate indices and masks do not agree.")
            for i in range(n):
                used = global_indices[i, global_mask[i]]
                if i in used or len(np.unique(used)) != len(used):
                    raise ValueError("Global context contains the target or duplicate candidates.")
            pool = np.empty((n, dim), dtype=np.float32)
            for start in range(0, n, 64):
                if (
                    not np.isfinite(target[start : start + 64]).all()
                    or not np.isfinite(local[start : start + 64]).all()
                ):
                    raise ValueError("Non-finite image features.")
                pool[start : start + 64] = np.mean(
                    target[start : start + 64], axis=1, dtype=np.float32
                )
            self.locations.extend((len(self.parts), i) for i in range(n))
            self.parts.append((target, local, local_mask, pool, global_indices, global_mask))
            indexes.append(index)
            if labels:
                y = np.load(folder / "targets.npy")
                mask = np.load(folder / "supervision_mask.npy")
                if y.shape != (n, genes) or mask.shape != y.shape or mask.dtype != bool:
                    raise ValueError("Invalid target or supervision mask shape.")
                if not np.array_equal(np.isfinite(y), mask):
                    raise ValueError(
                        "Unavailable targets must be NaN and masked; observed zeros remain valid."
                    )
                targets.append(y.astype(np.float32, copy=False))
                masks.append(mask)
        if not indexes:
            raise ValueError("No sections found for the requested splits.")
        self.index = pd.concat(indexes, ignore_index=True)
        if self.index.spot_key.duplicated().any():
            raise ValueError("Duplicate spot identities across sections.")
        self.splits = {split: np.flatnonzero(self.index.split.eq(split)) for split in splits}
        self.y = np.concatenate(targets) if labels else None
        self.mask = np.concatenate(masks) if labels else None

    def batch(self, ids, device):
        targets, contexts, valid = [], [], []
        for i in ids:
            section, row = self.locations[i]
            target, local, local_mask, pool, indices, global_mask = self.parts[section]
            far = np.array(pool[np.maximum(indices[row], 0)], copy=True)
            far[~global_mask[row]] = 0
            targets.append(target[row])
            contexts.append(np.concatenate([local[row], far]))
            valid.append(np.concatenate([local_mask[row], global_mask[row]]))
        arrays = [np.stack(targets), np.stack(contexts), np.stack(valid)]
        if self.y is not None:
            arrays.extend([self.y[ids], self.mask[ids]])
        return [torch.from_numpy(a).to(device) for a in arrays]
