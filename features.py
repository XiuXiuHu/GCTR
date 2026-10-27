import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from data import manifest_rows
from engine import seed_everything


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4194304), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_tile(image, box):
    left, top, right, bottom = map(int, box)
    tile = Image.new("RGB", (right - left, bottom - top), "white")
    cl, ct, cr, cb = max(0, left), max(0, top), min(image.width, right), min(image.height, bottom)
    if cr > cl and cb > ct:
        tile.paste(image.crop((cl, ct, cr, cb)), (cl - left, ct - top))
    return tile.resize((224, 224), Image.Resampling.BILINEAR)


def extract_features(options):
    from deepspotm import DeepSpotM

    if not options.weights:
        raise ValueError("Feature extraction requires --weights with an official model directory.")
    seed_everything(options.seed, options.threads)
    encoder, transform = DeepSpotM.from_pretrained(
        options.weights, source="scgpt", device=options.device
    )
    encoder.eval().requires_grad_(False)
    if any(p.dtype != torch.float32 for p in encoder.parameters()):
        raise ValueError("The feature encoder must use FP32.")
    Image.MAX_IMAGE_PIXELS = None
    rows = manifest_rows(options.manifest)
    if "image" not in rows:
        raise ValueError("Extraction requires an image column in the manifest.")
    for row in rows.itertuples():
        folder = Path(row.path)
        index = pd.read_csv(folder / "spot_index.csv", dtype=str)
        boxes = np.load(folder / "crop_boxes.npy")
        valid = np.load(folder / "local_mask.npy")
        if (
            boxes.shape != (len(index), 9, 4)
            or valid.shape != (len(index), 8)
            or valid.dtype != bool
        ):
            raise ValueError("Expected frozen boxes [N,9,4] and boolean local_mask [N,8].")
        if not np.issubdtype(boxes.dtype, np.integer) or np.any(boxes[:, :, 2:] <= boxes[:, :, :2]):
            raise ValueError("Crop boxes must be integer XYXY rectangles.")
        if not hasattr(row, "mpp") or float(row.mpp) <= 0:
            raise ValueError("Extraction requires a positive mpp value.")
        native = round(154.05 / float(row.mpp))
        if np.max(np.abs((boxes[:, :, 2:] - boxes[:, :, :2]) - native)) > 1:
            raise ValueError("Crop sizes do not match the 154.05 micrometer physical FOV.")
        if index.spot_key.duplicated().any():
            raise ValueError("Duplicate spot keys.")
        for name in ["target.npy", "local.npy", "features.json"]:
            if (folder / name).exists():
                raise FileExistsError(folder / name)
        with Image.open(row.image) as source:
            image = source.convert("RGB")

        def batch(start):
            keys = [divmod(i, 9) for i in range(start, min(start + 32, len(index) * 9))]
            tensor = torch.stack([transform(crop_tile(image, boxes[i, j])) for i, j in keys])
            with torch.inference_mode():
                raw = encoder.backbone.forward_patch_tokens(
                    tensor.to(options.device, dtype=torch.float32)
                )
                target = encoder.gene_decoder.patch_proj(raw)
                context = target.mean(dim=1)
            if (
                target.shape[1:] != (256, 256)
                or not torch.isfinite(target).all()
                or not torch.isfinite(context).all()
            ):
                raise ValueError("Unexpected Midnight features.")
            return keys, target.cpu().numpy(), context.cpu().numpy()

        keys, first_target, first_context = batch(0)
        _, repeated_target, repeated_context = batch(0)
        replay = max(
            float(np.max(np.abs(first_target - repeated_target))),
            float(np.max(np.abs(first_context - repeated_context))),
        )
        if replay > 1e-4:
            raise ValueError(f"Production replay failed: {replay}.")
        target = np.lib.format.open_memmap(
            folder / "target.npy", mode="w+", dtype=np.float32, shape=(len(index), 256, 256)
        )
        context = np.lib.format.open_memmap(
            folder / "local.npy", mode="w+", dtype=np.float32, shape=(len(index), 8, 256)
        )
        starts = list(range(0, len(index) * 9, 32))
        for start in starts:
            keys, t, c = batch(start)
            for j, (spot, tile) in enumerate(keys):
                if tile == 0:
                    target[spot] = t[j]
                else:
                    context[spot, tile - 1] = c[j]
        target.flush()
        context.flush()
        for start in [starts[0], starts[-1]]:
            keys, t, c = batch(start)
            for j, (spot, tile) in enumerate(keys):
                error = float(
                    np.max(
                        np.abs(
                            (target[spot] if tile == 0 else context[spot, tile - 1])
                            - (t[j] if tile == 0 else c[j])
                        )
                    )
                )
                replay = max(replay, error)
        if replay > 1e-4:
            raise ValueError(f"Cached-feature replay failed: {replay}.")
        metadata = dict(
            encoder="Midnight",
            precision="FP32",
            batch_tiles=32,
            resize=224,
            physical_fov_um=154.05,
            target_shape=list(target.shape),
            local_shape=list(context.shape),
            replay_max_abs_error=replay,
            image_sha256=checksum(row.image),
            weights_sha256=checksum(Path(options.weights) / "model.safetensors"),
            crop_boxes_sha256=checksum(folder / "crop_boxes.npy"),
            local_mask_sha256=checksum(folder / "local_mask.npy"),
            target_sha256=checksum(folder / "target.npy"),
            local_sha256=checksum(folder / "local.npy"),
        )
        (folder / "features.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        image.close()
        print(f"{row.sample_id}: {len(index)} spots extracted", flush=True)
