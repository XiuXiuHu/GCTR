# Dataset interface

Copy `manifest.example.csv` to `manifest.csv` and replace the example rows with your frozen sample/patient split. Paths are relative to the manifest. Train, validation and test patient IDs must be disjoint. Manifest row order followed by within-section spot order defines the prediction order.

Supply `genes.csv` with one `gene_symbol` column in frozen panel order, and `gene_queries.npy` containing aligned, pretrained scGPT-derived queries with shape `[256,256]`. Do not select genes from validation or test expression.

These are decoder-space queries from the same pretrained DeepSpot-M checkpoint used for target-decoder initialization, not raw 512-dimensional scGPT embeddings or random gene vectors. Use the frozen query file from the prepared benchmark when reproducing that benchmark.

Each section folder contains:

| File | Shape / contents |
|---|---|
| `spot_index.csv` | Unique `spot_key`; optional `sample_id` and `patient_id`; `x_um,y_um` for context construction |
| `target.npy` | FP32 `[N,256,256]`, projected Midnight spatial tokens |
| `local.npy` | FP32 `[N,8,256]`, independently encoded local tiles, pooled after projection |
| `local_mask.npy` | Boolean `[N,8]` |
| `global_indices.npy` | Integer `[N,72]`, section-local row indices, padded with `-1` |
| `global_mask.npy` | Boolean `[N,72]`, exactly `global_indices >= 0` |
| `targets.npy` | FP32 `[N,256]`, required for training and evaluation |
| `supervision_mask.npy` | Boolean `[N,256]`, required for training and evaluation |

For each spot, local order is top-left, top, top-right, left, right, bottom-left, bottom, bottom-right. The center target is excluded. Global rows index the same section's target cache; global vectors are FP32 means of these target tokens. Global indices must exclude the target and contain no repeated valid rows.

Expression targets use **full-gene** raw-count library normalization to 10,000, then `log1p`, then panel extraction. Never normalize only the selected panel. A measured zero count remains valid. A missing gene has `NaN` target and mask `False`. Zero-library spots are retained with an entirely false supervision row. Spots with no valid image context are retained.

For morphology-only inference, omit `targets.npy` and `supervision_mask.npy`; use the `predict` command.

## Optional context construction

If global indices have not already been frozen, provide `candidate_valid.npy` (boolean `[N]`) and physical coordinates `x_um,y_um` in `spot_index.csv`, then run:

```bash
python main.py context --manifest data/manifest.csv
```

Candidates must be image-valid: target center in bounds, no target crop padding, and a white-pixel fraction below 0.95, where a white pixel has all RGB channels at least 240. This validity mask is derived from the image and geometry, never expression. The sampler excludes the target and nearest 18 eligible spots, partitions the remainder into four distance strata, and selects up to 18 per stratum. Its geometry seed is fixed at 0 and combined with the spot identity hash; the training seed does not resample context. Existing global files are never overwritten.

## Optional image feature extraction

Provide `image` and `mpp` in the manifest, `crop_boxes.npy` with integer XYXY boxes `[N,9,4]`, and the frozen `local_mask.npy`. Box 0 is the target; boxes 1-8 follow local order. Boxes must already be expressed in the final registered image coordinate frame, including any required flip. Registration and patient mapping are dataset-specific inputs, not inferred by this project.

Use target/local physical side 154.05 µm and total local field 462.15 µm. Native dimensions depend on MPP. Crops are white-padded at image boundaries, resized to 224×224 with bilinear interpolation, and passed through official Midnight preprocessing. The encoder and projection remain frozen in FP32. The extraction path uses batches of 32 tiles with an unpadded final batch.

```bash
pip install -r requirements-images.txt
python main.py extract --manifest data/manifest.csv --weights weights/DeepSpotM --device cuda
```

The model directory must contain the official `config.json`, `tokens.csv`, `model.safetensors`, and any upstream assets required by the checkpoint. Obtain weights and gene representations from the official source under their own terms. They are not included here. Feature extraction never overwrites existing feature files.
