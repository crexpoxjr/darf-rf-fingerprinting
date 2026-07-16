# DARF RF Fingerprinting

Device analog/RF fingerprint classification using ORACLE SigMF recordings.

## Goal

Build a reproducible RF fingerprinting pipeline that:

- converts ORACLE raw SigMF files to model-ready I/Q windows,
- uses a leakage-aware grouped split protocol,
- trains a 1D CNN, and
- saves complete research artifacts for each run.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Current ORACLE Pipeline

1. Convert raw SigMF recordings to `X.npy` / `y.npy` with per-window metadata.
2. Build grouped train/val/test partitions by source recording.
3. Train CNN from YAML config.
4. Save metrics and artifacts for reproducibility.

Main components:

- `oracle_dataset_runner.py`: conversion entry point
- `src/datasets/oracle_converter.py`: SigMF parse + windowing + split manifest
- `src/datasets/oracle_loader.py`: DataLoader and split-manifest-aware partitioning
- `src/training/train.py`: config-driven training + artifact generation
- `train_oracle_cnn.py`: compatibility wrapper (delegates to `src/training/train.py`)
- `validate_oracle_dataset.py`: converted dataset validation

## Dtype Note (Important)

Although ORACLE metadata may advertise `cf32`, these files are parsed using an inferred raw dtype and in this repository are read as `complex128` (based on file-size/sample-count checks). The resulting converted I/Q channels are therefore `float64`.

`validate_oracle_dataset.py` has been updated accordingly and now validates `X` as `float64`.

## Quick Start

### 1. Convert and test loader

```bash
python oracle_dataset_runner.py --test-loader
```

### 2. Validate conversion

```bash
python validate_oracle_dataset.py --report oracle_report.txt
```

### 3. Train ORACLE CNN (recommended)

```bash
python -m src.training.train --config configs/oracle_cnn.yaml
```

### 4. Backward-compatible training command

```bash
python train_oracle_cnn.py
```

This now runs the same config-driven ORACLE pipeline as step 3.

## Split Protocol

Configured in `configs/oracle_cnn.yaml`:

```yaml
dataset:
  split:
    protocol: grouped_by_source_file
    train: 0.7
    val: 0.1
    test: 0.2
```

Windows from the same source recording are assigned to only one partition to reduce leakage risk.

The explicit split is saved in:

- `datasets/oracle/split_manifest.json`
- copied to `results/oracle_cnn/split_manifest.json`

Each manifest entry includes fields such as dataset name, device ID, source file, run, imbalance, window start/end, window length, stride, label, and split.

## Config Guide

Use `configs/oracle_cnn.yaml` as the single source of truth for conversion + training.

### Class Cap

Set the class/device cap in `dataset.max_classes`:

```yaml
dataset:
  max_classes: 8
```

Important: keep `model.classes` equal to the effective dataset class count.

```yaml
model:
  classes: 8
```

If these differ, training can become inconsistent or silently underperform.

### Memory and Dataset Size Controls

For dataset1, use these controls to avoid OOM during conversion:

```yaml
dataset:
  path: src/datasets/ORACLE/dataset1/neu_m044q5210
  max_classes: 8
  max_windows_per_recording: 512
  output_dtype: float32
  max_dataset_gib: 8.0
```

- `max_windows_per_recording`: down-samples windows per SigMF file.
- `output_dtype`: `float32` uses less memory than `float64`.
- `max_dataset_gib`: fail-fast guard before oversized conversion.

### Split Settings

Use grouped split for leakage safety and class coverage:

```yaml
dataset:
  split:
    protocol: grouped_by_source_file
    train: 0.7
    val: 0.1
    test: 0.2
```

The converter writes `datasets/oracle/split_manifest.json` with per-window fields:
- `dataset_name`, `device_id`, `source_file`, `run`
- `window_start`, `window_end`, `window_length`, `stride`
- `label`, `split`
- ORACLE conditions when available: `imbalance`, `distance`, `distance_ft`

### CLI Overrides (Optional)

You can override config values when converting:

```bash
python oracle_dataset_runner.py \
  --config configs/oracle_cnn.yaml \
  --max-classes 8 \
  --max-windows-per-recording 512 \
  --output-dtype float32 \
  --max-dataset-gib 8.0
```

## Per-Run Artifacts

Each ORACLE training run writes to `results/oracle_cnn/`:

- `config.yaml`
- `metrics.json`
- `confusion_matrix.png`
- `confusion_matrix.csv`
- `training_curves.png`
- `split_manifest.json`
- `oracle_cnn.pt`

`metrics.json` includes:

- `accuracy`
- `macro_f1`
- `precision`
- `recall`
- `confusion_matrix`
- `classification_report`
- dataset metadata
- training metadata (epochs, batch size, learning rate, final loss)
- run metadata (`random_seed`, `git_commit`, device)

## Notes on Current Results

If your subset currently has only one device class, you may observe near-perfect metrics and a 1x1 confusion matrix. This is expected for single-class evaluation and should not be interpreted as strong multi-device fingerprinting performance.

## References

- ORACLE dataset: https://genesys-lab.org/oracle
- SigMF: https://sigmf.io/