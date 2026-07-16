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