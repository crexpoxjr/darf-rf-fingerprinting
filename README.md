# DARF RF Fingerprinting

Device analog/RF fingerprint classification using ORACLE SigMF recordings and WiSig pickle datasets.

## Goal

Build a reproducible RF fingerprinting pipeline that:

- converts ORACLE raw SigMF files or WiSig pickle payloads to model-ready I/Q windows,
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

## Current Pipeline

1. Convert raw RF recordings to `X.npy` / `y.npy` with per-window metadata.
2. Build grouped train/val/test partitions by source recording.
3. Train CNN from YAML config.
4. Save metrics and artifacts for reproducibility.

Main components:

- `oracle_dataset_runner.py`: conversion entry point
- `src/datasets/oracle_converter.py`: SigMF parse + windowing + split manifest
- `src/datasets/wisig_converter.py`: WiSig pickle parse + split manifest
- `src/datasets/oracle_loader.py`: DataLoader and split-manifest-aware partitioning
- `src/training/train.py`: config-driven training + artifact generation
- `src/evaluation/robustness_suite.py`: perturbation and domain-shift robustness evaluation
- `train_oracle_cnn.py`: compatibility wrapper (delegates to `src/training/train.py`)
- `validate_oracle_dataset.py`: converted dataset validation

## Dtype Note (Important)

Although ORACLE metadata may advertise `cf32`, these files are parsed using an inferred raw dtype and in this repository are read as `complex128` (based on file-size/sample-count checks). The resulting converted I/Q channels are therefore `float64`.

`validate_oracle_dataset.py` has been updated accordingly and now validates `X` as `float64`.

## Quick Start

### 1. Convert and test loader

```bash
python oracle_dataset_runner.py --test-loader --config configs/1dcnn.yaml
```

`oracle_dataset_runner.py` now auto-loads `configs/oracle_cnn.yaml` when present.
This means `dataset.path`, class cap, window cap, dtype, and size guard are all
applied by default without passing `--config` manually.

If you only see one device (often `3123D76`), you are likely converting dataset2.
To force dataset1 explicitly:

```bash
python oracle_dataset_runner.py \
  --oracle-dir src/datasets/ORACLE/dataset1/neu_m044q5210 \
  --test-loader
```

### 2. Validate conversion

```bash
python validate_oracle_dataset.py --report oracle_report.txt
```

### 3. Train ORACLE CNN (recommended)

```bash
python -m src.training.train --config configs/1dcnn.yaml
```

### 3b. Train on WiSig

```bash
python -m src.training.train --config configs/wisig_1dcnn.yaml
```

The training entry point now accepts `dataset.name: wisig` and will convert a
WiSig pickle payload into the same saved `X.npy` / `y.npy` / `split_manifest.json`
layout used by ORACLE runs.

### 5. Run robustness evaluation

```bash
python -m src.evaluation.robustness_suite \
  --results-dir results/wisig_1dcnn
```

The robustness suite evaluates:

- SNR sweep: clean, 30, 20, 10, 5, and 0 dB
- carrier-frequency offset: controlled normalized CFO values
- phase rotation: static phase bias and random per-sample rotation
- timing offset: integer shifts and fractional delays
- fading and multipath: Rayleigh, Rician, and a simple 3-tap channel
- cross-receiver split: retrain on one receiver subset and test on held-out receivers
- cross-day split: retrain on earlier capture days and test on later days when multiple days exist
- adversarial perturbation: FGSM and PGD with bounded I/Q perturbations

The command writes `robustness_report.json` into the selected run directory by default.
For WiSig `SingleDay.pkl`, cross-day evaluation is expected to skip because the converted
dataset currently contains only one capture day.

### 4. Backward-compatible training command

```bash
python train_oracle_cnn.py
```

This now runs the same config-driven ORACLE pipeline as step 3.

## Split Protocol

Configured in `configs`:

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

### CNN Types

So far, both a Small 1D CNN and a multi-channel 1D CNN is implemented. To use the 1D CNN Model change
dataset:
  ...
  channels: 2
  channel_mode: iq
  ...
model:
  ...
  name: cnn1d
  in_channels: 2
  ...

To use the 1D multi-channel CNN, change:
dataset:
  ...
  channels: 4
  channel_mode: iqmp
  ...
model:
  ...
  name: cnn1d_multi
  in_channels: 4
  ...

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
  output_dtype: float64
  max_dataset_gib: 8.0
```

- `max_windows_per_recording`: down-samples windows per SigMF file.
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

You can also override source path + geometry directly:

```bash
python oracle_dataset_runner.py \
  --oracle-dir src/datasets/ORACLE/dataset1/neu_m044q5210 \
  --window-size 256 \
  --stride 128
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