# DARF RF Fingerprinting

Device Analog/RF Fingerprint Classification.

## Goal

Build a robust neural network architecture that consistently classifies wireless transmitters using unique RF imperfections.

## Pipeline

Synthetic I/Q -> Dataset Loader -> CNN Classifier -> Metrics + Robustness Testing

## Installation

From the repository root, create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

If you are using PyCharm, select `.venv/bin/python` as the project interpreter after installing the dependencies.

## Week 1

Verify the toy signal generator:

```bash
python3 -m src.toy_generator
```

Verify the dataset loader:

```bash
python3 test_dataset.py
```

Train the baseline CNN and save metrics:

```bash
python3 -m src.training.train
```

The training command writes the Week 1 metrics file here:

```text
results/week1_toy_cnn/metrics.json
```

## Week 2

# ORACLE Dataset Quick Start Guide

## 📋 Overview

Three Python scripts to convert and use the ORACLE RF fingerprinting dataset:

| Script | Purpose |
|--------|---------|
| `oracle_dataset_runner.py` | Main entry point - converts SigMF files to numpy arrays |
| `src/datasets/oracle_converter.py` | Core conversion logic (called by runner) |
| `src/datasets/oracle_loader.py` | PyTorch DataLoader integration |
| `validate_oracle_dataset.py` | Verify and analyze converted dataset |

## 🚀 Quick Start

### 1. Convert Dataset (Only do this with new datasets)

```bash
python oracle_dataset_runner.py --test-loader
```

This will:
- Read all `.sigmf-meta` and `.sigmf-data` files
- Extract device IDs and create I/Q channel pairs
- Generate sliding windows (256 samples, 50% overlap)
- Save to `datasets/oracle/` (numpy format)
- Test the DataLoader

**Output**: `X.npy`, `y.npy`, `device_mapping.json`, `dataset_info.json`

### 2. Validate Dataset

```bash
python validate_oracle_dataset.py --report oracle_report.txt
```

Shows:
- Dataset shape and memory usage
- Class distribution
- Signal statistics (mean, std, min, max)
- Per-device characteristics

### 3. Use in Training

```python
from src.datasets.oracle_loader import load_oracle_dataset

# Load data
train_loader, test_loader, metadata = load_oracle_dataset(
    dataset_dir="./datasets/oracle",
    batch_size=32,
    split_ratio=0.8
)

# Train model
for epoch in range(10):
    for batch in train_loader:
        x = batch['x']              # Shape: (32, 2, 256)
        y = batch['y']              # Shape: (32,)
        
        # Forward pass through CNN
        pred = model(x)
        loss = criterion(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 📊 Data Format

### Input Format (SigMF)

Raw RF recordings in Signal Metadata Format:
- **Files**: `*.sigmf-meta` (JSON metadata) + `*.sigmf-data` (binary)
- **Data Type**: Complex float32 (I + jQ)
- **Sample Rate**: 5 MHz
- **Format**: `Demod_WiFi_cable_X310_<DEVICE>_IQ#<IMBALANCE>_run<RUN>.sigmf-*`

### Output Format (Numpy)

Training-ready arrays:
```python
X.shape = (num_samples, 2, 256)    # I/Q channels × time samples
y.shape = (num_samples,)            # Class labels
```

**Normalization**: Applied automatically in DataLoader (z-score per sample)

## 🔧 Customization

### Adjust Window Parameters

```bash
python oracle_dataset_runner.py \
    --window-size 512 \              # Longer windows
    --stride 256                      # More overlap
```

### Change Output Location

```bash
python oracle_dataset_runner.py \
    --output-dir /path/to/datasets/oracle
```

### Load with Custom Settings

```python
train_loader, test_loader, meta = load_oracle_dataset(
    dataset_dir="./datasets/oracle",
    split_ratio=0.7,                 # 70% train, 30% test
    batch_size=64,                   # Larger batches
    shuffle_train=True,
    num_workers=4                    # Parallel loading
)
```

## 📈 Dataset Statistics

Current ORACLE Dataset (dataset2):

```
├─ Total files: 32 SigMF pairs
├─ Samples per file: ~2.2M complex samples
├─ Unique devices: 2-3
├─ IQ imbalance levels: 20+
├─ Windows @ 256 samples, 50% overlap: ~2,500-5,000
└─ Memory (full dataset): ~200-400 MB
```

**Class Distribution** (after conversion):
```
Device 3123D76 (Transmitter 1): ~50%
Device ...    (Transmitter 2): ~50%
```

## ⚠️ Troubleshooting

### Issue: "ORACLE directory not found"
```
Solution: Ensure data is at:
src/datasets/ORACLE/dataset2/KRI-16IQImbalances-DemodulatedData/
```

### Issue: ".sigmf-data file not found"
```
Solution: Both .sigmf-meta and .sigmf-data must exist for each recording
```

### Issue: "Unsupported datatype cf32"
```
Solution: Only complex float32 is supported. Check SigMF metadata version.
```

### Issue: Out of memory during conversion
```
Solution 1: Increase window stride (fewer samples created)
Solution 2: Reduce window size
Solution 3: Process smaller subset of files
```

## 📚 File Structure After Conversion

```
darf-rf-fingerprinting/
├── oracle_dataset_runner.py              # Main script
├── validate_oracle_dataset.py            # Validation script
├── ORACLE_DATASET_README.md              # Detailed guide
└── src/datasets/
    ├── oracle_converter.py               # Core converter
    ├── oracle_loader.py                  # PyTorch loader
    ├── ORACLE/
    │   └── dataset2/
    │       └── KRI-16IQImbalances-DemodulatedData/
    │           ├── *.sigmf-meta          # Raw input files
    │           ├── *.sigmf-data
    │           └── ...
    └── (after conversion)
        └── oracle/
            ├── X.npy                     # Converted I/Q samples
            ├── y.npy                     # Class labels
            ├── device_mapping.json       # Device ID → class
            └── dataset_info.json         # Metadata
```

## 📖 References

- **ORACLE Dataset Paper**: [Sankhe et al., INFOCOM 2019](https://genesys-lab.org/oracle)
- **SigMF Format**: [Signal Metadata Format Specification](https://sigmf.io/)
- **USRP Hardware**: [Ettus Research USRP Documentation](https://www.ettus.com/product/)

---

**Questions?** Check `ORACLE_DATASET_README.md` in the./src/datasets directory for technical details.