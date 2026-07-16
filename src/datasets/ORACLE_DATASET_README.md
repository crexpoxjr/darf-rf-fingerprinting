# ORACLE Dataset Conversion Guide

This directory contains tools to convert the ORACLE SigMF-format RF dataset into training-ready numpy arrays for your CNN model.

## What is the ORACLE Dataset?

The **ORACLE dataset** (from INFOCOM 2019) contains WiFi transmissions collected via USRP radio receivers. Key features:

- **Device Fingerprinting**: IQ samples from controlled RF transmitters with varying hardware impairments
- **Format**: SigMF (Signal Metadata Format) - standard for RF signal recordings
- **Hardware**: Ettus USRP B210 receivers, X310 transmitters
- **Samples**: Complex I/Q pairs at 5 MHz sample rate
- **Controlled Impairments**: Intentional IQ imbalances enable fingerprinting
- **Parsing note**: The ORACLE SigMF metadata in this repository says `cf32`, but the provided binary `.sigmf-data` files are actually stored as complex128 values. Reading them as `complex64` corrupts the samples and produces the wrong sample count. The converter now infers the correct dtype from the file size and uses `complex128` for the ORACLE files in this project.

## Files

### Conversion Scripts

1. **`oracle_converter.py`** - Core conversion logic
   - Reads SigMF metadata and binary data
   - Extracts device IDs from filenames
   - Creates sliding windows from raw signals
   - Converts to I/Q channel format
   - Saves as numpy arrays

2. **`oracle_loader.py`** - PyTorch integration
   - `OracleRFDataset` class for PyTorch DataLoader
   - Automatic train/test splitting
   - Signal normalization
   - Device metadata tracking

3. **`../oracle_dataset_runner.py`** - Main entry point
   - Command-line tool to run conversion
   - Validation and testing

## Usage

### Step 1: Run Conversion

```bash
cd darf-rf-fingerprinting
python oracle_dataset_runner.py
```

Options:
```bash
python oracle_dataset_runner.py \
    --window-size 512 \          # Adjust window size
    --stride 256 \                # Adjust overlap
    --output-dir ./datasets/oracle \  # Custom output location
    --test-loader                 # Test DataLoader after conversion
```

### Step 2: Use in Training

```python
from src.datasets.oracle_loader import load_oracle_dataset

# Load dataset
train_loader, test_loader, metadata = load_oracle_dataset(
    dataset_dir="./datasets/oracle",
    split_ratio=0.8,
    batch_size=32
)

print(metadata)
# {
#   'num_samples': 2560,
#   'num_train': 2048,
#   'num_test': 512,
#   'num_classes': 2,
#   'device_mapping': {'3123D76': 0, '...': 1},
#   'window_size': 256,
#   'input_shape': (2, 256)
# }

# Use with your CNN model
for batch in train_loader:
    x = batch['x']              # Shape: (32, 2, 256)
    y = batch['y']              # Shape: (32,)
    device = batch['metadata']['device']  # Device ID
```

## Data Format

### Input (SigMF Files)

```
ORACLE/dataset2/KRI-16IQImbalances-DemodulatedData/
├── Demod_WiFi_cable_X310_3123D76_IQ#1_run1.sigmf-meta
├── Demod_WiFi_cable_X310_3123D76_IQ#1_run1.sigmf-data
├── Demod_WiFi_cable_X310_3123D76_IQ#2_run1.sigmf-meta
├── Demod_WiFi_cable_X310_3123D76_IQ#2_run1.sigmf-data
└── ...
```

**Filename format**: `Demod_WiFi_cable_X310_<DEVICE_ID>_IQ#<IMBALANCE>_run<RUN>.sigmf-{meta,data}`

- `DEVICE_ID`: Transmitter hardware identifier (e.g., `3123D76`)
- `IMBALANCE`: IQ imbalance level (0-32)
- `RUN`: Recording run number

### Output (Numpy Arrays)

```
datasets/oracle/
├── X.npy                   # Shape: (num_samples, 2, 256)
│                           # Samples × I/Q channels × time
├── y.npy                   # Shape: (num_samples,)
│                           # Class labels (0-indexed)
├── device_mapping.json     # {"device_id": class_idx, ...}
└── dataset_info.json       # Metadata
```

## Technical Details

### I/Q Channel Separation

Raw SigMF data contains complex samples: `z = I + jQ`

Important parsing note: the ORACLE `.sigmf-data` files in this project are read as `complex128` even though the SigMF metadata declares `cf32`. The byte size of the files corresponds to 16 bytes per sample (two 8-byte floating-point values), which matches `complex128`. Reading as `complex64` produces the wrong sample count and corrupts the waveform.

Conversion creates:
```
[I_channel, Q_channel]  # Shape: (2, window_size)
```

### Windowing

Sliding window extraction with configurable overlap:

```
Window size: 256 samples
Stride: 128 samples (50% overlap)

Signal:  [====|==== ==== ==== ====|==== ...]
Window1:  ^^^^  (samples 0-255)
Window2:       ^^^^  (samples 128-383)
Window3:              ^^^^  (samples 256-511)
```

### Normalization

Per-sample z-score normalization (applied in DataLoader):

```
x_norm = (x - mean(x)) / (std(x) + 1e-8)
```

## Statistics

For the current dataset:

```
ORACLE Dataset (dataset2)
├─ Number of files: 32 SigMF pairs (2.2M samples each)
├─ Unique devices: 2-3
├─ Imbalance levels: 20+
├─ Windows @ 256 samples: ~2,500-5,000 total
└─ Format: Complex float32 @ 5 MHz sampling rate
```

## Troubleshooting

### Error: "ORACLE directory not found"
Ensure the dataset is at: `src/datasets/ORACLE/dataset2/KRI-16IQImbalances-DemodulatedData`

### Error: "Data file not found"
The `.sigmf-data` file must be adjacent to the `.sigmf-meta` file with the same base name.

### Error: "Unsupported datatype"
Only `cf32` (complex float32) is currently supported. Verify SigMF metadata.

### Small number of windows
Increase `--window-size` or `--stride` parameters. Larger windows = fewer samples, smaller stride = more overlap.

## Next Steps

1. **Data Analysis**: Plot sample I/Q constellation diagrams
   ```python
   import matplotlib.pyplot as plt
   sample = X[0, 0] + 1j * X[0, 1]
   plt.scatter(np.real(sample), np.imag(sample), alpha=0.5)
   plt.show()
   ```

2. **Model Training**: Use with your CNN1D model
   ```python
   from src.models.cnn1d import RF_CNN
   model = RF_CNN()
   # Train on oracle_train_loader
   ```

3. **Augmentation**: Add signal processing augmentations
   - Time-shift windows
   - Add Gaussian noise
   - Phase rotation

## References

- ORACLE Dataset: [Sankhe et al., INFOCOM 2019](https://genesys-lab.org/oracle)
- SigMF Standard: [Signal Metadata Format](https://sigmf.io/)
- USRP Hardware: [Ettus Research](https://www.ettus.com/)
