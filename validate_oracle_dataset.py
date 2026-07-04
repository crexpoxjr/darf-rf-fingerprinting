#!/usr/bin/env python3
"""
Validate and visualize ORACLE Dataset conversions

Shows signal properties, class distribution, and sample visualizations.
"""

import json
import numpy as np
from pathlib import Path
import argparse
import sys
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def load_dataset(dataset_dir: Path) -> tuple:
    """Load converted dataset."""
    X = np.load(dataset_dir / "X.npy")
    y = np.load(dataset_dir / "y.npy")
    
    with open(dataset_dir / "device_mapping.json", 'r') as f:
        device_mapping = json.load(f)
    
    with open(dataset_dir / "dataset_info.json", 'r') as f:
        info = json.load(f)
    
    return X, y, device_mapping, info


def validate_dataset(X: np.ndarray, y: np.ndarray, device_mapping: Dict) -> bool:
    """Check dataset integrity."""
    print("\n[Validation]")
    print("=" * 60)
    
    errors = []
    
    # Check shapes
    if X.shape[0] != y.shape[0]:
        errors.append(f"Shape mismatch: X({X.shape[0]}) vs y({y.shape[0]})")
    
    if X.shape[1] != 2:
        errors.append(f"Expected 2 channels, got {X.shape[1]}")
    
    # Check data types
    if X.dtype != np.float32:
        errors.append(f"Expected float32, got {X.dtype}")
    
    if y.dtype not in [np.int32, np.int64, int]:
        errors.append(f"Expected integer labels, got {y.dtype}")
    
    # Check label range
    unique_labels = np.unique(y)
    expected_labels = set(range(len(device_mapping)))
    if set(unique_labels) != expected_labels:
        errors.append(f"Label mismatch. Expected {expected_labels}, got {set(unique_labels)}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(X)):
        errors.append("Found NaN values in X")
    
    if np.any(np.isinf(X)):
        errors.append("Found Inf values in X")
    
    if errors:
        print("✗ Validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✓ All validation checks passed")
        return True


def print_statistics(X: np.ndarray, y: np.ndarray, device_mapping: Dict):
    """Print dataset statistics."""
    print("\n[Dataset Statistics]")
    print("=" * 60)
    
    print(f"\nShape and Size:")
    print(f"  X shape: {X.shape} (samples, channels, time)")
    print(f"  y shape: {y.shape}")
    print(f"  Total samples: {len(X):,}")
    print(f"  Memory: {X.nbytes / 1024 / 1024:.1f} MB")
    
    print(f"\nChannels:")
    print(f"  Count: 2 (I/Q)")
    print(f"  Time steps: {X.shape[2]}")
    
    print(f"\nDevices (Classes):")
    print(f"  Total: {len(device_mapping)}")
    for device_id, class_idx in sorted(device_mapping.items(), key=lambda x: x[1]):
        count = np.sum(y == class_idx)
        pct = 100 * count / len(y)
        print(f"    Class {class_idx}: {device_id} ({count:4d} samples, {pct:5.1f}%)")
    
    print(f"\nSignal Statistics (all samples):")
    print(f"  I channel - Mean: {X[:, 0, :].mean():8.5f}, Std: {X[:, 0, :].std():8.5f}")
    print(f"  Q channel - Mean: {X[:, 1, :].mean():8.5f}, Std: {X[:, 1, :].std():8.5f}")
    print(f"  I channel - Min: {X[:, 0, :].min():8.5f}, Max: {X[:, 0, :].max():8.5f}")
    print(f"  Q channel - Min: {X[:, 1, :].min():8.5f}, Max: {X[:, 1, :].max():8.5f}")


def print_sample_info(X: np.ndarray, y: np.ndarray, device_mapping: Dict, num_samples: int = 5):
    """Print info about specific samples."""
    print(f"\n[Sample Info (first {num_samples} samples)]")
    print("=" * 60)
    
    id_to_device = {v: k for k, v in device_mapping.items()}
    
    print(f"\n{'Idx':<5} {'Class':<7} {'Device':<10} {'I-Mean':<10} {'Q-Mean':<10} {'Power':<10}")
    print("-" * 60)
    
    for i in range(min(num_samples, len(X))):
        i_channel = X[i, 0, :]
        q_channel = X[i, 1, :]
        power = np.sqrt(i_channel**2 + q_channel**2).mean()
        
        device_id = id_to_device.get(int(y[i]), "?")
        
        print(f"{i:<5} {int(y[i]):<7} {device_id:<10} {i_channel.mean():<10.5f} {q_channel.mean():<10.5f} {power:<10.5f}")


def analyze_signal_characteristics(X: np.ndarray, y: np.ndarray, device_mapping: Dict):
    """Analyze per-device signal characteristics."""
    print(f"\n[Per-Device Signal Characteristics]")
    print("=" * 60)
    
    id_to_device = {v: k for k, v in device_mapping.items()}
    
    print(f"\n{'Device':<12} {'Class':<7} {'Samples':<10} {'Avg Power':<12} {'Avg Energy':<12} {'Peak Power':<12}")
    print("-" * 65)
    
    for class_idx in sorted(device_mapping.values()):
        mask = y == class_idx
        class_data = X[mask]
        
        # Calculate metrics
        power = np.sqrt(class_data[:, 0, :]**2 + class_data[:, 1, :]**2)
        avg_power = power.mean()
        avg_energy = np.sum(power**2, axis=1).mean()
        peak_power = power.max()
        
        device_id = id_to_device[class_idx]
        
        print(f"{device_id:<12} {class_idx:<7} {np.sum(mask):<10} {avg_power:<12.5f} {avg_energy:<12.5f} {peak_power:<12.5f}")


def save_report(
    X: np.ndarray,
    y: np.ndarray,
    device_mapping: Dict,
    info: Dict,
    output_file: Path
):
    """Save analysis report to file."""
    with open(output_file, 'w') as f:
        f.write("ORACLE Dataset Validation Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Generated: {np.datetime64('today')}\n\n")
        
        f.write("Dataset Structure\n")
        f.write("-" * 70 + "\n")
        f.write(f"X shape: {X.shape}\n")
        f.write(f"y shape: {y.shape}\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Total memory: {X.nbytes / 1024 / 1024:.1f} MB\n\n")
        
        f.write("Configuration\n")
        f.write("-" * 70 + "\n")
        for key, value in info.items():
            if key != "device_mapping":
                f.write(f"{key}: {value}\n")
        f.write(f"device_mapping: {device_mapping}\n\n")
        
        f.write("Class Distribution\n")
        f.write("-" * 70 + "\n")
        id_to_device = {v: k for k, v in device_mapping.items()}
        for device_id, class_idx in sorted(device_mapping.items(), key=lambda x: x[1]):
            count = np.sum(y == class_idx)
            pct = 100 * count / len(y)
            f.write(f"Class {class_idx} ({device_id}): {count} samples ({pct:.1f}%)\n")
        
        f.write("\nSignal Statistics\n")
        f.write("-" * 70 + "\n")
        f.write(f"I channel - Mean: {X[:, 0, :].mean():.5f}, Std: {X[:, 0, :].std():.5f}\n")
        f.write(f"Q channel - Mean: {X[:, 1, :].mean():.5f}, Std: {X[:, 1, :].std():.5f}\n")
        f.write(f"I channel - Range: [{X[:, 0, :].min():.5f}, {X[:, 0, :].max():.5f}]\n")
        f.write(f"Q channel - Range: [{X[:, 1, :].min():.5f}, {X[:, 1, :].max():.5f}]\n")
    
    print(f"\n✓ Report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and analyze converted ORACLE dataset"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset directory (default: ./datasets/oracle)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples to show details for (default: 5)"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Save report to file"
    )
    
    args = parser.parse_args()
    
    # Determine dataset path
    project_root = Path(__file__).parent
    dataset_dir = args.dataset_dir or (project_root / "datasets/oracle")
    
    print("=" * 70)
    print("ORACLE Dataset Validation")
    print("=" * 70)
    print(f"\nDataset directory: {dataset_dir}")
    
    # Check if dataset exists
    if not dataset_dir.exists():
        print(f"\n✗ Dataset directory not found: {dataset_dir}")
        print("\nPlease run oracle_dataset_runner.py first to convert the dataset")
        sys.exit(1)
    
    # Load dataset
    try:
        X, y, device_mapping, info = load_dataset(dataset_dir)
    except Exception as e:
        print(f"\n✗ Failed to load dataset: {e}")
        sys.exit(1)
    
    # Run validation
    if not validate_dataset(X, y, device_mapping):
        sys.exit(1)
    
    # Print statistics
    print_statistics(X, y, device_mapping)
    print_sample_info(X, y, device_mapping, args.samples)
    analyze_signal_characteristics(X, y, device_mapping)
    
    # Save report if requested
    if args.report:
        save_report(X, y, device_mapping, info, args.report)
    
    print("\n" + "=" * 70)
    print("✓ Validation Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
