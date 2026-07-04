#!/usr/bin/env python3
"""
Convert and validate ORACLE Dataset

Usage:
    python oracle_dataset_runner.py [--output-dir PATH]
"""

import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.oracle_converter import OracleConverter
from src.datasets.oracle_loader import load_oracle_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert ORACLE SigMF dataset to training format"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ./datasets/oracle)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=256,
        help="Window size in samples (default: 256)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Stride between windows (default: 128)"
    )
    parser.add_argument(
        "--test-loader",
        action="store_true",
        help="Test loading dataset after conversion"
    )

    args = parser.parse_args()

    # Determine paths
    project_root = Path(__file__).parent
    oracle_data = (
        project_root / 
        "src/datasets/ORACLE/dataset2/KRI-16IQImbalances-DemodulatedData"
    )
    output_dir = args.output_dir or (project_root / "datasets/oracle")

    print("=" * 70)
    print("ORACLE Dataset Conversion")
    print("=" * 70)

    # Verify source data exists
    if not oracle_data.exists():
        print(f"\n✗ ERROR: ORACLE data not found at {oracle_data}")
        print("\nPlease ensure the ORACLE dataset is located at:")
        print(f"  {oracle_data}")
        sys.exit(1)

    print(f"\nInput:  {oracle_data}")
    print(f"Output: {output_dir}")
    print(f"Window size: {args.window_size} samples")
    print(f"Stride: {args.stride} samples")

    # Convert dataset
    try:
        print("\n[1/3] Initializing converter...")
        converter = OracleConverter(
            oracle_dir=oracle_data,
            window_size=args.window_size,
            stride=args.stride
        )

        print("\n[2/3] Converting dataset...")
        X, y, device_mapping = converter.convert_dataset()

        print("\n[3/3] Saving converted data...")
        converter.save_dataset(X, y, device_mapping, output_dir)

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test loader if requested
    if args.test_loader:
        print("\n" + "=" * 70)
        print("Testing DataLoader")
        print("=" * 70)

        try:
            train_loader, test_loader, metadata = load_oracle_dataset(
                output_dir,
                split_ratio=0.8,
                batch_size=32
            )

            print("\n✓ DataLoader test successful!")
            print(f"\nDataset Info:")
            for key, value in metadata.items():
                if key != "device_mapping":
                    print(f"  {key}: {value}")

            print(f"\nDevice Mapping:")
            for device_id, class_idx in metadata["device_mapping"].items():
                print(f"  {device_id} -> class {class_idx}")

            # Show sample
            print("\n✓ Sample batch shapes:")
            batch = next(iter(train_loader))
            print(f"  x: {batch['x'].shape}")
            print(f"  y: {batch['y'].shape}")

        except Exception as e:
            print(f"\n✗ DataLoader test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\n" + "=" * 70)
    print("✓ Complete!")
    print("=" * 70)
    print(f"\nYou can now train your model using:")
    print(f"  from src.datasets.oracle_loader import load_oracle_dataset")
    print(f"  train_loader, test_loader, meta = load_oracle_dataset('{output_dir}')")


if __name__ == "__main__":
    main()
