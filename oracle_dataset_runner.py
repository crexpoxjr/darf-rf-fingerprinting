#!/usr/bin/env python3
"""
Convert and validate ORACLE Dataset

Usage:
    python oracle_dataset_runner.py [--output-dir PATH]
"""

import argparse
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
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config (e.g., configs/1dcnn.yaml) to read dataset.path"
    )
    parser.add_argument(
        "--oracle-dir",
        type=Path,
        default=None,
        help="Raw ORACLE root directory. Overrides path from --config/default"
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
        "--max-classes",
        type=int,
        default=None,
        help="Optional cap on the number of devices/classes to convert"
    )
    parser.add_argument(
        "--max-windows-per-recording",
        type=int,
        default=None,
        help="Optional cap on windows emitted per SigMF recording"
    )
    parser.add_argument(
        "--output-dtype",
        type=str,
        default="float64",
        help="Stored dtype for converted I/Q channels (default: float64)"
    )
    parser.add_argument(
        "--max-dataset-gib",
        type=float,
        default=8.0,
        help="Abort if estimated X size exceeds this and no window cap is set"
    )
    parser.add_argument(
        "--test-loader",
        action="store_true",
        help="Test loading dataset after conversion"
    )

    args = parser.parse_args()

    # Determine paths
    project_root = Path(__file__).parent
    default_config_path = project_root / "configs" / "oracle_cnn.yaml"

    selected_config_path = args.config
    if selected_config_path is None and default_config_path.exists():
        selected_config_path = default_config_path

    config_oracle_path = None
    config_window_size = None
    config_stride = None
    config_model_classes = None
    config_dataset_max_classes = None
    config_dataset_max_windows = None
    config_dataset_output_dtype = None
    config_dataset_max_gib = None
    if selected_config_path is not None:
        import yaml

        config_path = selected_config_path if selected_config_path.is_absolute() else project_root / selected_config_path
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        dataset_cfg = config.get("dataset", {})
        config_oracle_path = dataset_cfg.get("path")
        config_window_size = dataset_cfg.get("window_length")
        config_stride = dataset_cfg.get("stride")
        config_model_classes = config.get("model", {}).get("classes")
        config_dataset_max_classes = dataset_cfg.get("max_classes")
        config_dataset_max_windows = dataset_cfg.get("max_windows_per_recording")
        config_dataset_output_dtype = dataset_cfg.get("output_dtype")
        config_dataset_max_gib = dataset_cfg.get("max_dataset_gib")

    oracle_data = args.oracle_dir or config_oracle_path or (
        "src/datasets/ORACLE/dataset2/KRI-16IQImbalances-DemodulatedData"
    )
    oracle_data = Path(oracle_data)
    if not oracle_data.is_absolute():
        oracle_data = project_root / oracle_data

    effective_max_classes = args.max_classes
    if effective_max_classes is None:
        if config_dataset_max_classes is not None:
            effective_max_classes = int(config_dataset_max_classes)
        elif config_model_classes is not None:
            effective_max_classes = int(config_model_classes)

    effective_max_windows = args.max_windows_per_recording
    if effective_max_windows is None and config_dataset_max_windows is not None:
        effective_max_windows = int(config_dataset_max_windows)

    effective_output_dtype = args.output_dtype
    if effective_output_dtype == "float32" and config_dataset_output_dtype is not None:
        effective_output_dtype = str(config_dataset_output_dtype)

    effective_max_dataset_gib = args.max_dataset_gib
    if config_dataset_max_gib is not None and args.max_dataset_gib == 8.0:
        effective_max_dataset_gib = float(config_dataset_max_gib)

    effective_window_size = args.window_size
    if config_window_size is not None and args.window_size == 256:
        effective_window_size = int(config_window_size)

    effective_stride = args.stride
    if config_stride is not None and args.stride == 128:
        effective_stride = int(config_stride)
    elif config_stride is None and config_window_size is not None and args.stride == 128:
        effective_stride = max(1, int(config_window_size) // 2)

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
    if selected_config_path is not None:
        print(f"Config: {config_path}")
    print(f"Window size: {effective_window_size} samples")
    print(f"Stride: {effective_stride} samples")
    if effective_max_classes is not None:
        print(f"Max classes: {effective_max_classes}")
    if effective_max_windows is not None:
        print(f"Max windows/recording: {effective_max_windows}")
    print(f"Output dtype: {effective_output_dtype}")

    # Convert dataset
    try:
        print("\n[1/3] Initializing converter...")
        converter = OracleConverter(
            oracle_dir=oracle_data,
            window_size=effective_window_size,
            stride=effective_stride,
            max_classes=effective_max_classes,
            max_windows_per_recording=effective_max_windows,
            output_dtype=effective_output_dtype,
            max_dataset_gib=effective_max_dataset_gib,
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
