"""
ORACLE Dataset Converter

Converts ORACLE SigMF-format RF signal recordings into training-ready numpy arrays.
Handles complex I/Q data from USRP receivers and extracts device fingerprints.

The ORACLE dataset contains WiFi transmissions with controlled IQ imbalances,
enabling device fingerprinting based on hardware impairments.
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import re


class OracleConverter:
    """Convert ORACLE SigMF dataset to training-ready format."""

    def __init__(
        self,
        oracle_dir: Path,
        window_size: int = 256,
        stride: int = 128,
        min_samples: int = 512
    ):
        """
        Initialize converter.

        Args:
            oracle_dir: Path to ORACLE/dataset2/KRI-16IQImbalances-DemodulatedData
            window_size: Samples per window (default 256)
            stride: Samples between windows (default 128, 50% overlap)
            min_samples: Minimum samples required to create a window
        """
        self.oracle_dir = Path(oracle_dir)
        self.window_size = window_size
        self.stride = stride
        self.min_samples = min_samples

        if not self.oracle_dir.exists():
            raise ValueError(f"ORACLE directory not found: {oracle_dir}")

        self.sigmf_meta_files = sorted(
            self.oracle_dir.glob("*.sigmf-meta")
        )
        print(f"Found {len(self.sigmf_meta_files)} SigMF recordings")

    def parse_filename(self, filename: str) -> Dict:
        """
        Extract metadata from SigMF filename.

        Expected format:
        Demod_WiFi_cable_X310_<DEVICE_ID>_IQ#<IMBALANCE>_run<RUN>.sigmf-meta

        Returns:
            Dictionary with device_id, imbalance, run_number
        """
        # Example: Demod_WiFi_cable_X310_3123D76_IQ#1_run1.sigmf-meta
        match = re.search(
            r"X310_([A-F0-9]+)_IQ#(\d+)_run(\d+)",
            filename
        )

        if not match:
            raise ValueError(f"Cannot parse filename: {filename}")

        return {
            "device_id": match.group(1),
            "imbalance": int(match.group(2)),
            "run": int(match.group(3))
        }

    def read_sigmf_data(
        self,
        meta_file: Path
    ) -> Tuple[np.ndarray, Dict]:
        """
        Read SigMF metadata and data files.

        Args:
            meta_file: Path to .sigmf-meta file

        Returns:
            Tuple of (I/Q data array, metadata dict)
        """
        # Read metadata
        with open(meta_file, 'r') as f:
            meta = json.load(f)

        # Get data file path - use relative path
        data_file = meta_file.with_suffix('.sigmf-data')
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Extract key metadata
        global_meta = meta.get("_metadata", {}).get("global", {})
        sample_rate = global_meta.get("core:sample_rate", 5e6)
        datatype = global_meta.get("core:datatype", "cf32")

        # Get sample count from annotations
        annotations = meta.get("_metadata", {}).get("annotations", [{}])
        sample_count = annotations[0].get("core:sample_count", None)

        # Read binary data (cf32 = complex float32)
        if datatype == "cf32":
            iq_data = np.fromfile(
                data_file,
                dtype=np.complex64
            )
        else:
            raise ValueError(f"Unsupported datatype: {datatype}")

        if sample_count and len(iq_data) != sample_count:
            print(f"Warning: expected {sample_count} samples, got {len(iq_data)}")

        # Extract filename metadata
        filename_meta = self.parse_filename(meta_file.name)

        return iq_data, {
            "sample_rate": sample_rate,
            "sample_count": len(iq_data),
            "datatype": datatype,
            **filename_meta,
            "file": meta_file.name
        }

    def create_windows(
        self,
        signal: np.ndarray
    ) -> List[np.ndarray]:
        """
        Create sliding windows from signal.

        Args:
            signal: 1D complex array of I/Q samples

        Returns:
            List of windowed signals (window_size,)
        """
        windows = []
        num_windows = (len(signal) - self.window_size) // self.stride + 1

        for i in range(num_windows):
            start = i * self.stride
            end = start + self.window_size
            windows.append(signal[start:end])

        return windows

    def signal_to_channels(
        self,
        signal: np.ndarray
    ) -> np.ndarray:
        """
        Convert complex I/Q signal to multi-channel format.

        Creates separate I and Q channels for network input.

        Args:
            signal: 1D complex array

        Returns:
            Array of shape (2, window_size) with [I, Q] channels
        """
        i_channel = np.real(signal)
        q_channel = np.imag(signal)

        return np.stack([i_channel, q_channel], axis=0)

    def convert_dataset(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Convert entire ORACLE dataset to training format.

        Returns:
            Tuple of (X, y, device_mapping)
                X: shape (num_samples, 2, window_size) - I/Q channels
                y: shape (num_samples,) - device class labels (0-indexed)
                device_mapping: dict mapping device_id to class index
        """
        all_signals = []
        all_labels = []
        device_to_label = {}
        next_label = 0

        print("\nProcessing ORACLE dataset...")

        for meta_file in self.sigmf_meta_files:
            try:
                iq_signal, meta = self.read_sigmf_data(meta_file)

                device_id = meta["device_id"]

                # Assign device label if new
                if device_id not in device_to_label:
                    device_to_label[device_id] = next_label
                    print(f"  Device {device_id} -> class {next_label}")
                    next_label += 1

                label = device_to_label[device_id]

                # Create windows
                windows = self.create_windows(iq_signal)

                for window in windows:
                    # Convert to I/Q channels
                    channels = self.signal_to_channels(window)
                    all_signals.append(channels)
                    all_labels.append(label)

                print(f"  ✓ {meta_file.name}: {len(windows)} windows")

            except Exception as e:
                print(f"  ✗ Error processing {meta_file.name}: {e}")
                continue

        # Stack into arrays
        X = np.stack(all_signals, axis=0)
        y = np.array(all_labels)

        print(f"\nDataset created:")
        print(f"  X shape: {X.shape} (samples, channels, time)")
        print(f"  y shape: {y.shape}")
        print(f"  Unique devices: {len(device_to_label)}")
        print(f"  Device mapping: {device_to_label}")

        return X, y, device_to_label

    def save_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        device_mapping: Dict,
        output_dir: Path
    ) -> None:
        """
        Save converted dataset to numpy files.

        Args:
            X: Signal array
            y: Labels array
            device_mapping: Device ID to class mapping
            output_dir: Directory to save files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save arrays
        np.save(output_dir / "X.npy", X)
        np.save(output_dir / "y.npy", y)

        # Save metadata
        with open(output_dir / "device_mapping.json", 'w') as f:
            json.dump(device_mapping, f, indent=2)

        # Save info
        info = {
            "X_shape": X.shape,
            "y_shape": y.shape,
            "num_devices": len(device_mapping),
            "device_mapping": device_mapping,
            "window_size": self.window_size,
            "stride": self.stride
        }

        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)

        print(f"\n✓ Dataset saved to {output_dir}")
        print(f"  - X.npy: {X.shape}")
        print(f"  - y.npy: {y.shape}")
        print(f"  - device_mapping.json")
        print(f"  - dataset_info.json")


def main():
    """Example usage."""
    # Path to ORACLE dataset
    oracle_path = (
        Path(__file__).parent /
        "ORACLE/dataset2/KRI-16IQImbalances-DemodulatedData"
    )

    # Convert dataset
    converter = OracleConverter(
        oracle_dir=oracle_path,
        window_size=256,
        stride=128
    )

    X, y, device_mapping = converter.convert_dataset()

    # Save to output
    output_dir = Path(__file__).parent / "../../datasets/oracle"
    converter.save_dataset(X, y, device_mapping, output_dir)

    print("\n✓ Conversion complete! Ready for training.")


if __name__ == "__main__":
    main()
