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
        min_samples: int = 512,
        max_classes: int | None = 8,
        max_windows_per_recording: int | None = None,
        output_dtype: str = "float32",
        max_dataset_gib: float = 8.0,
    ):
        """
        Initialize converter.

        Args:
            oracle_dir: Path to ORACLE/dataset2/KRI-16IQImbalances-DemodulatedData
            window_size: Samples per window (default 256)
            stride: Samples between windows (default 128, 50% overlap)
            min_samples: Minimum samples required to create a window
            max_classes: Optional cap on the number of devices/classes to load
            max_windows_per_recording: Optional cap on windows emitted per file
            output_dtype: Stored dtype for converted I/Q channels
            max_dataset_gib: Abort if estimated X size exceeds this and no window cap is set
        """
        self.oracle_dir = Path(oracle_dir)
        self.window_size = window_size
        self.stride = stride
        self.min_samples = min_samples
        self.max_classes = max_classes
        self.max_windows_per_recording = max_windows_per_recording
        self.output_dtype = np.dtype(output_dtype)
        self.max_dataset_gib = max_dataset_gib

        if not self.oracle_dir.exists():
            raise ValueError(f"ORACLE directory not found: {oracle_dir}")

        # dataset1 is nested (e.g., .../KRI-16Devices-RawData/<distance>/*.sigmf-meta),
        # so we always discover SigMF metadata recursively.
        self.sigmf_meta_files = sorted(
            self.oracle_dir.rglob("*.sigmf-meta")
        )
        print(f"Found {len(self.sigmf_meta_files)} SigMF recordings")

    def parse_filename(self, filename: str) -> Dict:
        """
        Extract metadata from SigMF filename.

        Supported formats:
        - dataset2: Demod_WiFi_cable_X310_<DEVICE_ID>_IQ#<IMBALANCE>_run<RUN>.sigmf-meta
        - dataset1: WiFi_air_X310_<DEVICE_ID>_<DISTANCE>_run<RUN>.sigmf-meta

        Returns:
            Dictionary with device_id, imbalance, run_number
        """
        # dataset2 example: Demod_WiFi_cable_X310_3123D76_IQ#1_run1.sigmf-meta
        match = re.search(
            r"X310_([A-F0-9]+)_IQ#(\d+)_run(\d+)",
            filename
        )

        if match:
            return {
                "device_id": match.group(1),
                "imbalance": int(match.group(2)),
                "run": int(match.group(3))
            }

        # dataset1 example: WiFi_air_X310_3123D76_44ft_run2.sigmf-meta
        dataset1_match = re.search(
            r"X310_([A-F0-9]+)_([^_]+)_run(\d+)",
            filename
        )
        if dataset1_match:
            distance_token = dataset1_match.group(2)
            distance_ft = None
            distance_match = re.search(r"(\d+)ft", distance_token)
            if distance_match:
                distance_ft = int(distance_match.group(1))

            return {
                "device_id": dataset1_match.group(1),
                "distance": distance_token,
                "distance_ft": distance_ft,
                "imbalance": None,
                "run": int(dataset1_match.group(3))
            }

        raise ValueError(f"Cannot parse filename: {filename}")

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

        # The ORACLE SigMF files in this repository advertise cf32, but the
        # binary payloads are actually stored as complex128 values (16 bytes per
        # sample). Reading them as complex64 corrupts the data by producing the
        # wrong number of samples. We infer the correct dtype from the file size
        # when possible and fall back to complex128 for ORACLE data.
        inferred_dtype = np.complex128
        if sample_count is not None:
            file_size = data_file.stat().st_size
            if file_size % sample_count == 0:
                bytes_per_sample = file_size // sample_count
                if bytes_per_sample == 8:
                    inferred_dtype = np.complex64
                elif bytes_per_sample == 16:
                    inferred_dtype = np.complex128

        iq_data = np.fromfile(data_file, dtype=inferred_dtype)

        if sample_count and len(iq_data) != sample_count:
            print(
                f"Warning: expected {sample_count} samples from metadata, "
                f"got {len(iq_data)} with dtype {inferred_dtype}"
            )

        # Extract filename metadata
        filename_meta = self.parse_filename(meta_file.name)

        relative_path = str(meta_file.relative_to(self.oracle_dir))

        return iq_data, {
            "sample_rate": sample_rate,
            "sample_count": len(iq_data),
            "datatype": datatype,
            "dtype_used": str(inferred_dtype),
            **filename_meta,
            "file": relative_path
        }

    def create_windows(
        self,
        signal: np.ndarray
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        Create sliding windows from signal.

        Args:
            signal: 1D complex array of I/Q samples

        Returns:
            List of (window, start, end) tuples
        """
        windows = []
        num_windows = (len(signal) - self.window_size) // self.stride + 1

        for i in range(num_windows):
            start = i * self.stride
            end = start + self.window_size
            windows.append((signal[start:end], start, end))

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

    def convert_dataset(
        self,
        split_config: Dict | None = None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
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
        window_metadata = []
        device_to_label = {}
        skipped_devices = set()
        next_label = 0

        # First pass: estimate size to avoid unpredictable OOM when no window cap is set.
        if self.max_windows_per_recording is None:
            total_windows_est = 0
            for meta_file in self.sigmf_meta_files:
                try:
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)
                    annotations = meta.get("_metadata", {}).get("annotations", [{}])
                    sample_count = annotations[0].get("core:sample_count")
                    if sample_count is None or sample_count < self.window_size:
                        continue
                    total_windows_est += (sample_count - self.window_size) // self.stride + 1
                except Exception:
                    continue

            estimated_bytes = total_windows_est * 2 * self.window_size * self.output_dtype.itemsize
            estimated_gib = estimated_bytes / (1024 ** 3)
            if estimated_gib > self.max_dataset_gib:
                raise MemoryError(
                    "Estimated converted dataset is too large for current settings "
                    f"({estimated_gib:.2f} GiB > {self.max_dataset_gib:.2f} GiB). "
                    "Set max_windows_per_recording, increase stride, or lower max_classes."
                )

        print("\nProcessing ORACLE dataset...")

        for meta_file in self.sigmf_meta_files:
            try:
                iq_signal, meta = self.read_sigmf_data(meta_file)

                device_id = meta["device_id"]

                # Assign device label if new
                if device_id not in device_to_label:
                    if self.max_classes is not None and next_label >= self.max_classes:
                        skipped_devices.add(device_id)
                        continue

                    device_to_label[device_id] = next_label
                    print(f"  Device {device_id} -> class {next_label}")
                    next_label += 1

                label = device_to_label[device_id]

                # Create windows
                windows = self.create_windows(iq_signal)
                if self.max_windows_per_recording is not None and len(windows) > self.max_windows_per_recording:
                    selected_idx = np.linspace(
                        0,
                        len(windows) - 1,
                        num=self.max_windows_per_recording,
                        dtype=np.int64,
                    )
                    windows = [windows[i] for i in np.unique(selected_idx)]

                for window, start, end in windows:
                    # Convert to I/Q channels
                    channels = self.signal_to_channels(window).astype(self.output_dtype, copy=False)
                    all_signals.append(channels)
                    all_labels.append(label)
                    window_metadata.append({
                        "dataset_name": "oracle",
                        "device_id": device_id,
                        "source_file": meta["file"],
                        "run": meta["run"],
                        "imbalance": meta["imbalance"],
                        "distance": meta.get("distance"),
                        "distance_ft": meta.get("distance_ft"),
                        "window_start": int(start),
                        "window_end": int(end),
                        "window_length": self.window_size,
                        "stride": self.stride,
                        "label": int(label),
                        "split": None,
                    })

                print(f"  ✓ {meta['file']}: {len(windows)} windows")

            except Exception as e:
                print(f"  ✗ Error processing {meta_file.name}: {e}")
                continue

        # Stack into arrays
        X = np.stack(all_signals, axis=0)
        y = np.array(all_labels)
        self.window_metadata = self.assign_window_splits(
            window_metadata,
            split_config=split_config,
            seed=seed,
        )

        print(f"\nDataset created:")
        print(f"  X shape: {X.shape} (samples, channels, time)")
        print(f"  y shape: {y.shape}")
        print(f"  Unique devices: {len(device_to_label)}")
        print(f"  Device mapping: {device_to_label}")
        if skipped_devices:
            print(f"  Skipped devices after class cap ({self.max_classes}): {sorted(skipped_devices)}")

        return X, y, device_to_label

    def assign_window_splits(self, window_metadata: List[Dict], split_config: Dict | None = None, seed: int = 42) -> List[Dict]:
        """Assign grouped, class-stratified splits to avoid source leakage and class dropouts."""
        if not window_metadata:
            return []

        split_config = split_config or {
            "protocol": "grouped_by_source_file",
            "train": 0.7,
            "val": 0.1,
            "test": 0.2,
        }
        protocol = split_config.get("protocol", "grouped_by_source_file")
        if protocol not in {"grouped_by_source_file", "grouped_stratified_by_label"}:
            return window_metadata

        train_ratio = float(split_config.get("train", 0.7))
        val_ratio = float(split_config.get("val", 0.1))
        test_ratio = float(split_config.get("test", 0.2))
        total = train_ratio + val_ratio + test_ratio
        if total <= 0:
            raise ValueError("Split ratios must sum to a positive value")

        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

        # Build one label per source file; each source should map to one device label.
        source_to_label = {}
        for record in window_metadata:
            source = record["source_file"]
            label = int(record["label"])
            existing = source_to_label.get(source)
            if existing is None:
                source_to_label[source] = label
            elif existing != label:
                raise ValueError(f"Source file {source} maps to multiple labels: {existing}, {label}")

        label_to_sources: Dict[int, List[str]] = {}
        for source, label in source_to_label.items():
            label_to_sources.setdefault(label, []).append(source)

        rng = np.random.default_rng(seed)
        split_for_source = {}

        # Stratify by device label while keeping whole source files together.
        for label, sources in sorted(label_to_sources.items()):
            shuffled = list(rng.permutation(sorted(sources)))
            n = len(shuffled)

            if n == 1:
                train_n, val_n, test_n = 1, 0, 0
            elif n == 2:
                train_n, val_n, test_n = 1, 0, 1
            else:
                train_n = max(1, int(np.floor(n * train_ratio)))
                val_n = int(np.floor(n * val_ratio)) if val_ratio > 0 else 0
                test_n = n - train_n - val_n

                # Keep at least one eval source for labels with n >= 2.
                if test_n < 1:
                    test_n = 1
                    if val_n > 0:
                        val_n -= 1
                    else:
                        train_n = max(1, train_n - 1)

                # If val split is requested and n >= 3, enforce at least one
                # source in val by borrowing from train when needed.
                if val_ratio > 0 and n >= 3 and val_n == 0:
                    if train_n > 1:
                        train_n -= 1
                        val_n = 1
                    elif test_n > 1:
                        test_n -= 1
                        val_n = 1

                # Final consistency guard.
                while train_n + val_n + test_n > n:
                    if val_n > 0:
                        val_n -= 1
                    elif train_n > 1:
                        train_n -= 1
                    else:
                        test_n -= 1

            train_sources = shuffled[:train_n]
            val_sources = shuffled[train_n:train_n + val_n]
            test_sources = shuffled[train_n + val_n:train_n + val_n + test_n]

            for source in train_sources:
                split_for_source[source] = "train"
            for source in val_sources:
                split_for_source[source] = "val"
            for source in test_sources:
                split_for_source[source] = "test"

        for record in window_metadata:
            record["split"] = split_for_source.get(record["source_file"], "test")
            record["split_protocol"] = "grouped_stratified_by_label"

        return window_metadata

    def save_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        device_mapping: Dict,
        output_dir: Path,
        window_metadata: List[Dict] | None = None,
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

        manifest = window_metadata if window_metadata is not None else getattr(self, "window_metadata", [])
        if manifest:
            with open(output_dir / "split_manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)

        # Save info
        info = {
            "X_shape": X.shape,
            "y_shape": y.shape,
            "num_devices": len(device_mapping),
            "device_mapping": device_mapping,
            "window_size": self.window_size,
            "stride": self.stride,
            "max_classes": self.max_classes,
            "max_windows_per_recording": self.max_windows_per_recording,
            "output_dtype": str(self.output_dtype),
            "max_dataset_gib": self.max_dataset_gib,
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
