"""
WiSig Dataset Converter

Converts WiSig pickle-format RF signal recordings into training-ready numpy arrays.
The WiSig payload already contains fixed-length I/Q windows, so conversion mainly
normalizes tensor layout, assigns labels, and builds a split manifest that matches
the rest of the RF fingerprinting pipeline.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class WiSigConverter:
    """Convert WiSig pickle datasets to training-ready format."""

    def __init__(
        self,
        wisig_path: Path,
        window_size: int = 256,
        max_classes: int | None = None,
        max_windows_per_source: int | None = None,
        output_dtype: str = "float32",
    ):
        self.dataset_path = self._resolve_dataset_path(wisig_path)
        self.window_size = int(window_size)
        self.max_classes = max_classes
        self.max_windows_per_source = max_windows_per_source
        self.output_dtype = np.dtype(output_dtype)

        with open(self.dataset_path, "rb") as handle:
            self.payload = pickle.load(handle)

        if not isinstance(self.payload, dict) or "data" not in self.payload:
            raise ValueError(
                f"WiSig payload at {self.dataset_path} is missing the expected 'data' field"
            )

        self.tx_list = [str(item) for item in self.payload.get("tx_list", [])]
        self.rx_list = [str(item) for item in self.payload.get("rx_list", [])]
        self.capture_date_list = [str(item) for item in self.payload.get("capture_date_list", [])]
        self.equalized_list = list(self.payload.get("equalized_list", []))

        print(
            "Loaded WiSig payload "
            f"{self.dataset_path.name}: tx={len(self.payload['data'])}, "
            f"rx={len(self.rx_list)}, dates={len(self.capture_date_list)}, "
            f"equalization_modes={len(self.equalized_list)}"
        )

    def _resolve_dataset_path(self, wisig_path: Path) -> Path:
        path = Path(wisig_path)
        if not path.exists():
            raise ValueError(f"WiSig dataset not found: {wisig_path}")

        if path.is_file():
            return path

        candidates = sorted(path.glob("*.pkl")) + sorted(path.glob("*.pickle"))
        if not candidates:
            raise ValueError(f"No WiSig pickle file found under: {path}")

        preferred = path / "SingleDay.pkl"
        if preferred.exists():
            return preferred
        return candidates[0]

    def _normalize_block(self, block: object) -> np.ndarray:
        array = np.asarray(block, dtype=np.float32)

        if array.ndim != 3:
            raise ValueError(f"Unsupported WiSig block shape: {array.shape}")

        if array.shape[-1] == 2:
            array = np.transpose(array, (0, 2, 1))
        elif array.shape[1] != 2:
            raise ValueError(f"Expected a 2-channel WiSig block, got shape {array.shape}")

        if array.shape[2] != self.window_size:
            if array.shape[2] > self.window_size:
                array = array[:, :, :self.window_size]
            else:
                raise ValueError(
                    f"Expected WiSig window size {self.window_size}, got {array.shape[2]}"
                )

        return array.astype(self.output_dtype, copy=False)

    def _iter_sources(self):
        data = self.payload["data"]
        fallback_tx_list = [str(i) for i in range(len(data))]
        tx_list = self.tx_list or fallback_tx_list

        for tx_idx, tx_entry in enumerate(data):
            tx_id = tx_list[tx_idx] if tx_idx < len(tx_list) else str(tx_idx)

            if not isinstance(tx_entry, (list, tuple)):
                continue

            for rx_idx, rx_entry in enumerate(tx_entry):
                rx_id = self.rx_list[rx_idx] if rx_idx < len(self.rx_list) else str(rx_idx)
                if not isinstance(rx_entry, (list, tuple)):
                    continue

                for capture_idx, capture_entry in enumerate(rx_entry):
                    capture_date = (
                        self.capture_date_list[capture_idx]
                        if capture_idx < len(self.capture_date_list)
                        else f"capture_{capture_idx}"
                    )

                    if not isinstance(capture_entry, (list, tuple)):
                        continue

                    for equalized_idx, block in enumerate(capture_entry):
                        equalized = (
                            self.equalized_list[equalized_idx]
                            if equalized_idx < len(self.equalized_list)
                            else equalized_idx
                        )
                        source_key = (
                            f"tx={tx_id}/rx={rx_id}/date={capture_date}/equalized={equalized}"
                        )
                        yield {
                            "tx_id": tx_id,
                            "rx_id": rx_id,
                            "capture_date": capture_date,
                            "equalized": int(equalized),
                            "source_key": source_key,
                            "windows": self._normalize_block(block),
                        }

    def convert_dataset(
        self,
        split_config: Dict | None = None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Convert the WiSig dataset to numpy arrays plus metadata."""
        all_signals = []
        all_labels = []
        window_metadata = []
        device_to_label = {}
        skipped_devices = set()
        next_label = 0

        print("\nProcessing WiSig dataset...")

        for source in self._iter_sources():
            device_id = source["tx_id"]
            if device_id not in device_to_label:
                if self.max_classes is not None and next_label >= self.max_classes:
                    skipped_devices.add(device_id)
                    continue

                device_to_label[device_id] = next_label
                print(f"  Device {device_id} -> class {next_label}")
                next_label += 1

            label = device_to_label[device_id]
            windows = source["windows"]

            if self.max_windows_per_source is not None and len(windows) > self.max_windows_per_source:
                selected_idx = np.linspace(
                    0,
                    len(windows) - 1,
                    num=self.max_windows_per_source,
                    dtype=np.int64,
                )
                windows = windows[np.unique(selected_idx)]

            for window_idx, window in enumerate(windows):
                all_signals.append(window)
                all_labels.append(label)
                window_metadata.append({
                    "dataset_name": "wisig",
                    "device_id": device_id,
                    "source_file": source["source_key"],
                    "receiver_id": source["rx_id"],
                    "capture_date": source["capture_date"],
                    "equalized": source["equalized"],
                    "window_index": int(window_idx),
                    "window_start": int(window_idx * self.window_size),
                    "window_end": int((window_idx + 1) * self.window_size),
                    "window_length": self.window_size,
                    "stride": self.window_size,
                    "label": int(label),
                    "split": None,
                })

            print(f"  ✓ {source['source_key']}: {len(windows)} windows")

        if not all_signals:
            raise ValueError("No WiSig samples were extracted from the payload")

        X = np.stack(all_signals, axis=0)
        y = np.array(all_labels, dtype=np.int64)
        self.window_metadata = self.assign_window_splits(
            window_metadata,
            split_config=split_config,
            seed=seed,
        )

        print("\nDataset created:")
        print(f"  X shape: {X.shape} (samples, channels, time)")
        print(f"  y shape: {y.shape}")
        print(f"  Unique devices: {len(device_to_label)}")
        print(f"  Device mapping: {device_to_label}")
        if skipped_devices:
            print(f"  Skipped devices after class cap ({self.max_classes}): {sorted(skipped_devices)}")

        return X, y, device_to_label

    def assign_window_splits(
        self,
        window_metadata: List[Dict],
        split_config: Dict | None = None,
        seed: int = 42,
    ) -> List[Dict]:
        """Assign grouped splits for WiSig sources while keeping source blocks intact."""
        if not window_metadata:
            return []

        split_config = split_config or {
            "protocol": "grouped_by_source_file",
            "train": 0.7,
            "val": 0.1,
            "test": 0.2,
        }
        protocol = split_config.get("protocol", "grouped_by_source_file")
        if protocol == "grouped_stratified_by_label_and_distance":
            protocol = "grouped_stratified_by_label"

        if protocol not in {
            "random_by_window",
            "grouped_by_source_file",
            "grouped_stratified_by_label",
        }:
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

        if protocol == "random_by_window":
            indices = np.arange(len(window_metadata))
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

            n = len(indices)
            train_n = int(np.floor(n * train_ratio))
            val_n = int(np.floor(n * val_ratio)) if val_ratio > 0 else 0
            if train_n <= 0:
                train_n = 1
            if train_n + val_n >= n:
                val_n = max(0, n - train_n - 1)
            test_n = n - train_n - val_n

            split_for_index = {}
            for idx in indices[:train_n]:
                split_for_index[int(idx)] = "train"
            for idx in indices[train_n:train_n + val_n]:
                split_for_index[int(idx)] = "val"
            for idx in indices[train_n + val_n:train_n + val_n + test_n]:
                split_for_index[int(idx)] = "test"

            for i, record in enumerate(window_metadata):
                record["split"] = split_for_index.get(i, "test")
                record["split_protocol"] = "random_by_window"

            return window_metadata

        source_to_label = {}
        for record in window_metadata:
            source = record["source_file"]
            label = int(record["label"])
            existing = source_to_label.get(source)
            if existing is None:
                source_to_label[source] = label
            elif existing != label:
                raise ValueError(
                    f"WiSig source {source} maps to multiple labels: {existing}, {label}"
                )

        label_to_sources: Dict[int, List[str]] = {}
        for source, label in source_to_label.items():
            label_to_sources.setdefault(label, []).append(source)

        rng = np.random.default_rng(seed)
        split_for_source = {}

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

                if test_n < 1:
                    test_n = 1
                    if val_n > 0:
                        val_n -= 1
                    else:
                        train_n = max(1, train_n - 1)

                if val_ratio > 0 and n >= 3 and val_n == 0:
                    if train_n > 1:
                        train_n -= 1
                        val_n = 1
                    elif test_n > 1:
                        test_n -= 1
                        val_n = 1

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
            record["split_protocol"] = protocol

        return window_metadata

    def save_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        device_mapping: Dict,
        output_dir: Path,
        window_metadata: List[Dict] | None = None,
    ) -> None:
        """Save converted WiSig dataset to numpy files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "X.npy", X)
        np.save(output_dir / "y.npy", y)

        with open(output_dir / "device_mapping.json", "w", encoding="utf-8") as handle:
            json.dump(device_mapping, handle, indent=2)

        manifest = window_metadata if window_metadata is not None else getattr(self, "window_metadata", [])
        if manifest:
            with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)

        info = {
            "dataset_name": "wisig",
            "dataset_path": str(self.dataset_path),
            "X_shape": X.shape,
            "y_shape": y.shape,
            "num_devices": len(device_mapping),
            "device_mapping": device_mapping,
            "window_size": self.window_size,
            "max_classes": self.max_classes,
            "max_windows_per_source": self.max_windows_per_source,
            "output_dtype": str(self.output_dtype),
            "num_receivers": len(self.rx_list),
            "capture_dates": self.capture_date_list,
            "equalized_list": self.equalized_list,
        }

        with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2)

        print(f"\n✓ Dataset saved to {output_dir}")
        print(f"  - X.npy: {X.shape}")
        print(f"  - y.npy: {y.shape}")
        print("  - device_mapping.json")
        print("  - dataset_info.json")