"""
Dataset loader for ORACLE and WiSig RF fingerprinting data.

This module provides a common PyTorch data-loading interface for:
- converted ORACLE datasets stored as X.npy / y.npy / metadata JSON
- WiSig pickle files such as src/datasets/WiSig/SingleDay.pkl
"""
import yaml
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _coerce_to_channels_time(X: np.ndarray, window_size: int = 256) -> np.ndarray:
    """Normalize an array to shape (num_
    samples, 2, window_size)."""
    arr = np.asarray(X, dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {arr.shape}")

    if arr.shape[1] != 2 and arr.shape[2] == 2:
        arr = np.transpose(arr, (0, 2, 1))

    if arr.shape[1] != 2:
        raise ValueError(f"Expected 2 channels, got shape {arr.shape}")

    if arr.shape[2] != window_size:
        if arr.shape[2] > window_size:
            arr = arr[:, :, :window_size]
        else:
            raise ValueError(
                f"Expected window size {window_size}, got {arr.shape[2]}"
            )

    return arr


class OracleRFDataset(Dataset):
    """Generic PyTorch Dataset for RF fingerprinting samples."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        device_mapping: Dict[str, int],
        window_size: int = 256,
        normalize: bool = True
    ):
        self.X = _coerce_to_channels_time(X, window_size=window_size)
        self.y = np.asarray(y)
        self.device_mapping = device_mapping
        self.window_size = window_size
        self.normalize = normalize

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X and y must have matching lengths: {self.X.shape[0]} vs {self.y.shape[0]}"
            )

        self.num_classes = len(device_mapping)
        self.id_to_device = {v: k for k, v in device_mapping.items()}

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Dict:
        signal = self.X[index].astype(np.float32)
        label = int(self.y[index])

        if self.normalize:
            signal = (signal - signal.mean()) / (signal.std() + 1e-8)

        return {
            "x": torch.from_numpy(signal),
            "y": torch.tensor(label, dtype=torch.long),
            "metadata": {
                "device": self.id_to_device.get(label, "unknown"),
                "class": label,
            },
        }


def _load_wisig_dataset(
    dataset_path: Path,
    split_ratio: float = 0.8,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0,
    window_size: int = 256,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Load a WiSig pickle file into train/test loaders."""
    dataset_path = Path(dataset_path)

    if dataset_path.is_dir():
        candidates = list(dataset_path.glob("*.pkl")) + list(dataset_path.glob("*.pickle"))
        if not candidates:
            raise FileNotFoundError(
                f"No WiSig pickle file found in {dataset_path}"
            )
        dataset_path = candidates[0]

    with open(dataset_path, "rb") as handle:
        payload = pickle.load(handle)

    raw_data = payload.get("data", [])
    tx_ids = payload.get("tx_list", [str(i) for i in range(len(raw_data))])

    if not raw_data:
        raise ValueError("WiSig payload did not contain any data entries")

    X_list = []
    y_list = []
    device_mapping = {}

    for tx_idx, tx_entry in enumerate(raw_data):
        tx_id = str(tx_ids[tx_idx]) if tx_idx < len(tx_ids) else str(tx_idx)
        if tx_id not in device_mapping:
            device_mapping[tx_id] = len(device_mapping)

        label = device_mapping[tx_id]

        for rx_entry in tx_entry:
            if not isinstance(rx_entry, (list, tuple)) or len(rx_entry) == 0:
                continue

            sample = rx_entry[0]
            array = np.asarray(sample, dtype=np.float32)

            if isinstance(sample, (list, tuple)) and len(sample) == 2:
                first = np.asarray(sample[0], dtype=np.float32)
                second = np.asarray(sample[1], dtype=np.float32)
                if first.ndim == 3 and second.ndim == 3:
                    array = np.stack([first, second], axis=1)
                else:
                    array = np.stack([first, second], axis=0)
            elif array.ndim == 2:
                array = array[np.newaxis, :, :]

            if array.ndim == 3 and array.shape[-1] == 2:
                array = np.transpose(array, (0, 2, 1))
            elif array.ndim == 3 and array.shape[1] == 2:
                array = np.transpose(array, (0, 2, 1))
            elif array.ndim == 4 and array.shape[1] == 2 and array.shape[-1] == 2:
                array = np.transpose(array, (0, 1, 3, 2))
                array = array.reshape(array.shape[0] * array.shape[1], array.shape[2], array.shape[3])
            else:
                raise ValueError(f"Unsupported WiSig sample shape: {array.shape}")

            if array.ndim == 3 and array.shape[1] != 2:
                raise ValueError(f"Expected 2 channels after conversion, got {array.shape}")

            if array.ndim == 3 and array.shape[2] != window_size:
                if array.shape[2] > window_size:
                    array = array[:, :, :window_size]
                else:
                    raise ValueError(
                        f"Expected window size {window_size}, got {array.shape[2]}"
                    )

            if array.ndim == 3:
                for sample_window in array:
                    X_list.append(sample_window)
                    y_list.append(label)

    if not X_list:
        raise ValueError("Failed to extract any WiSig samples")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    print(f"Loaded WiSig dataset from {dataset_path}:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Devices: {device_mapping}")

    return _build_loaders(
        X,
        y,
        device_mapping,
        split_ratio=split_ratio,
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
        window_size=window_size,
        normalize=normalize,
        source_name="wisig",
    )


def _build_loaders(
    X: np.ndarray,
    y: np.ndarray,
    device_mapping: Dict[str, int],
    split_ratio: float = 0.8,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0,
    window_size: int = 256,
    normalize: bool = True,
    source_name: str = "dataset",
    split_manifest: list | None = None,
    split_config: Dict | None = None,
) -> Tuple[DataLoader, DataLoader, Dict]:
    num_samples = len(X)
    if split_manifest and len(split_manifest) == num_samples:
        split_labels = np.array([entry.get("split", "test") for entry in split_manifest], dtype=object)
        train_indices = np.where(split_labels == "train")[0]
        eval_indices = np.where(np.isin(split_labels, ["val", "test"]))[0]
        if len(train_indices) == 0 or len(eval_indices) == 0:
            raise ValueError("Split manifest did not produce both train and evaluation partitions")
    else:
        split_idx = int(num_samples * split_ratio)
        indices = np.random.permutation(num_samples)
        train_indices = indices[:split_idx]
        eval_indices = indices[split_idx:]

    X_train, X_eval = X[train_indices], X[eval_indices]
    y_train, y_eval = y[train_indices], y[eval_indices]

    train_dataset = OracleRFDataset(
        X_train,
        y_train,
        device_mapping,
        window_size=window_size,
        normalize=normalize,
    )
    test_dataset = OracleRFDataset(
        X_eval,
        y_eval,
        device_mapping,
        window_size=window_size,
        normalize=normalize,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    metadata = {
        "num_samples": num_samples,
        "num_train": len(X_train),
        "num_eval": len(X_eval),
        "num_classes": len(device_mapping),
        "device_mapping": device_mapping,
        "window_size": window_size,
        "input_shape": X_train[0].shape,
        "source": source_name,
        "split_manifest": split_manifest is not None,
        "split_config": split_config,
    }

    return train_loader, test_loader, metadata


def load_rfdataset(
    dataset_source: Path,
    split_ratio: float = 0.8,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0,
    window_size: int = 256,
    normalize: bool = True,
    split_config: Dict | None = None,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Load either a converted ORACLE dataset or a WiSig pickle file."""
    dataset_source = Path(dataset_source)

    if dataset_source.suffix.lower() in {".pkl", ".pickle"}:
        return _load_wisig_dataset(
            dataset_source,
            split_ratio=split_ratio,
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            num_workers=num_workers,
            window_size=window_size,
            normalize=normalize,
        )

    if dataset_source.is_dir():
        npy_path = dataset_source / "X.npy"
        y_path = dataset_source / "y.npy"
        mapping_path = dataset_source / "device_mapping.json"
        info_path = dataset_source / "dataset_info.json"

        if npy_path.exists() and y_path.exists() and mapping_path.exists() and info_path.exists():
            X = np.load(npy_path)
            y = np.load(y_path)
            with open(mapping_path, "r") as handle:
                device_mapping = json.load(handle)
            with open(info_path, "r") as handle:
                info = json.load(handle)

            print(f"Loaded ORACLE dataset from {dataset_source}:")
            print(f"  X shape: {X.shape}")
            print(f"  y shape: {y.shape}")
            print(f"  Devices: {device_mapping}")

            split_manifest = None
            if (dataset_source / "split_manifest.json").exists():
                with open(dataset_source / "split_manifest.json", "r") as handle:
                    split_manifest = json.load(handle)

            return _build_loaders(
                X,
                y,
                device_mapping,
                split_ratio=split_ratio,
                batch_size=batch_size,
                shuffle_train=shuffle_train,
                num_workers=num_workers,
                window_size=info.get("window_size", window_size),
                normalize=normalize,
                source_name="oracle",
                split_manifest=split_manifest,
                split_config=split_config,
            )

        if (dataset_source / "SingleDay.pkl").exists():
            return _load_wisig_dataset(
                dataset_source / "SingleDay.pkl",
                split_ratio=split_ratio,
                batch_size=batch_size,
                shuffle_train=shuffle_train,
                num_workers=num_workers,
                window_size=window_size,
                normalize=normalize,
            )

    raise FileNotFoundError(
        f"Could not find a supported dataset at {dataset_source}. "
        "Expected either a converted ORACLE directory or a WiSig .pkl file."
    )


def load_oracle_dataset(
    dataset_dir: Path,
    split_ratio: float = 0.8,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Backward-compatible wrapper for the original ORACLE loader API."""
    return load_rfdataset(
        dataset_dir,
        split_ratio=split_ratio,
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    dataset_path = Path(__file__).parent / "WiSig"
    train_loader, test_loader, metadata = load_rfdataset(
        dataset_path,
        split_ratio=0.8,
        batch_size=8,
    )

    print("\n✓ Dataloaders ready!")
    print(f"  Metadata: {json.dumps(metadata, indent=2)}")

    batch = next(iter(train_loader))
    print("\nExample batch:")
    print(f"  x shape: {batch['x'].shape}")
    print(f"  y shape: {batch['y'].shape}")
    print(f"  metadata: {batch['metadata']}")
