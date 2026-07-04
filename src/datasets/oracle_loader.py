"""
ORACLE Dataset Loader

Provides convenient methods to load converted ORACLE dataset
and integrate with PyTorch training pipeline.
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader


class OracleRFDataset(Dataset):
    """
    PyTorch Dataset for ORACLE RF fingerprinting data.

    Handles normalized I/Q samples and device labels.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        device_mapping: Dict[str, int],
        window_size: int = 256,
        normalize: bool = True
    ):
        """
        Initialize dataset.

        Args:
            X: Signal array of shape (num_samples, channels, time)
            y: Label array of shape (num_samples,)
            device_mapping: Mapping from device_id to class index
            window_size: Expected window size
            normalize: Whether to normalize signals
        """
        self.X = X
        self.y = y
        self.device_mapping = device_mapping
        self.window_size = window_size
        self.normalize = normalize

        assert X.shape[0] == y.shape[0], "X and y must have same length"
        assert X.shape[1] == 2, "Expected 2 channels (I/Q)"
        assert X.shape[2] == window_size, f"Expected window size {window_size}"

        self.num_classes = len(device_mapping)
        self.id_to_device = {v: k for k, v in device_mapping.items()}

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Dict:
        """Get single sample."""
        signal = self.X[index].astype(np.float32)
        label = self.y[index]

        # Normalize if requested
        if self.normalize:
            signal = (signal - signal.mean()) / (signal.std() + 1e-8)

        return {
            "x": torch.from_numpy(signal),
            "y": torch.tensor(label, dtype=torch.long),
            "metadata": {
                "device": self.id_to_device.get(int(label), "unknown"),
                "class": int(label)
            }
        }


def load_oracle_dataset(
    dataset_dir: Path,
    split_ratio: float = 0.8,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Load converted ORACLE dataset and create train/test loaders.

    Args:
        dataset_dir: Directory containing converted dataset files
        split_ratio: Fraction for training data (rest for testing)
        batch_size: Batch size for DataLoader
        shuffle_train: Whether to shuffle training data
        num_workers: Number of workers for DataLoader

    Returns:
        Tuple of (train_loader, test_loader, metadata_dict)
    """
    dataset_dir = Path(dataset_dir)

    # Load arrays
    X = np.load(dataset_dir / "X.npy")
    y = np.load(dataset_dir / "y.npy")

    # Load metadata
    with open(dataset_dir / "device_mapping.json", 'r') as f:
        device_mapping = json.load(f)

    with open(dataset_dir / "dataset_info.json", 'r') as f:
        info = json.load(f)

    print(f"Loaded ORACLE dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Devices: {device_mapping}")

    # Split into train/test
    num_samples = len(X)
    split_idx = int(num_samples * split_ratio)

    # Shuffle indices before split
    indices = np.random.permutation(num_samples)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Create datasets
    train_dataset = OracleRFDataset(
        X_train, y_train, device_mapping,
        window_size=info["window_size"],
        normalize=True
    )

    test_dataset = OracleRFDataset(
        X_test, y_test, device_mapping,
        window_size=info["window_size"],
        normalize=True
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    metadata = {
        "num_samples": num_samples,
        "num_train": len(X_train),
        "num_test": len(X_test),
        "num_classes": len(device_mapping),
        "device_mapping": device_mapping,
        "window_size": info["window_size"],
        "input_shape": X_train[0].shape  # (channels, time)
    }

    return train_loader, test_loader, metadata


if __name__ == "__main__":
    """Example usage."""
    dataset_dir = Path(__file__).parent / "../../datasets/oracle"

    if not dataset_dir.exists():
        print(f"Dataset not found at {dataset_dir}")
        print("Please run oracle_converter.py first to convert the ORACLE dataset")
    else:
        train_loader, test_loader, metadata = load_oracle_dataset(
            dataset_dir,
            split_ratio=0.8,
            batch_size=32
        )

        print("\n✓ Dataloaders ready!")
        print(f"  Metadata: {json.dumps(metadata, indent=2)}")

        # Example: iterate one batch
        print("\nExample batch:")
        batch = next(iter(train_loader))
        print(f"  x shape: {batch['x'].shape}")
        print(f"  y shape: {batch['y'].shape}")
        print(f"  metadata: {batch['metadata']}")
