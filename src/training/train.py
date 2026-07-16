import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Make project imports work when running as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.oracle_converter import OracleConverter
from src.datasets.oracle_loader import load_rfdataset
from src.evaluation.metrics import calculate_metrics
from src.models.cnn1d import RF_CNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> Dict[str, Any]:
    import yaml

    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base_dir / path


def get_git_commit_hash(repo_path: Path) -> str:
    """Get the current git commit hash, or 'unknown' if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def save_confusion_matrix_png(cm: np.ndarray, output_path: Path, class_labels: list = None) -> None:
    """Save confusion matrix as PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    
    if class_labels is None:
        class_labels = [str(i) for i in range(len(cm))]
    
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    
    for i in range(len(cm)):
        for j in range(len(cm)):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="w" if cm[i, j] > cm.max() / 2 else "k")
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_csv(cm: np.ndarray, output_path: Path, class_labels: list = None) -> None:
    """Save confusion matrix as CSV."""
    if class_labels is None:
        class_labels = [str(i) for i in range(len(cm))]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["True \\\\ Predicted"] + class_labels)
        for i, label in enumerate(class_labels):
            writer.writerow([label] + cm[i].tolist())


def save_training_curves_png(losses: list, output_path: Path) -> None:
    """Save training loss curve as PNG."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", linestyle="-", linewidth=2, markersize=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def build_oracle_loaders(config: Dict[str, Any], project_root: Path) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    dataset_cfg = config["dataset"]
    model_cfg = config.get("model", {})
    dataset_path = resolve_path(dataset_cfg["path"], project_root)

    if not dataset_path.exists():
        raise FileNotFoundError(f"ORACLE dataset path does not exist: {dataset_path}")

    # If dataset.max_classes is not set, default to model.classes to avoid
    # converting more devices than the classifier head can represent.
    max_classes_cfg = dataset_cfg.get("max_classes")
    if max_classes_cfg is None:
        max_classes_cfg = model_cfg.get("classes")

    converter = OracleConverter(
        oracle_dir=dataset_path,
        window_size=int(dataset_cfg.get("window_length", 256)),
        stride=int(dataset_cfg.get("window_length", 256) // 2),
        max_classes=(
            int(max_classes_cfg)
            if max_classes_cfg is not None
            else None
        ),
        max_windows_per_recording=(
            int(dataset_cfg["max_windows_per_recording"])
            if dataset_cfg.get("max_windows_per_recording") is not None
            else None
        ),
        output_dtype=str(dataset_cfg.get("output_dtype", "float32")),
        max_dataset_gib=float(dataset_cfg.get("max_dataset_gib", 8.0)),
    )

    split_config = dataset_cfg.get("split", 0.8)
    if isinstance(split_config, (int, float)):
        split_config = {
            "protocol": "grouped_by_source_file",
            "train": float(split_config),
            "val": 0.0,
            "test": max(0.0, 1.0 - float(split_config)),
        }

    X, y, device_mapping = converter.convert_dataset(
        split_config=split_config,
        seed=int(dataset_cfg.get("seed", 42)),
    )
    output_dir = resolve_path(dataset_cfg.get("output_dir", "datasets/oracle"), project_root)
    converter.save_dataset(X, y, device_mapping, output_dir, window_metadata=converter.window_metadata)

    train_loader, test_loader, metadata = load_rfdataset(
        output_dir,
        split_ratio=float(split_config.get("train", 0.8)),
        batch_size=int(dataset_cfg.get("batch_size", 32)),
        shuffle_train=True,
        num_workers=0,
        window_size=int(dataset_cfg.get("window_length", 256)),
        normalize=bool(dataset_cfg.get("normalize", True)),
        split_config=split_config,
    )
    return train_loader, test_loader, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN on the ORACLE dataset")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path)
    seed = int(config["dataset"].get("seed", 42))
    set_seed(seed)

    train_loader, test_loader, metadata = build_oracle_loaders(config, project_root)

    configured_classes = config["model"].get("classes")
    num_classes = metadata.get("num_classes", 2)
    if configured_classes is not None and int(configured_classes) != num_classes:
        print(
            f"Warning: model.classes={configured_classes} does not match dataset classes={num_classes}. "
            f"Using dataset class count instead."
        )
    model = RF_CNN(classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"].get("learning_rate", 0.001)))
    loss_fn = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_dir = resolve_path(config["training"].get("output_dir", "results/oracle_cnn"), project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_output_dir = resolve_path(config["dataset"].get("output_dir", "datasets/oracle"), project_root)
    split_manifest_path = dataset_output_dir / "split_manifest.json"
    if split_manifest_path.exists():
        shutil.copy2(split_manifest_path, output_dir / "split_manifest.json")

    epochs = int(config["training"].get("epochs", 10))
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f}")

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            labels = pred.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(labels.cpu().tolist())

    metrics = calculate_metrics(y_true, y_pred)
    metrics["dataset"] = metadata
    metrics["training"] = {
        "epochs": epochs,
        "batch_size": int(config["dataset"].get("batch_size", 32)),
        "learning_rate": float(config["training"].get("learning_rate", 0.001)),
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "final_loss": float(epoch_losses[-1]) if epoch_losses else None,
    }
    metrics["metadata"] = {
        "random_seed": seed,
        "git_commit": get_git_commit_hash(project_root),
        "device": str(device),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with open(output_dir / "config.yaml", "w", encoding="utf-8") as handle:
        import yaml

        yaml.safe_dump(config, handle, sort_keys=False)

    # Save confusion matrix as PNG and CSV
    cm = np.array(metrics["confusion_matrix"])
    device_mapping = metadata.get("device_mapping", {})
    class_labels = [device_id for device_id, _ in sorted(device_mapping.items(), key=lambda x: x[1])]
    save_confusion_matrix_png(cm, output_dir / "confusion_matrix.png", class_labels=class_labels)
    save_confusion_matrix_csv(cm, output_dir / "confusion_matrix.csv", class_labels=class_labels)

    # Save training curves
    if epoch_losses:
        save_training_curves_png(epoch_losses, output_dir / "training_curves.png")

    torch.save(model.state_dict(), output_dir / "oracle_cnn.pt")
    print(f"\nSaved artifacts:")
    print(f"  - metrics.json")
    print(f"  - config.yaml")
    print(f"  - oracle_cnn.pt")
    print(f"  - confusion_matrix.png")
    print(f"  - confusion_matrix.csv")
    print(f"  - training_curves.png")
    print(f"  - split_manifest.json")
    print(f"\nRun metadata:")
    print(f"  Random seed: {seed}")
    print(f"  Git commit: {metrics['metadata']['git_commit'][:7]}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
