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
from src.datasets.wisig_converter import WiSigConverter
from src.evaluation.metrics import calculate_metrics
from src.models.cnn1d import RF_CNN
from src.models.cnn1d_multi import RF_CNN_MULTI
from src.models.hypernea_proto import HyperNEAPrototype
from src.training.neuroevolution import evolve_model


def build_model(model_cfg: Dict[str, Any], num_classes: int, input_channels: int, window_size: int = 256):
    model_name = str(model_cfg.get("name", "cnn1d")).lower()
    in_channels = int(model_cfg.get("in_channels", input_channels))

    if model_name == "cnn1d":
        return RF_CNN(classes=num_classes), model_name
    if model_name == "cnn1d_multi":
        return RF_CNN_MULTI(classes=num_classes, in_channels=in_channels), model_name
    if model_name == "hypernea_proto":
        return (
            HyperNEAPrototype(
                classes=num_classes,
                input_channels=in_channels,
                window_size=int(model_cfg.get("window_size", window_size)),
                hidden_size=int(model_cfg.get("hidden_size", 32)),
                cppn_hidden=int(model_cfg.get("cppn_hidden", 24)),
            ),
            model_name,
        )

    raise ValueError(
        f"Unsupported model.name '{model_name}'. Supported: 'cnn1d', 'cnn1d_multi', 'hypernea_proto'."
    )


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


def save_training_curves_png(losses: list, output_path: Path, title: str = "Training Loss Curve", y_label: str = "Loss") -> None:
    """Save training curve as PNG."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", linestyle="-", linewidth=2, markersize=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def _is_converted_dataset_dir(dataset_path: Path) -> bool:
    return dataset_path.is_dir() and all(
        (dataset_path / name).exists()
        for name in ("X.npy", "y.npy", "device_mapping.json", "dataset_info.json")
    )


def _normalize_split_config(split_config: Dict[str, Any] | float | int | None) -> Dict[str, Any]:
    if isinstance(split_config, (int, float)):
        return {
            "protocol": "grouped_by_source_file",
            "train": float(split_config),
            "val": 0.0,
            "test": max(0.0, 1.0 - float(split_config)),
        }

    if split_config is None:
        return {
            "protocol": "grouped_by_source_file",
            "train": 0.8,
            "val": 0.0,
            "test": 0.2,
        }

    return split_config


def build_dataset_loaders(config: Dict[str, Any], project_root: Path) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    dataset_cfg = config["dataset"]
    model_cfg = config.get("model", {})
    dataset_name = str(dataset_cfg.get("name", "oracle")).lower()
    dataset_path = resolve_path(dataset_cfg["path"], project_root)

    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_name} dataset path does not exist: {dataset_path}")

    # If dataset.max_classes is not set, default to model.classes to avoid
    # converting more devices than the classifier head can represent.
    max_classes_cfg = dataset_cfg.get("max_classes")
    if max_classes_cfg is None:
        max_classes_cfg = model_cfg.get("classes")

    split_config = _normalize_split_config(dataset_cfg.get("split", 0.8))
    output_dir = resolve_path(
        dataset_cfg.get("output_dir", f"datasets/{dataset_name}"),
        project_root,
    )

    load_source = dataset_path
    if not _is_converted_dataset_dir(dataset_path):
        if dataset_name == "oracle":
            converter = OracleConverter(
                oracle_dir=dataset_path,
                window_size=int(dataset_cfg.get("window_length", 256)),
                stride=int(dataset_cfg.get("stride", dataset_cfg.get("window_length", 256) // 2)),
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
        elif dataset_name == "wisig":
            max_windows_per_source_cfg = dataset_cfg.get("max_windows_per_source")
            if max_windows_per_source_cfg is None:
                max_windows_per_source_cfg = dataset_cfg.get("max_windows_per_recording")

            converter = WiSigConverter(
                wisig_path=dataset_path,
                window_size=int(dataset_cfg.get("window_length", 256)),
                max_classes=(
                    int(max_classes_cfg)
                    if max_classes_cfg is not None
                    else None
                ),
                max_windows_per_source=(
                    int(max_windows_per_source_cfg)
                    if max_windows_per_source_cfg is not None
                    else None
                ),
                output_dtype=str(dataset_cfg.get("output_dtype", "float32")),
            )
        else:
            raise ValueError(
                f"Unsupported dataset.name '{dataset_name}'. Supported values are 'oracle' and 'wisig'."
            )

        X, y, device_mapping = converter.convert_dataset(
            split_config=split_config,
            seed=int(dataset_cfg.get("seed", 42)),
        )
        converter.save_dataset(
            X,
            y,
            device_mapping,
            output_dir,
            window_metadata=converter.window_metadata,
        )
        load_source = output_dir

    train_loader, test_loader, metadata = load_rfdataset(
        load_source,
        split_ratio=float(split_config.get("train", 0.8)),
        batch_size=int(dataset_cfg.get("batch_size", 32)),
        shuffle_train=True,
        num_workers=0,
        window_size=int(dataset_cfg.get("window_length", 256)),
        normalize=bool(dataset_cfg.get("normalize", True)),
        channels=int(dataset_cfg.get("channels", 2)),
        channel_mode=str(dataset_cfg.get("channel_mode", "iq")),
        split_config=split_config,
    )
    return train_loader, test_loader, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN on an RF fingerprint dataset")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path)
    seed = int(config["dataset"].get("seed", 42))
    set_seed(seed)

    train_loader, test_loader, metadata = build_dataset_loaders(config, project_root)

    configured_classes = config["model"].get("classes")
    num_classes = metadata.get("num_classes", 2)
    if configured_classes is not None and int(configured_classes) != num_classes:
        print(
            f"Warning: model.classes={configured_classes} does not match dataset classes={num_classes}. "
            f"Using dataset class count instead."
        )
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "cnn1d"))
    if model_name.lower() == "cnn1d_multi" and int(config["dataset"].get("channels", 2)) != 4:
        print("Warning: model=cnn1d_multi expects dataset.channels=4. Overriding channels to 4 for this run.")
        config["dataset"]["channels"] = 4
        train_loader, test_loader, metadata = build_dataset_loaders(config, project_root)
        num_classes = metadata.get("num_classes", num_classes)

    input_shape = metadata.get("input_shape", [2, int(config["dataset"].get("window_length", 256))])
    input_channels = int(input_shape[0])
    model, resolved_model_name = build_model(
        model_cfg,
        num_classes,
        input_channels=input_channels,
        window_size=int(config["dataset"].get("window_length", 256)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_dir = resolve_path(config["training"].get("output_dir", "results/oracle_cnn"), project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_output_dir = resolve_path(config["dataset"].get("output_dir", "datasets/oracle"), project_root)
    split_manifest_path = dataset_output_dir / "split_manifest.json"
    if split_manifest_path.exists():
        shutil.copy2(split_manifest_path, output_dir / "split_manifest.json")

    epoch_losses = []
    training_method = str(config["training"].get("method", "adam")).lower()
    if resolved_model_name == "hypernea_proto":
        training_method = "neuroevolution"

    if training_method == "neuroevolution":
        epoch_losses, evolution_summary = evolve_model(
            model,
            train_loader,
            device,
            config["training"],
            seed=seed,
        )
        training_summary = {
            "model_name": resolved_model_name,
            "epochs": int(len(epoch_losses)),
            "batch_size": int(config["dataset"].get("batch_size", 32)),
            "learning_rate": None,
            "optimizer": evolution_summary["algorithm"],
            "loss_function": "fitness=train_accuracy",
            "final_loss": None,
            "best_fitness": float(evolution_summary["best_fitness"]),
            "evolution": evolution_summary,
        }
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"].get("learning_rate", 0.001)))
        loss_fn = torch.nn.CrossEntropyLoss()
        epochs = int(config["training"].get("epochs", 10))
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

        training_summary = {
            "model_name": resolved_model_name,
            "epochs": epochs,
            "batch_size": int(config["dataset"].get("batch_size", 32)),
            "learning_rate": float(config["training"].get("learning_rate", 0.001)),
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "final_loss": float(epoch_losses[-1]) if epoch_losses else None,
        }

    def evaluate_accuracy(loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                predicted = pred.argmax(dim=1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        return float(correct / total) if total > 0 else 0.0

    training_accuracy = evaluate_accuracy(train_loader)

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
    metrics["training"] = training_summary
    metrics["metadata"] = {
        "random_seed": seed,
        "git_commit": get_git_commit_hash(project_root),
        "device": str(device),
        "training_accuracy": training_accuracy,
        "testing_accuracy": float(metrics["accuracy"]),
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
        if training_method == "neuroevolution":
            save_training_curves_png(
                epoch_losses,
                output_dir / "training_curves.png",
                title="Evolution Fitness Curve",
                y_label="Best Fitness",
            )
        else:
            save_training_curves_png(epoch_losses, output_dir / "training_curves.png")

    model_artifact_name = f"{resolved_model_name}.pt"
    torch.save(model.state_dict(), output_dir / model_artifact_name)
    print(f"\nSaved artifacts:")
    print(f"  - metrics.json")
    print(f"  - config.yaml")
    print(f"  - {model_artifact_name}")
    print(f"  - confusion_matrix.png")
    print(f"  - confusion_matrix.csv")
    print(f"  - training_curves.png")
    print(f"  - split_manifest.json")
    print(f"\nRun metadata:")
    print(f"  Random seed: {seed}")
    print(f"  Git commit: {metrics['metadata']['git_commit'][:7]}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Training accuracy: {training_accuracy:.4f}")
    print(f"  Testing accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()