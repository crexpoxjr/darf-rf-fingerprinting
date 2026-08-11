import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import calculate_metrics
from src.training.train import build_dataset_loaders, build_model, load_config, resolve_path, set_seed


class RawSignalDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        channels: int = 2,
        normalize: bool = True,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.channels = int(channels)
        self.normalize = bool(normalize)

        if self.X.ndim != 3 or self.X.shape[1] != 2:
            raise ValueError(f"Expected raw RF array of shape (N, 2, T), got {self.X.shape}")
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of samples")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        raw = torch.from_numpy(self.X[index])
        x = preprocess_raw_tensor(raw.unsqueeze(0), channels=self.channels, normalize=self.normalize).squeeze(0)
        y = torch.tensor(int(self.y[index]), dtype=torch.long)
        return {"x": x, "y": y}


def preprocess_raw_tensor(raw_x: torch.Tensor, channels: int, normalize: bool) -> torch.Tensor:
    if raw_x.ndim == 2:
        raw_x = raw_x.unsqueeze(0)

    if raw_x.ndim != 3 or raw_x.shape[1] != 2:
        raise ValueError(f"Expected raw input shape (B, 2, T), got {tuple(raw_x.shape)}")

    signal = raw_x
    if int(channels) == 4:
        i = signal[:, 0, :]
        q = signal[:, 1, :]
        magnitude = torch.sqrt(torch.clamp(i.square() + q.square(), min=1e-12))
        phase = torch.atan2(q, i)
        signal = torch.stack([i, q, magnitude, phase], dim=1)
    elif int(channels) != 2:
        raise ValueError(f"Supported channel counts are 2 or 4, got {channels}")

    if normalize:
        mean = signal.mean(dim=(1, 2), keepdim=True)
        std = signal.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        signal = (signal - mean) / std

    return signal


def raw_to_complex(X: np.ndarray) -> np.ndarray:
    return X[:, 0, :].astype(np.float32) + 1j * X[:, 1, :].astype(np.float32)


def complex_to_raw(z: np.ndarray) -> np.ndarray:
    return np.stack([np.real(z), np.imag(z)], axis=1).astype(np.float32)


def apply_awgn(X: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    if math.isinf(snr_db):
        return X.copy()

    rng = np.random.default_rng(seed)
    signal_power = np.mean(np.square(X), axis=(1, 2), keepdims=True)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=X.shape).astype(np.float32)
    return (X + noise).astype(np.float32)


def apply_cfo(X: np.ndarray, normalized_offset: float) -> np.ndarray:
    z = raw_to_complex(X)
    time = np.arange(z.shape[1], dtype=np.float32)
    phasor = np.exp(1j * 2.0 * np.pi * normalized_offset * time)
    return complex_to_raw(z * phasor[np.newaxis, :])


def apply_phase_bias(X: np.ndarray, phase_radians: float) -> np.ndarray:
    z = raw_to_complex(X)
    return complex_to_raw(z * np.exp(1j * phase_radians))


def apply_random_phase_rotation(X: np.ndarray, max_abs_phase_radians: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = raw_to_complex(X)
    phase = rng.uniform(-max_abs_phase_radians, max_abs_phase_radians, size=(z.shape[0], 1))
    return complex_to_raw(z * np.exp(1j * phase))


def apply_integer_timing_shift(X: np.ndarray, shift: int) -> np.ndarray:
    shifted = np.zeros_like(X, dtype=np.float32)
    if shift == 0:
        return X.copy()

    if shift > 0:
        shifted[:, :, shift:] = X[:, :, :-shift]
    else:
        shifted[:, :, :shift] = X[:, :, -shift:]
    return shifted


def apply_fractional_delay(X: np.ndarray, delay: float) -> np.ndarray:
    if abs(delay) < 1e-12:
        return X.copy()

    time = np.arange(X.shape[2], dtype=np.float32)
    shifted_time = time - delay
    delayed = np.empty_like(X, dtype=np.float32)
    for sample_idx in range(X.shape[0]):
        for channel_idx in range(X.shape[1]):
            delayed[sample_idx, channel_idx] = np.interp(
                time,
                shifted_time,
                X[sample_idx, channel_idx],
                left=0.0,
                right=0.0,
            )
    return delayed


def apply_rayleigh_flat_fading(X: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = raw_to_complex(X)
    taps = (
        rng.normal(size=(z.shape[0], 1)) + 1j * rng.normal(size=(z.shape[0], 1))
    ) / np.sqrt(2.0)
    faded = z * taps
    power = np.sqrt(np.mean(np.abs(faded) ** 2, axis=1, keepdims=True)).clip(min=1e-8)
    faded = faded / power
    return complex_to_raw(faded)


def apply_rician_flat_fading(X: np.ndarray, k_factor: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = raw_to_complex(X)
    los = np.sqrt(k_factor / (k_factor + 1.0))
    scatter = (
        rng.normal(size=(z.shape[0], 1)) + 1j * rng.normal(size=(z.shape[0], 1))
    ) / np.sqrt(2.0 * (k_factor + 1.0))
    faded = z * (los + scatter)
    power = np.sqrt(np.mean(np.abs(faded) ** 2, axis=1, keepdims=True)).clip(min=1e-8)
    faded = faded / power
    return complex_to_raw(faded)


def apply_multipath_channel(
    X: np.ndarray,
    tap_delays: Sequence[int],
    tap_gains: Sequence[float],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = raw_to_complex(X)
    output = np.zeros_like(z, dtype=np.complex64)

    for sample_idx in range(z.shape[0]):
        response = np.zeros(z.shape[1], dtype=np.complex64)
        for delay, gain in zip(tap_delays, tap_gains):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            tap = gain * np.exp(1j * phase)
            delayed = np.zeros(z.shape[1], dtype=np.complex64)
            delayed[delay:] = z[sample_idx, : z.shape[1] - delay] * tap
            response += delayed
        norm = np.sqrt(np.mean(np.abs(response) ** 2)).clip(min=1e-8)
        output[sample_idx] = response / norm

    return complex_to_raw(output)


def evaluate_model(
    model: torch.nn.Module,
    X_raw: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    channels: int,
    normalize: bool,
    device: torch.device,
) -> Dict[str, Any]:
    dataset = RawSignalDataset(X_raw, y, channels=channels, normalize=normalize)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            labels = batch["y"].to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    metrics = calculate_metrics(y_true, y_pred)
    metrics["num_samples"] = int(len(y_true))
    return metrics


def evaluate_fgsm(
    model: torch.nn.Module,
    X_raw: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    channels: int,
    normalize: bool,
    epsilon: float,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    for start in range(0, len(X_raw), batch_size):
        batch_x = torch.tensor(X_raw[start:start + batch_size], dtype=torch.float32, device=device)
        batch_y = torch.tensor(y[start:start + batch_size], dtype=torch.long, device=device)
        batch_x.requires_grad_(True)

        logits = model(preprocess_raw_tensor(batch_x, channels=channels, normalize=normalize))
        loss = torch.nn.functional.cross_entropy(logits, batch_y)
        model.zero_grad(set_to_none=True)
        loss.backward()

        adv_x = batch_x + epsilon * batch_x.grad.sign()
        logits_adv = model(preprocess_raw_tensor(adv_x.detach(), channels=channels, normalize=normalize))
        pred = logits_adv.argmax(dim=1)
        y_true.extend(batch_y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    metrics = calculate_metrics(y_true, y_pred)
    metrics["num_samples"] = int(len(y_true))
    metrics["epsilon"] = float(epsilon)
    return metrics


def evaluate_pgd(
    model: torch.nn.Module,
    X_raw: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    channels: int,
    normalize: bool,
    epsilon: float,
    alpha: float,
    steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    for start in range(0, len(X_raw), batch_size):
        batch_x = torch.tensor(X_raw[start:start + batch_size], dtype=torch.float32, device=device)
        batch_y = torch.tensor(y[start:start + batch_size], dtype=torch.long, device=device)
        adv_x = batch_x.clone().detach()

        for _ in range(steps):
            adv_x.requires_grad_(True)
            logits = model(preprocess_raw_tensor(adv_x, channels=channels, normalize=normalize))
            loss = torch.nn.functional.cross_entropy(logits, batch_y)
            model.zero_grad(set_to_none=True)
            loss.backward()

            step = alpha * adv_x.grad.sign()
            adv_x = adv_x.detach() + step
            delta = torch.clamp(adv_x - batch_x, min=-epsilon, max=epsilon)
            adv_x = batch_x + delta

        logits_adv = model(preprocess_raw_tensor(adv_x.detach(), channels=channels, normalize=normalize))
        pred = logits_adv.argmax(dim=1)
        y_true.extend(batch_y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    metrics = calculate_metrics(y_true, y_pred)
    metrics["num_samples"] = int(len(y_true))
    metrics["epsilon"] = float(epsilon)
    metrics["alpha"] = float(alpha)
    metrics["steps"] = int(steps)
    return metrics


def load_run_context(project_root: Path, results_dir: Path, config_path: Path | None) -> Dict[str, Any]:
    resolved_results_dir = results_dir if results_dir.is_absolute() else project_root / results_dir
    if not resolved_results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {resolved_results_dir}")

    if config_path is None:
        config_path = resolved_results_dir / "config.yaml"
    else:
        config_path = config_path if config_path.is_absolute() else project_root / config_path

    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    dataset_dir = resolve_path(
        dataset_cfg.get("output_dir", f"datasets/{dataset_cfg.get('name', 'oracle')}"),
        project_root,
    )

    if not dataset_dir.exists():
        build_dataset_loaders(config, project_root)

    X = np.load(dataset_dir / "X.npy")
    y = np.load(dataset_dir / "y.npy")
    with open(dataset_dir / "device_mapping.json", "r", encoding="utf-8") as handle:
        device_mapping = json.load(handle)
    with open(dataset_dir / "dataset_info.json", "r", encoding="utf-8") as handle:
        dataset_info = json.load(handle)
    with open(dataset_dir / "split_manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    model_cfg = config["model"]
    model_name = str(model_cfg.get("name", "cnn1d")).lower()
    input_channels = int(dataset_cfg.get("channels", 2))
    model, resolved_model_name = build_model(model_cfg, len(device_mapping), input_channels=input_channels)
    model_path = resolved_results_dir / f"{resolved_model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    return {
        "config": config,
        "dataset_dir": dataset_dir,
        "results_dir": resolved_results_dir,
        "X": X,
        "y": y,
        "manifest": manifest,
        "dataset_info": dataset_info,
        "device_mapping": device_mapping,
        "model": model,
        "model_name": model_name,
        "resolved_model_name": resolved_model_name,
        "device": device,
    }


def select_partition_indices(manifest: List[Dict[str, Any]], partition: str) -> np.ndarray:
    if partition == "eval":
        return np.array([idx for idx, entry in enumerate(manifest) if entry.get("split") in {"val", "test"}], dtype=np.int64)
    return np.array([idx for idx, entry in enumerate(manifest) if entry.get("split") == partition], dtype=np.int64)


def subset_by_indices(X: np.ndarray, y: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X[indices], y[indices]


def train_and_evaluate_split(
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    config: Dict[str, Any],
    seed: int,
    epochs_override: int | None,
    device: torch.device,
) -> Dict[str, Any]:
    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Train and test indices must both be non-empty")

    train_labels = set(np.unique(y[train_indices]).tolist())
    test_labels = set(np.unique(y[test_indices]).tolist())
    if train_labels != test_labels:
        missing = sorted(test_labels - train_labels)
        if missing:
            raise ValueError(f"Held-out split contains labels not seen during training: {missing}")

    set_seed(seed)

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    channels = int(dataset_cfg.get("channels", 2))
    normalize = bool(dataset_cfg.get("normalize", True))
    batch_size = int(dataset_cfg.get("batch_size", 32))
    epochs = int(epochs_override if epochs_override is not None else config["training"].get("epochs", 10))

    model, _ = build_model(model_cfg, int(len(np.unique(y))), input_channels=channels)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"].get("learning_rate", 0.001)))
    loss_fn = torch.nn.CrossEntropyLoss()

    train_dataset = RawSignalDataset(X[train_indices], y[train_indices], channels=channels, normalize=normalize)
    test_dataset = RawSignalDataset(X[test_indices], y[test_indices], channels=channels, normalize=normalize)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    losses = []
    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            labels = batch["y"].to(device)
            logits = model(x)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses.append(running_loss / max(len(train_loader), 1))

    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            labels = batch["y"].to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    metrics = calculate_metrics(y_true, y_pred)
    metrics["num_train"] = int(len(train_indices))
    metrics["num_test"] = int(len(test_indices))
    metrics["epochs"] = int(epochs)
    metrics["final_loss"] = float(losses[-1]) if losses else None
    return metrics


def build_cross_receiver_result(
    X: np.ndarray,
    y: np.ndarray,
    manifest: List[Dict[str, Any]],
    config: Dict[str, Any],
    seed: int,
    epochs_override: int | None,
    device: torch.device,
    holdout_ratio: float,
) -> Dict[str, Any]:
    receivers = sorted({entry.get("receiver_id") for entry in manifest if entry.get("receiver_id")})
    if len(receivers) < 2:
        return {"status": "skipped", "reason": "receiver_id metadata unavailable or only one receiver present"}

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(receivers))
    holdout_count = max(1, int(round(len(shuffled) * holdout_ratio)))
    holdout = set(shuffled[:holdout_count])
    train_indices = np.array([idx for idx, entry in enumerate(manifest) if entry.get("receiver_id") not in holdout], dtype=np.int64)
    test_indices = np.array([idx for idx, entry in enumerate(manifest) if entry.get("receiver_id") in holdout], dtype=np.int64)

    metrics = train_and_evaluate_split(X, y, train_indices, test_indices, config, seed, epochs_override, device)
    metrics["status"] = "ok"
    metrics["holdout_receivers"] = sorted(holdout)
    return metrics


def build_cross_day_result(
    X: np.ndarray,
    y: np.ndarray,
    manifest: List[Dict[str, Any]],
    config: Dict[str, Any],
    seed: int,
    epochs_override: int | None,
    device: torch.device,
) -> Dict[str, Any]:
    days = sorted({entry.get("capture_date") for entry in manifest if entry.get("capture_date")})
    if len(days) < 2:
        return {"status": "skipped", "reason": "capture_date metadata unavailable or only one capture day present"}

    split_at = max(1, len(days) - max(1, int(round(len(days) * 0.3))))
    train_days = set(days[:split_at])
    test_days = set(days[split_at:])
    train_indices = np.array([idx for idx, entry in enumerate(manifest) if entry.get("capture_date") in train_days], dtype=np.int64)
    test_indices = np.array([idx for idx, entry in enumerate(manifest) if entry.get("capture_date") in test_days], dtype=np.int64)

    metrics = train_and_evaluate_split(X, y, train_indices, test_indices, config, seed, epochs_override, device)
    metrics["status"] = "ok"
    metrics["train_days"] = sorted(train_days)
    metrics["test_days"] = sorted(test_days)
    return metrics


def summarize(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "num_samples": int(metrics.get("num_samples", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RF CNN robustness evaluations")
    parser.add_argument("--results-dir", type=Path, required=True, help="Training run directory containing config.yaml and model checkpoint")
    parser.add_argument("--config", type=Path, default=None, help="Optional config override")
    parser.add_argument("--output", type=Path, default=None, help="Optional robustness report path")
    parser.add_argument("--partition", type=str, default="eval", choices=["eval", "train", "val", "test"], help="Partition to perturb for non-retraining tests")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Limit the perturbed evaluation set for quick runs")
    parser.add_argument("--cross-split-epochs", type=int, default=None, help="Override epochs for cross-receiver/day retraining")
    parser.add_argument("--cross-receiver-holdout-ratio", type=float, default=0.3, help="Fraction of receivers to hold out")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    context = load_run_context(project_root, args.results_dir, args.config)
    config = context["config"]
    seed = int(args.seed if args.seed is not None else config["dataset"].get("seed", 42))
    set_seed(seed)

    batch_size = int(args.batch_size if args.batch_size is not None else config["dataset"].get("batch_size", 32))
    channels = int(config["dataset"].get("channels", 2))
    normalize = bool(config["dataset"].get("normalize", True))
    partition_indices = select_partition_indices(context["manifest"], args.partition)
    if len(partition_indices) == 0:
        raise ValueError(f"No samples found for partition '{args.partition}'")

    if args.max_eval_samples is not None:
        partition_indices = partition_indices[: int(args.max_eval_samples)]

    X_eval, y_eval = subset_by_indices(context["X"], context["y"], partition_indices)
    model = context["model"]
    device = context["device"]

    report: Dict[str, Any] = {
        "results_dir": str(context["results_dir"]),
        "dataset_dir": str(context["dataset_dir"]),
        "model_name": context["resolved_model_name"],
        "partition": args.partition,
        "seed": seed,
        "baseline": summarize(evaluate_model(model, X_eval, y_eval, batch_size, channels, normalize, device)),
        "snr_sweep": [],
        "carrier_frequency_offset": [],
        "phase_rotation": [],
        "timing_offset": [],
        "fading_multipath": [],
        "cross_receiver_split": {},
        "cross_day_split": {},
        "adversarial": {},
    }

    for snr_db in [math.inf, 30.0, 20.0, 10.0, 5.0, 0.0]:
        scenario_name = "clean" if math.isinf(snr_db) else f"snr_{int(snr_db)}db"
        metrics = evaluate_model(model, apply_awgn(X_eval, snr_db, seed), y_eval, batch_size, channels, normalize, device)
        report["snr_sweep"].append({"scenario": scenario_name, **summarize(metrics)})

    for offset in [0.0, 0.0005, 0.001, 0.0025, 0.005]:
        metrics = evaluate_model(model, apply_cfo(X_eval, offset), y_eval, batch_size, channels, normalize, device)
        report["carrier_frequency_offset"].append({"normalized_offset": offset, **summarize(metrics)})

    for degrees in [0.0, 15.0, 30.0, 60.0, 90.0]:
        radians = math.radians(degrees)
        metrics = evaluate_model(model, apply_phase_bias(X_eval, radians), y_eval, batch_size, channels, normalize, device)
        report["phase_rotation"].append({"scenario": f"static_{int(degrees)}deg", "phase_degrees": degrees, **summarize(metrics)})

    random_phase_metrics = evaluate_model(
        model,
        apply_random_phase_rotation(X_eval, math.pi, seed),
        y_eval,
        batch_size,
        channels,
        normalize,
        device,
    )
    report["phase_rotation"].append({"scenario": "random_uniform_pm180deg", **summarize(random_phase_metrics)})

    for shift in [-8, -4, 4, 8]:
        metrics = evaluate_model(model, apply_integer_timing_shift(X_eval, shift), y_eval, batch_size, channels, normalize, device)
        report["timing_offset"].append({"scenario": f"integer_shift_{shift}", "shift_samples": shift, **summarize(metrics)})

    for delay in [0.25, 0.5, 1.0]:
        metrics = evaluate_model(model, apply_fractional_delay(X_eval, delay), y_eval, batch_size, channels, normalize, device)
        report["timing_offset"].append({"scenario": f"fractional_delay_{delay}", "fractional_delay": delay, **summarize(metrics)})

    rayleigh = evaluate_model(model, apply_rayleigh_flat_fading(X_eval, seed), y_eval, batch_size, channels, normalize, device)
    report["fading_multipath"].append({"scenario": "rayleigh_flat", **summarize(rayleigh)})

    rician = evaluate_model(model, apply_rician_flat_fading(X_eval, k_factor=5.0, seed=seed), y_eval, batch_size, channels, normalize, device)
    report["fading_multipath"].append({"scenario": "rician_flat_k5", **summarize(rician)})

    multipath = evaluate_model(
        model,
        apply_multipath_channel(X_eval, tap_delays=[0, 1, 3], tap_gains=[1.0, 0.45, 0.2], seed=seed),
        y_eval,
        batch_size,
        channels,
        normalize,
        device,
    )
    report["fading_multipath"].append({"scenario": "multipath_3tap", **summarize(multipath)})

    report["cross_receiver_split"] = build_cross_receiver_result(
        context["X"],
        context["y"],
        context["manifest"],
        config,
        seed,
        args.cross_split_epochs,
        device,
        float(args.cross_receiver_holdout_ratio),
    )
    report["cross_day_split"] = build_cross_day_result(
        context["X"],
        context["y"],
        context["manifest"],
        config,
        seed,
        args.cross_split_epochs,
        device,
    )

    fgsm = evaluate_fgsm(model, X_eval, y_eval, batch_size, channels, normalize, epsilon=0.02, device=device)
    pgd = evaluate_pgd(model, X_eval, y_eval, batch_size, channels, normalize, epsilon=0.02, alpha=0.005, steps=5, device=device)
    report["adversarial"] = {
        "fgsm": summarize(fgsm) | {"epsilon": float(fgsm["epsilon"])},
        "pgd": summarize(pgd) | {
            "epsilon": float(pgd["epsilon"]),
            "alpha": float(pgd["alpha"]),
            "steps": int(pgd["steps"]),
        },
    }

    output_path = args.output
    if output_path is None:
        output_path = context["results_dir"] / "robustness_report.json"
    elif not output_path.is_absolute():
        output_path = project_root / output_path

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved robustness report to {output_path}")


if __name__ == "__main__":
    main()