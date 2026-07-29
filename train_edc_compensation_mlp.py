#!/usr/bin/env python3
"""Train the PathRIR Compensation-MLP.

For each room and microphone, the model learns the time-binned residual energy
between the full and pruned RIRs. Its inputs describe the room geometry,
source and microphone positions, pruning result, and pruned-RIR energy.

The script needs a dataset from ``build_ism_pruning_dataset.py`` and a trained
Pruning-MLP checkpoint.

Example:
  python train_edc_compensation_mlp.py \
      --data-dir ./data/train_order10 \
      --pruning-ckpt ./checkpoints/pruning_mlp/best_by_safe_recall.pt \
      --out-dir ./checkpoints/comp_mlp \
      --epochs 80 \
      --batch-size 512 \
      --num-bins 64 \
      --device cuda

Use the same pruning settings for this script and evaluation. If
``--cache-npz`` is set, reuse that cache only with the same source dataset and
pruning settings.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

import evaluate_pathrir as eval_backend


# Configuration


@dataclass
class TrainConfig:
    data_dir: str
    pruning_ckpt: str
    out_dir: str
    seed: int = 0
    val_frac: float = 0.10
    test_frac: float = 0.10
    max_files: int = 0
    max_train_samples: int = 0
    max_val_samples: int = 0

    # Compensation target
    num_bins: int = 64
    target_eps: float = 1e-14
    target_log_floor: float = -14.0
    order_feature_max: int = 15

    # Pruning policy used to create residual targets
    decision_mode: str = "prob"  # prob | importance | either | both
    prob_threshold: float = 0.5
    importance_threshold: float = 1e-4
    early_keep_order: int = 1
    min_keep_rate_after: float = 0.20
    max_keep_rate_after: float = 0.50
    min_keep_count_after: int = 48
    budget_score: str = "importance"  # importance | prob
    budget_relative_to: str = "visible"  # visible | all
    min_visible_keep_count: int = 0

    # Model and training
    batch_size: int = 512
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    depth: int = 4
    dropout: float = 0.05
    mse_weight: float = 1.0
    edc_weight: float = 0.5
    nonzero_weight: float = 2.0
    grad_clip: float = 5.0
    num_workers: int = 0
    device: str = "auto"
    save_every: int = 0
    cache_npz: str = ""


# Reproducibility and device selection


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Normalize the shorthand "cuda0" to PyTorch's "cuda:0" form.
    if device_arg.startswith("cuda") and device_arg != "cuda" and ":" not in device_arg:
        suffix = device_arg[len("cuda") :]
        if suffix.isdigit():
            return torch.device(f"cuda:{suffix}")
    return torch.device(device_arg)


# Model definitions


class TinyPruningMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, depth: int = 3, dropout: float = 0.05) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.keep_head = nn.Linear(hidden_dim, 1)
        self.importance_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "keep_logit": self.keep_head(h),
            "importance_log_norm": self.importance_head(h),
        }


class CompensationMLP(nn.Module):
    """Predict normalized log residual-energy bins from scene and pruning features."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, depth: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def torch_load_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_pruning_model(path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, object]]:
    ckpt = torch_load_checkpoint(path, device)
    cfg = ckpt.get("config", {})
    feature_names = list(ckpt["feature_names"])
    model = TinyPruningMLP(
        input_dim=len(feature_names),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        depth=int(cfg.get("depth", 3)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    aux = {
        "feature_names": feature_names,
        "feature_mean": np.asarray(ckpt["feature_mean"], dtype=np.float32).reshape(1, -1),
        "feature_std": np.asarray(ckpt["feature_std"], dtype=np.float32).reshape(1, -1),
        "log_imp_mean": float(ckpt.get("log_imp_mean", 0.0)),
        "log_imp_std": float(ckpt.get("log_imp_std", 1.0)),
    }
    return model, aux


# Data loading


def read_manifest_paths(data_dir: Path) -> List[Path]:
    manifest = data_dir / "manifest.jsonl"
    if not manifest.exists():
        return []
    paths: List[Path] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "ok":
                continue
            p = Path(str(rec.get("path", "")))
            if not p.is_absolute():
                p = data_dir / p
            if p.exists() and p.suffix == ".npz":
                paths.append(p)
    return sorted(set(paths))


def find_npz_files(data_dir: Path, max_files: int, seed: int) -> List[Path]:
    files = read_manifest_paths(data_dir)
    if not files:
        files = sorted(data_dir.glob("room_*.npz"))
    if not files:
        raise FileNotFoundError(f"No room_*.npz files found in {data_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    if max_files and max_files > 0:
        files = files[:max_files]
    return files


def split_files(files: Sequence[Path], val_frac: float, test_frac: float, seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n = len(files)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = max(1, n - n_val - n_test)
    train = files[:n_train]
    val = files[n_train : n_train + n_val]
    test = files[n_train + n_val :]
    if not val and len(train) > 1:
        val = [train.pop()]
    if not test and len(train) > 1:
        test = [train.pop()]
    return train, val, test


# Compensation features and targets


def energy_bins(x: np.ndarray, num_bins: int) -> np.ndarray:
    n = len(x)
    edges = np.linspace(0, n, num_bins + 1, dtype=np.int64)
    e = np.zeros(num_bins, dtype=np.float32)
    xx = x.astype(np.float64) ** 2
    for b in range(num_bins):
        s, t = int(edges[b]), int(edges[b + 1])
        if t > s:
            e[b] = float(np.sum(xx[s:t]))
    return e


def log10_safe(x: np.ndarray | float, eps: float = 1e-20) -> np.ndarray | float:
    return np.log10(np.maximum(x, 0.0) + eps)


def traincfg_to_evalcfg(cfg: TrainConfig) -> "eval_backend.EvalConfig":
    """Create the evaluation settings used to generate training targets."""
    return eval_backend.EvalConfig(
        data_dir="", ckpt="", out_dir="",
        batch_size=8192,
        decision_mode=cfg.decision_mode,
        prob_threshold=cfg.prob_threshold,
        importance_threshold=cfg.importance_threshold,
        order_budget_mode="score",
        early_keep_order=cfg.early_keep_order,
        min_keep_rate_after=cfg.min_keep_rate_after,
        max_keep_rate_after=cfg.max_keep_rate_after,
        min_keep_count_after=cfg.min_keep_count_after,
        budget_score=cfg.budget_score,
        budget_relative_to=cfg.budget_relative_to,
        min_visible_keep_count=cfg.min_visible_keep_count,
    )


def build_one_room_samples(
    path: Path,
    pruning_model: nn.Module,
    pruning_aux: Dict[str, object],
    device: torch.device,
    cfg: TrainConfig,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, float]]:
    """Create compensation samples with the pruning policy used at inference."""
    eval_cfg = traincfg_to_evalcfg(cfg)
    scene = eval_backend.load_geometry_from_npz(path, include_saved_ref=True)
    pruned_rir, stats, data, _prob, imp = eval_backend.run_pruned_online_ism(
        scene, pruning_model, pruning_aux, device, eval_cfg
    )

    # Generate the complete feature vector and keep its column names.
    comp_aux: Dict[str, object] = {
        "num_bins": int(cfg.num_bins),
        "config": {"order_feature_max": int(cfg.order_feature_max)},
        "feature_names": None,
    }
    x_all = eval_backend.build_compensation_features(data, pruned_rir, stats, imp, comp_aux)
    feature_names = list(comp_aux["feature_names"])  # type: ignore[arg-type]

    # Use the stored pyroomacoustics RIR as the full reference.
    full_rir = np.asarray(scene["full_rirs"], dtype=np.float32)
    if not np.any(np.abs(full_rir) > 0):
        raise RuntimeError(
            f"{path.name} does not contain a reference RIR. "
            "Generate the dataset again with build_ism_pruning_dataset.py."
        )
    rir_len = min(full_rir.shape[1], pruned_rir.shape[1])
    residual = full_rir[:, :rir_len] - pruned_rir[:, :rir_len]

    ys: List[np.ndarray] = []
    for mic in range(residual.shape[0]):
        target_energy = energy_bins(residual[mic], cfg.num_bins)
        target_log = np.asarray(log10_safe(target_energy, cfg.target_eps), dtype=np.float32)
        target_log = np.maximum(target_log, float(cfg.target_log_floor)).astype(np.float32)
        ys.append(target_log)

    n_nodes = int(np.asarray(data["node_order"]).shape[0])
    kept = np.asarray(stats["kept_mask"], dtype=bool)
    room_stats = {
        "num_nodes": float(n_nodes),
        "kept_nodes": float(kept.sum()),
        "missing_nodes": float((~kept).sum()),
        "expanded_node_reduction": float(stats.get("expanded_node_reduction", 0.0)),
        "contribution_node_reduction": float(stats.get("contribution_node_reduction", 0.0)),
    }
    return x_all, np.stack(ys, axis=0), feature_names, room_stats


def build_compensation_dataset(
    files: Sequence[Path],
    pruning_model: nn.Module,
    pruning_aux: Dict[str, object],
    device: torch.device,
    cfg: TrainConfig,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, float]]]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    stats: List[Dict[str, float]] = []
    feature_names: Optional[List[str]] = None

    iterator: Iterable[Path]
    if tqdm is not None:
        iterator = tqdm(files, desc=desc, leave=False)
    else:
        iterator = files

    for p in iterator:
        try:
            x, y, names, st = build_one_room_samples(p, pruning_model, pruning_aux, device, cfg)
        except Exception as exc:
            print(f"[warning] Skipped {p.name}: {exc}", file=sys.stderr)
            continue
        if feature_names is None:
            feature_names = names
        xs.append(x)
        ys.append(y)
        stats.append(st)

    if not xs or feature_names is None:
        raise RuntimeError(f"No valid samples were built for {desc}")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), feature_names, stats


def subsample(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(x.shape[0]), size=max_samples, replace=False)
    return x[idx], y[idx]


def compute_standardizer(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def make_loader(x: np.ndarray, y_norm: np.ndarray, y_log: np.ndarray, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y_norm.astype(np.float32)),
        torch.from_numpy(y_log.astype(np.float32)),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# Loss and metrics


def edc_log_from_log_bins(log_bins: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    energy = torch.pow(10.0, log_bins)
    rev_cum = torch.cumsum(torch.flip(energy, dims=[1]), dim=1)
    cum = torch.flip(rev_cum, dims=[1])
    return torch.log10(torch.clamp(cum, min=eps))


def compensation_loss(
    pred_norm: torch.Tensor,
    y_norm: torch.Tensor,
    y_log: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_log = pred_norm * y_std + y_mean

    # Give more weight to bins with meaningful residual energy.
    nonzero = (y_log > cfg.target_log_floor + 0.5).float()
    weights = 1.0 + cfg.nonzero_weight * nonzero
    mse_each = F.smooth_l1_loss(pred_norm, y_norm, reduction="none", beta=0.5)
    mse_loss = (mse_each * weights).sum() / weights.sum().clamp_min(1.0)

    pred_edc = edc_log_from_log_bins(pred_log)
    true_edc = edc_log_from_log_bins(y_log)
    edc_loss = F.smooth_l1_loss(pred_edc, true_edc, reduction="mean", beta=0.25)

    loss = cfg.mse_weight * mse_loss + cfg.edc_weight * edc_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "mse_loss": float(mse_loss.detach().cpu()),
        "edc_loss": float(edc_loss.detach().cpu()),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    y_mean_np: np.ndarray,
    y_std_np: np.ndarray,
) -> Dict[str, float]:
    model.eval()
    y_mean = torch.from_numpy(y_mean_np.astype(np.float32)).to(device)
    y_std = torch.from_numpy(y_std_np.astype(np.float32)).to(device)
    losses: List[float] = []
    mse_losses: List[float] = []
    edc_losses: List[float] = []
    pred_logs: List[np.ndarray] = []
    true_logs: List[np.ndarray] = []

    for x, y_norm, y_log in loader:
        x = x.to(device, non_blocking=True)
        y_norm = y_norm.to(device, non_blocking=True)
        y_log = y_log.to(device, non_blocking=True)
        pred_norm = model(x)
        loss, st = compensation_loss(pred_norm, y_norm, y_log, y_mean, y_std, cfg)
        losses.append(st["loss"])
        mse_losses.append(st["mse_loss"])
        edc_losses.append(st["edc_loss"])
        pred_log = (pred_norm * y_std + y_mean).detach().cpu().numpy()
        pred_logs.append(pred_log)
        true_logs.append(y_log.detach().cpu().numpy())

    pred = np.concatenate(pred_logs, axis=0)
    true = np.concatenate(true_logs, axis=0)
    rmse_log = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae_log = float(np.mean(np.abs(pred - true)))

    pred_energy = np.power(10.0, pred)
    true_energy = np.power(10.0, true)
    total_pred = pred_energy.sum(axis=1)
    total_true = true_energy.sum(axis=1)
    total_energy_log_rmse = float(np.sqrt(np.mean((np.log10(total_pred + 1e-20) - np.log10(total_true + 1e-20)) ** 2)))

    pred_edc = np.log10(np.maximum(np.cumsum(pred_energy[:, ::-1], axis=1)[:, ::-1], 1e-20))
    true_edc = np.log10(np.maximum(np.cumsum(true_energy[:, ::-1], axis=1)[:, ::-1], 1e-20))
    edc_rmse_log = float(np.sqrt(np.mean((pred_edc - true_edc) ** 2)))

    return {
        "loss": float(np.mean(losses)),
        "mse_loss": float(np.mean(mse_losses)),
        "edc_loss": float(np.mean(edc_losses)),
        "rmse_log10_bin_energy": rmse_log,
        "mae_log10_bin_energy": mae_log,
        "rmse_log10_total_energy": total_energy_log_rmse,
        "rmse_log10_edc": edc_rmse_log,
        "num_samples": float(true.shape[0]),
    }


def format_metrics(prefix: str, m: Dict[str, float]) -> str:
    keys = ["loss", "mse_loss", "edc_loss", "rmse_log10_bin_energy", "rmse_log10_total_energy", "rmse_log10_edc"]
    return " | ".join([prefix] + [f"{k}={m[k]:.4f}" for k in keys if k in m])


# Checkpoints


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: TrainConfig,
    feature_names: List[str],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    best_metrics: Dict[str, float],
) -> None:
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "config": asdict(cfg),
        "feature_names": feature_names,
        "feature_mean": x_mean.astype(np.float32),
        "feature_std": x_std.astype(np.float32),
        "target_mean": y_mean.astype(np.float32),
        "target_std": y_std.astype(np.float32),
        "target": "log10_missing_residual_energy_bins",
        "model_class": "CompensationMLP",
        "best_metrics": best_metrics,
    }
    torch.save(ckpt, path)


# Training


def maybe_load_cache(cache_path: Path) -> Optional[Dict[str, object]]:
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=True) as z:
        return {
            "x_train": np.asarray(z["x_train"], dtype=np.float32),
            "y_train": np.asarray(z["y_train"], dtype=np.float32),
            "x_val": np.asarray(z["x_val"], dtype=np.float32),
            "y_val": np.asarray(z["y_val"], dtype=np.float32),
            "x_test": np.asarray(z["x_test"], dtype=np.float32),
            "y_test": np.asarray(z["y_test"], dtype=np.float32),
            "feature_names": [str(x) for x in z["feature_names"].tolist()],
        }


def save_cache(cache_path: Path, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        feature_names=np.asarray(feature_names, dtype=object),
    )


def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pruning_model, pruning_aux = load_pruning_model(Path(cfg.pruning_ckpt), device=device)

    cache_data = None
    if cfg.cache_npz:
        cache_data = maybe_load_cache(Path(cfg.cache_npz))

    if cache_data is not None:
        print(f"Loaded cached compensation dataset: {cfg.cache_npz}")
        x_train = cache_data["x_train"]  # type: ignore[assignment]
        y_train = cache_data["y_train"]  # type: ignore[assignment]
        x_val = cache_data["x_val"]  # type: ignore[assignment]
        y_val = cache_data["y_val"]  # type: ignore[assignment]
        x_test = cache_data["x_test"]  # type: ignore[assignment]
        y_test = cache_data["y_test"]  # type: ignore[assignment]
        feature_names = cache_data["feature_names"]  # type: ignore[assignment]
        train_files, val_files, test_files = [], [], []
    else:
        files = find_npz_files(Path(cfg.data_dir), max_files=cfg.max_files, seed=cfg.seed)
        train_files, val_files, test_files = split_files(files, cfg.val_frac, cfg.test_frac, cfg.seed)
        print(f"Room split: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

        x_train, y_train, feature_names, train_stats = build_compensation_dataset(train_files, pruning_model, pruning_aux, device, cfg, "Training samples")
        x_val, y_val, _, val_stats = build_compensation_dataset(val_files, pruning_model, pruning_aux, device, cfg, "Validation samples")
        x_test, y_test, _, test_stats = build_compensation_dataset(test_files, pruning_model, pruning_aux, device, cfg, "Test samples")

        x_train, y_train = subsample(x_train, y_train, cfg.max_train_samples, cfg.seed + 1)
        x_val, y_val = subsample(x_val, y_val, cfg.max_val_samples, cfg.seed + 2)
        x_test, y_test = subsample(x_test, y_test, cfg.max_val_samples, cfg.seed + 3)

        if cfg.cache_npz:
            save_cache(Path(cfg.cache_npz), x_train, y_train, x_val, y_val, x_test, y_test, feature_names)
            print(f"Saved cached compensation dataset: {cfg.cache_npz}")

    x_mean, x_std = compute_standardizer(x_train)
    x_train_n = standardize(x_train, x_mean, x_std)
    x_val_n = standardize(x_val, x_mean, x_std)
    x_test_n = standardize(x_test, x_mean, x_std)

    y_mean = y_train.mean(axis=0, keepdims=True).astype(np.float32)
    y_std = np.maximum(y_train.std(axis=0, keepdims=True), 1e-6).astype(np.float32)
    y_train_n = ((y_train - y_mean) / y_std).astype(np.float32)
    y_val_n = ((y_val - y_mean) / y_std).astype(np.float32)
    y_test_n = ((y_test - y_mean) / y_std).astype(np.float32)

    print(f"Sample split: train={x_train.shape[0]} val={x_val.shape[0]} test={x_test.shape[0]}")
    print(f"Model: input_features={x_train.shape[1]} output_bins={y_train.shape[1]} device={device}")

    train_loader = make_loader(x_train_n, y_train_n, y_train, cfg.batch_size, True, cfg.num_workers)
    val_loader = make_loader(x_val_n, y_val_n, y_val, cfg.batch_size, False, cfg.num_workers)
    test_loader = make_loader(x_test_n, y_test_n, y_test, cfg.batch_size, False, cfg.num_workers)

    model = CompensationMLP(
        input_dim=x_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))

    metadata = {
        "config": asdict(cfg),
        "feature_names": feature_names,
        "input_dim": int(x_train.shape[1]),
        "output_bins": int(y_train.shape[1]),
        "train_files": [str(p) for p in train_files],
        "val_files": [str(p) for p in val_files],
        "test_files": [str(p) for p in test_files],
    }
    (out_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    y_mean_t = torch.from_numpy(y_mean).to(device)
    y_std_t = torch.from_numpy(y_std).to(device)
    best_val = float("inf")
    best_edc = float("inf")
    history_path = out_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        start = time.time()
        stats_epoch: List[Dict[str, float]] = []
        iterator: Iterable
        if tqdm is not None:
            iterator = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        else:
            iterator = train_loader

        for x, y_norm, y_log in iterator:
            x = x.to(device, non_blocking=True)
            y_norm = y_norm.to(device, non_blocking=True)
            y_log = y_log.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss, st = compensation_loss(pred, y_norm, y_log, y_mean_t, y_std_t, cfg)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            stats_epoch.append(st)
            if tqdm is not None:
                iterator.set_postfix(loss=f"{st['loss']:.4f}")

        scheduler.step()
        train_loss = float(np.mean([s["loss"] for s in stats_epoch]))
        val_metrics = evaluate_model(model, val_loader, device, cfg, y_mean, y_std)
        elapsed = time.time() - start

        rec = {"epoch": epoch, "train_loss": train_loss, "val": val_metrics, "lr": float(scheduler.get_last_lr()[0]), "elapsed_sec": elapsed}
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} " + format_metrics("val", val_metrics) + f" elapsed={elapsed:.1f}s")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(out_dir / "best_by_val_loss.pt", model, optimizer, epoch, cfg, feature_names, x_mean, x_std, y_mean, y_std, val_metrics)
            save_checkpoint(out_dir / "best_model.pt", model, optimizer, epoch, cfg, feature_names, x_mean, x_std, y_mean, y_std, val_metrics)
            print(f"Saved best_by_val_loss.pt at epoch {epoch}")

        if val_metrics["rmse_log10_edc"] < best_edc:
            best_edc = val_metrics["rmse_log10_edc"]
            save_checkpoint(out_dir / "best_by_edc_loss.pt", model, optimizer, epoch, cfg, feature_names, x_mean, x_std, y_mean, y_std, val_metrics)
            print(f"Saved best_by_edc_loss.pt at epoch {epoch}")

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(out_dir / f"model_epoch_{epoch:03d}.pt", model, optimizer, epoch, cfg, feature_names, x_mean, x_std, y_mean, y_std, val_metrics)

    # Evaluate the checkpoint selected by EDC error.
    best_path = out_dir / "best_by_edc_loss.pt"
    ckpt = torch_load_checkpoint(best_path, device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate_model(model, test_loader, device, cfg, y_mean, y_std)
    print(format_metrics("test(best_by_edc_loss)", test_metrics))

    summary = {
        "best_val_loss": best_val,
        "best_val_edc_rmse_log10": best_edc,
        "test_best_by_edc_loss": test_metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


# Command-line interface


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the PathRIR Compensation-MLP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--pruning-ckpt", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)

    p.add_argument("--num-bins", type=int, default=64)
    p.add_argument("--target-eps", type=float, default=1e-14)
    p.add_argument("--target-log-floor", type=float, default=-14.0)
    p.add_argument("--order-feature-max", type=int, default=15)

    p.add_argument("--decision-mode", type=str, choices=["prob", "importance", "either", "both"], default="prob")
    p.add_argument("--prob-threshold", type=float, default=0.5)
    p.add_argument("--importance-threshold", type=float, default=1e-4)
    p.add_argument("--early-keep-order", type=int, default=1)
    p.add_argument("--min-keep-rate-after", type=float, default=0.20)
    p.add_argument("--max-keep-rate-after", type=float, default=0.50)
    p.add_argument("--min-keep-count-after", type=int, default=48)
    p.add_argument("--budget-score", type=str, choices=["importance", "prob"], default="importance")
    p.add_argument("--budget-relative-to", type=str, choices=["visible", "all"], default="visible",
                   help="Candidate set used to calculate per-order keep rates. Use the same value for evaluation.")
    p.add_argument("--min-visible-keep-count", type=int, default=0,
                   help="Minimum visible candidates kept per order after applying the budget.")

    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--mse-weight", type=float, default=1.0)
    p.add_argument("--edc-weight", type=float, default=0.5)
    p.add_argument("--nonzero-weight", type=float, default=2.0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--cache-npz", type=str, default="",
                   help="Optional file for saving and reusing compensation samples.")
    return p.parse_args()


def args_to_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(**vars(args))


def main() -> None:
    args = parse_args()
    cfg = args_to_config(args)
    train(cfg)


if __name__ == "__main__":
    main()
