#!/usr/bin/env python3
"""Train the PathRIR Pruning-MLP.

The input is a directory of room ``.npz`` files produced by
``build_ism_pruning_dataset.py``. The model reads physical features for each
image-source node and predicts both a keep probability and the logarithmic
importance of its subtree.

The output directory contains checkpoints selected by several validation
criteria. ``best_model.pt`` and ``best_by_safe_recall.pt`` contain the same
model.

Example:
  python train_ism_pruning_mlp.py \
    --data-dir ./data/train_order10 \
    --out-dir ./checkpoints/pruning_mlp \
    --epochs 50 --batch-size 8192 --hidden-dim 64 --depth 3 \
    --fn-cost 8.0 --reg-weight 0.25 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    seed: int = 0
    val_frac: float = 0.10
    test_frac: float = 0.10
    max_files: int = 0
    max_train_nodes: int = 0
    max_val_nodes: int = 0
    target: str = "energy_ratio"
    importance_eps: float = 1e-12
    importance_log_floor: float = -12.0
    drop_features: str = "wall_id"
    batch_size: int = 8192
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    depth: int = 3
    dropout: float = 0.05
    cls_weight: float = 1.0
    reg_weight: float = 0.25
    fn_cost: float = 8.0
    reg_keep_weight: float = 2.0
    grad_clip: float = 5.0
    decision_threshold: float = 0.5
    num_workers: int = 0
    device: str = "auto"
    save_every: int = 0

    # Weights used to rank checkpoints by recall and pruning rate.
    # safe_recall_score = recall_keep - safe_fnr_weight * FNR + safe_prune_weight * prune_rate
    safe_fnr_weight: float = 0.25
    safe_prune_weight: float = 0.05


TARGET_TO_KEY = {
    "energy_ratio": "label_subtree_energy_ratio",
    "l2_ratio": "label_subtree_l2_ratio",
    "energy": "label_subtree_energy",
    "peak": "label_subtree_peak",
}


class TinyPruningMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, depth: int = 3, dropout: float = 0.05) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.keep_head = nn.Linear(hidden_dim, 1)
        self.importance_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {"keep_logit": self.keep_head(h), "importance_log_norm": self.importance_head(h)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def decode_feature_names(raw: np.ndarray) -> List[str]:
    out = []
    for x in raw.tolist():
        out.append(x.decode("utf-8") if isinstance(x, bytes) else str(x))
    return out


def read_manifest_paths(data_dir: Path) -> List[Path]:
    manifest = data_dir / "manifest.jsonl"
    paths: List[Path] = []
    if not manifest.exists():
        return paths
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


def find_npz_files(data_dir: Path, max_files: int) -> List[Path]:
    files = read_manifest_paths(data_dir) or sorted(data_dir.glob("room_*.npz"))
    if max_files and max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No room_*.npz files found in {data_dir}")
    return files


def split_files(files: Sequence[Path], val_frac: float, test_frac: float, seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    files = list(files)
    random.Random(seed).shuffle(files)
    n = len(files)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = max(1, n - n_val - n_test)
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    if not val and len(train) > 1:
        val = [train.pop()]
    if not test and len(train) > 1:
        test = [train.pop()]
    return train, val, test


def parse_drop_features(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def select_feature_indices(feature_names: List[str], drop_features: Sequence[str]) -> np.ndarray:
    drop = set(drop_features)
    keep = [i for i, n in enumerate(feature_names) if n not in drop]
    if not keep:
        raise ValueError("All features were dropped")
    return np.asarray(keep, dtype=np.int64)


def subsample_nodes(
    x: np.ndarray,
    y_keep: np.ndarray,
    y_imp: np.ndarray,
    max_nodes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample nodes while keeping all positive examples where possible."""
    if max_nodes <= 0 or x.shape[0] <= max_nodes:
        return x, y_keep, y_imp
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y_keep.reshape(-1) > 0.5)
    neg = np.flatnonzero(y_keep.reshape(-1) <= 0.5)
    if len(pos) >= max_nodes:
        idx = rng.choice(pos, max_nodes, replace=False)
    else:
        neg_sample = rng.choice(neg, min(max_nodes - len(pos), len(neg)), replace=False)
        idx = np.concatenate([pos, neg_sample])
        rng.shuffle(idx)
    return x[idx], y_keep[idx], y_imp[idx]


def load_split_arrays(
    files: Sequence[Path],
    feature_idx: np.ndarray,
    target_key: str,
    max_nodes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    imps: List[np.ndarray] = []
    iterator: Iterable[Path] = tqdm(files, desc="Loading rooms", leave=False) if tqdm else files
    for p in iterator:
        try:
            with np.load(p, allow_pickle=True) as z:
                x = np.asarray(z["node_features"], dtype=np.float32)[:, feature_idx]
                y = np.asarray(z["label_keep"], dtype=np.float32).reshape(-1, 1)
                imp = np.asarray(z[target_key], dtype=np.float32).reshape(-1, 1)
        except Exception as exc:
            print(f"[warning] Could not load {p}: {exc}")
            continue
        if x.shape[0] == y.shape[0] == imp.shape[0]:
            xs.append(x)
            ys.append(y)
            imps.append(imp)
        else:
            print(f"[warning] Skipped {p}: node counts do not match")
    if not xs:
        raise RuntimeError("No valid files loaded for split")
    return subsample_nodes(np.concatenate(xs), np.concatenate(ys), np.concatenate(imps), max_nodes, seed)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def make_log_importance(y: np.ndarray, eps: float, floor: float) -> np.ndarray:
    log_y = np.log10(np.maximum(y, 0.0) + eps).astype(np.float32)
    return np.maximum(log_y, floor).astype(np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, ylog: np.ndarray, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
        torch.from_numpy(ylog.astype(np.float32)),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def multitask_loss(
    out: Dict[str, torch.Tensor],
    y_keep: torch.Tensor,
    y_log_imp_norm: torch.Tensor,
    cfg: TrainConfig,
    pos_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pw = torch.tensor(float(pos_weight), dtype=out["keep_logit"].dtype, device=out["keep_logit"].device)
    cls = F.binary_cross_entropy_with_logits(out["keep_logit"], y_keep, pos_weight=pw)

    reg_each = F.smooth_l1_loss(out["importance_log_norm"], y_log_imp_norm, reduction="none", beta=0.5)
    reg_w = 1.0 + cfg.reg_keep_weight * y_keep
    reg = (reg_each * reg_w).sum() / reg_w.sum().clamp_min(1.0)

    loss = cfg.cls_weight * cls + cfg.reg_weight * reg
    return loss, {
        "loss": float(loss.detach().cpu()),
        "cls_loss": float(cls.detach().cpu()),
        "reg_loss": float(reg.detach().cpu()),
    }


def binary_metrics(prob: np.ndarray, y: np.ndarray, threshold: float) -> Dict[str, float]:
    prob = prob.reshape(-1)
    y = y.reshape(-1).astype(np.int64)
    pred = (prob >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "acc": (tp + tn) / max(tp + fp + tn + fn, 1),
        "precision_keep": precision,
        "recall_keep": recall,
        "specificity_prune": tn / max(tn + fp, 1),
        "f1_keep": f1,
        "false_negative_rate": fn / max(tp + fn, 1),
        "prune_rate": float((pred == 0).mean()),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def auroc(prob: np.ndarray, y: np.ndarray) -> float:
    prob = prob.reshape(-1)
    y = y.reshape(-1).astype(np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(prob)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(prob) + 1, dtype=np.float64)
    sorted_prob = prob[order]
    start = 0
    while start < len(prob):
        end = start + 1
        while end < len(prob) and sorted_prob[end] == sorted_prob[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    pos_weight: float,
    log_mean: float,
    log_std: float,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    cls_losses: List[float] = []
    reg_losses: List[float] = []
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    pred_logs: List[np.ndarray] = []
    true_logs: List[np.ndarray] = []
    for x, y, ylog in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ylog = ylog.to(device, non_blocking=True)
        out = model(x)
        loss, stats = multitask_loss(out, y, ylog, cfg, pos_weight)
        losses.append(stats["loss"])
        cls_losses.append(stats["cls_loss"])
        reg_losses.append(stats["reg_loss"])
        probs.append(torch.sigmoid(out["keep_logit"]).cpu().numpy())
        labels.append(y.cpu().numpy())
        pred_logs.append(out["importance_log_norm"].cpu().numpy())
        true_logs.append(ylog.cpu().numpy())

    prob = np.concatenate(probs)
    y = np.concatenate(labels)
    pred_log = np.concatenate(pred_logs) * log_std + log_mean
    true_log = np.concatenate(true_logs) * log_std + log_mean
    m = binary_metrics(prob, y, cfg.decision_threshold)
    m.update({
        "loss": float(np.mean(losses)),
        "cls_loss": float(np.mean(cls_losses)),
        "reg_loss": float(np.mean(reg_losses)),
        "auroc": auroc(prob, y),
        "rmse_log10_importance": float(np.sqrt(np.mean((pred_log - true_log) ** 2))),
        "mae_log10_importance": float(np.mean(np.abs(pred_log - true_log))),
        "num_nodes": float(len(y)),
        "positive_frac": float(y.mean()),
    })
    return m


def format_metrics(prefix: str, m: Dict[str, float]) -> str:
    keys = [
        "loss",
        "cls_loss",
        "reg_loss",
        "auroc",
        "recall_keep",
        "precision_keep",
        "f1_keep",
        "false_negative_rate",
        "prune_rate",
        "rmse_log10_importance",
    ]
    return " | ".join([prefix] + [f"{k}={m[k]:.4f}" for k in keys if k in m and np.isfinite(float(m[k]))])


def safe_recall_score(metrics: Dict[str, float], cfg: TrainConfig) -> float:
    return (
        metrics["recall_keep"]
        - cfg.safe_fnr_weight * metrics["false_negative_rate"]
        + cfg.safe_prune_weight * metrics["prune_rate"]
    )


def checkpoint_scores(val: Dict[str, float], cfg: TrainConfig) -> Dict[str, float]:
    """Return checkpoint scores with larger values treated as better."""
    return {
        "safe_recall": safe_recall_score(val, cfg),
        "val_loss": -val["loss"],
        "f1": val["f1_keep"],
        "auroc": val["auroc"],
        "regression": -val["rmse_log10_importance"],
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: TrainConfig,
    feature_names: List[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    log_mean: float,
    log_std: float,
    pos_weight: float,
    criterion_name: str,
    criterion_score: float,
    val_metrics: Dict[str, float],
    train_loss: float,
) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "config": asdict(cfg),
        "feature_names": feature_names,
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "log_imp_mean": float(log_mean),
        "log_imp_std": float(log_std),
        "pos_weight": float(pos_weight),
        "selection_criterion": criterion_name,
        "selection_score": float(criterion_score),
        "val_metrics_at_save": val_metrics,
        "train_loss_at_save": float(train_loss),
        "best_val_metric": float(criterion_score),
        "model_class": "TinyPruningMLP",
    }, path)


def load_checkpoint_into_model(path: Path, model: nn.Module, device: torch.device) -> Dict[str, object]:
    ckpt = torch_load(path, device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_npz_files(Path(cfg.data_dir), cfg.max_files)
    train_files, val_files, test_files = split_files(files, cfg.val_frac, cfg.test_frac, cfg.seed)
    with np.load(train_files[0], allow_pickle=True) as z:
        all_feature_names = decode_feature_names(z["feature_names"])
    feature_idx = select_feature_indices(all_feature_names, parse_drop_features(cfg.drop_features))
    feature_names = [all_feature_names[i] for i in feature_idx.tolist()]
    target_key = TARGET_TO_KEY[cfg.target]

    print(f"Dataset: {cfg.data_dir}")
    print(f"Room split: train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    print(f"Target: {cfg.target} ({target_key}); features={len(feature_names)}; device={device}")

    xtr, ytr, itr = load_split_arrays(train_files, feature_idx, target_key, cfg.max_train_nodes, cfg.seed + 11)
    xv, yv, iv = load_split_arrays(val_files, feature_idx, target_key, cfg.max_val_nodes, cfg.seed + 17)
    xt, yt, it = load_split_arrays(test_files, feature_idx, target_key, cfg.max_val_nodes, cfg.seed + 23)

    fmean = xtr.mean(axis=0, keepdims=True).astype(np.float32)
    fstd = np.maximum(xtr.std(axis=0, keepdims=True), 1e-6).astype(np.float32)
    xtr = standardize(xtr, fmean, fstd)
    xv = standardize(xv, fmean, fstd)
    xt = standardize(xt, fmean, fstd)

    ytr_log = make_log_importance(itr, cfg.importance_eps, cfg.importance_log_floor)
    yv_log = make_log_importance(iv, cfg.importance_eps, cfg.importance_log_floor)
    yt_log = make_log_importance(it, cfg.importance_eps, cfg.importance_log_floor)
    log_mean = float(ytr_log.mean())
    log_std = float(max(ytr_log.std(), 1e-6))
    ytr_logn = ((ytr_log - log_mean) / log_std).astype(np.float32)
    yv_logn = ((yv_log - log_mean) / log_std).astype(np.float32)
    yt_logn = ((yt_log - log_mean) / log_std).astype(np.float32)

    pos = float(ytr.sum())
    neg = float(ytr.size - ytr.sum())
    pos_weight = cfg.fn_cost * neg / max(pos, 1.0)
    print(
        f"Node split: train={xtr.shape[0]} val={xv.shape[0]} test={xt.shape[0]} "
        f"positive_frac={float(ytr.mean()):.6f} pos_weight={pos_weight:.3f}"
    )

    train_loader = make_loader(xtr, ytr, ytr_logn, cfg.batch_size, True, cfg.num_workers)
    val_loader = make_loader(xv, yv, yv_logn, cfg.batch_size, False, cfg.num_workers)
    test_loader = make_loader(xt, yt, yt_logn, cfg.batch_size, False, cfg.num_workers)

    model = TinyPruningMLP(len(feature_names), cfg.hidden_dim, cfg.depth, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    metadata = {
        "config": asdict(cfg),
        "feature_names": feature_names,
        "all_feature_names": all_feature_names,
        "target_key": target_key,
        "train_files": [str(p) for p in train_files],
        "val_files": [str(p) for p in val_files],
        "test_files": [str(p) for p in test_files],
        "feature_mean": fmean.reshape(-1).astype(float).tolist(),
        "feature_std": fstd.reshape(-1).astype(float).tolist(),
        "log_imp_mean": log_mean,
        "log_imp_std": log_std,
        "pos_weight": pos_weight,
        "checkpoint_policy": {
            "best_by_safe_recall.pt": "maximize recall_keep - safe_fnr_weight * FNR + safe_prune_weight * prune_rate",
            "best_by_val_loss.pt": "minimize validation total multitask loss",
            "best_by_f1.pt": "maximize keep-class F1 at decision_threshold",
            "best_by_auroc.pt": "maximize threshold-independent AUROC",
            "best_by_regression.pt": "minimize RMSE of log10 subtree importance",
            "best_model.pt": "same checkpoint as best_by_safe_recall.pt",
        },
    }
    (out_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    hist_path = out_dir / "history.jsonl"
    if hist_path.exists():
        hist_path.unlink()

    ckpt_names = {
        "safe_recall": "best_by_safe_recall.pt",
        "val_loss": "best_by_val_loss.pt",
        "f1": "best_by_f1.pt",
        "auroc": "best_by_auroc.pt",
        "regression": "best_by_regression.pt",
    }
    best_scores = {name: -float("inf") for name in ckpt_names}
    best_epochs = {name: -1 for name in ckpt_names}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        stats_list = []
        t0 = time.time()
        iterator: Iterable = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False) if tqdm else train_loader
        for x, y, ylog in iterator:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            ylog = ylog.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss, stats = multitask_loss(out, y, ylog, cfg, pos_weight)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            stats_list.append(stats)
            if tqdm:
                iterator.set_postfix(loss=f"{stats['loss']:.4f}")
        sched.step()

        val = evaluate(model, val_loader, device, cfg, pos_weight, log_mean, log_std)
        train_loss = float(np.mean([s["loss"] for s in stats_list]))
        scores = checkpoint_scores(val, cfg)
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val,
            "checkpoint_scores": scores,
            "lr": float(sched.get_last_lr()[0]),
            "elapsed_sec": time.time() - t0,
        }
        with hist_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} " + format_metrics("val", val) + f" elapsed={rec['elapsed_sec']:.1f}s")

        saved_this_epoch: List[str] = []
        for criterion, score in scores.items():
            if score > best_scores[criterion]:
                best_scores[criterion] = score
                best_epochs[criterion] = epoch
                ckpt_path = out_dir / ckpt_names[criterion]
                save_checkpoint(
                    ckpt_path,
                    model,
                    opt,
                    epoch,
                    cfg,
                    feature_names,
                    fmean,
                    fstd,
                    log_mean,
                    log_std,
                    pos_weight,
                    criterion,
                    score,
                    val,
                    train_loss,
                )
                saved_this_epoch.append(ckpt_names[criterion])

                # Save the recall-selected model under the default name as well.
                if criterion == "safe_recall":
                    save_checkpoint(
                        out_dir / "best_model.pt",
                        model,
                        opt,
                        epoch,
                        cfg,
                        feature_names,
                        fmean,
                        fstd,
                        log_mean,
                        log_std,
                        pos_weight,
                        criterion,
                        score,
                        val,
                        train_loss,
                    )

        if saved_this_epoch:
            print("Saved " + ", ".join(saved_this_epoch) + f" at epoch {epoch}")

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(
                out_dir / f"model_epoch_{epoch:03d}.pt",
                model,
                opt,
                epoch,
                cfg,
                feature_names,
                fmean,
                fstd,
                log_mean,
                log_std,
                pos_weight,
                f"epoch_{epoch:03d}",
                float(scores["safe_recall"]),
                val,
                train_loss,
            )

    # Evaluate the checkpoint selected by each validation criterion.
    test_by_checkpoint: Dict[str, Dict[str, object]] = {}
    for criterion, filename in ckpt_names.items():
        ckpt_path = out_dir / filename
        if not ckpt_path.exists():
            continue
        ckpt = load_checkpoint_into_model(ckpt_path, model, device)
        test = evaluate(model, test_loader, device, cfg, pos_weight, log_mean, log_std)
        test_by_checkpoint[filename] = {
            "criterion": criterion,
            "epoch": int(ckpt.get("epoch", -1)),
            "selection_score": float(ckpt.get("selection_score", float("nan"))),
            "val_metrics_at_save": ckpt.get("val_metrics_at_save", {}),
            "test": test,
        }
        print(f"test {filename} epoch={ckpt.get('epoch', -1)} " + format_metrics("test", test))

    summary = {
        "best_epochs": best_epochs,
        "best_scores": best_scores,
        "test_by_checkpoint": test_by_checkpoint,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved training summary: {out_dir / 'summary.json'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the PathRIR Pruning-MLP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--max-train-nodes", type=int, default=0)
    p.add_argument("--max-val-nodes", type=int, default=0)
    p.add_argument("--target", choices=sorted(TARGET_TO_KEY.keys()), default="energy_ratio")
    p.add_argument("--importance-eps", type=float, default=1e-12)
    p.add_argument("--importance-log-floor", type=float, default=-12.0)
    p.add_argument("--drop-features", default="wall_id")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--cls-weight", type=float, default=1.0)
    p.add_argument("--reg-weight", type=float, default=0.25)
    p.add_argument("--fn-cost", type=float, default=8.0)
    p.add_argument("--reg-keep-weight", type=float, default=2.0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--decision-threshold", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--safe-fnr-weight", type=float, default=0.25)
    p.add_argument("--safe-prune-weight", type=float, default=0.05)
    return p.parse_args()


def args_to_config(a: argparse.Namespace) -> TrainConfig:
    return TrainConfig(**vars(a))


def main() -> None:
    train(args_to_config(parse_args()))


if __name__ == "__main__":
    main()
