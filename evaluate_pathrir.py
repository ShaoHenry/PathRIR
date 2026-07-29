#!/usr/bin/env python3
"""Evaluate PathRIR on a saved room dataset.

With both checkpoints, the script compares Self-Full-ISM, Prune-RIR, and
PathRIR. It reports image-source reduction, runtime, cosine distance, NMSE,
EDC error, RT60 error, and DRR error for each room.

Example:
  python evaluate_pathrir.py \
    --data-dir ./data/iwaenc_testset_order10 \
    --ckpt ./checkpoints/pruning_mlp/best_by_safe_recall.pt \
    --comp-ckpt ./checkpoints/comp_mlp/best_by_edc_loss.pt \
    --out-dir ./results/table1 \
    --max-files 20 \
    --device cuda

Omit ``--comp-ckpt`` when only the full and pruning-only results are needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
import tracemalloc
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from polygon_ism_engine import ExtrudedPolygonRoom, ISMTree

try:
    import pyroomacoustics as pra
except ImportError:  # pragma: no cover
    pra = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# Configuration


@dataclass
class EvalConfig:
    data_dir: str
    ckpt: str
    out_dir: str
    comp_ckpt: str = ""
    max_files: int = 0
    seed: int = 0
    device: str = "auto"
    batch_size: int = 8192

    # Initial pruning decision
    decision_mode: str = "prob"  # prob | importance | either | both
    prob_threshold: float = 0.5
    importance_threshold: float = 1e-4

    # Per-order keep budget
    order_budget_mode: str = "score"  # none | score
    early_keep_order: int = 1
    min_keep_rate_after: float = 0.20
    max_keep_rate_after: float = 0.50
    min_keep_count_after: int = 48
    budget_score: str = "importance"  # importance | prob
    # Minimum visible candidates kept after applying the order budget.
    min_visible_keep_count: int = 0

    # Compensation
    comp_gain: float = 1.0
    comp_seed: int = 0
    comp_start_ms: float = 40.0
    # "absolute" rectifies the tail; "signed" preserves its sign.
    comp_tail_mode: str = "absolute"

    # Evaluation and output
    timing_repeats: int = 3
    pyroom_timing_repeats: int = 1
    # Memory tracing adds overhead to the Python pruning loop.
    profile_memory: bool = True
    direct_window_ms: float = 2.5
    edc_floor_db: float = -80.0
    rt60_mode: str = "t20"
    csv_name: str = "detailed_room_metrics.csv"
    paper_csv_name: str = "paper_metrics_long.csv"
    paper_summary_csv_name: str = "paper_summary.csv"
    summary_name: str = "summary.json"
    measure_pyroom_runtime: bool = False
    save_rirs: bool = False
    save_rir_wavs: bool = False
    rir_wav_dir: str = "Output_RIR"

    # Self-Full-ISM source: pyroom_extract | stored_nodes | true_expand
    full_ism_mode: str = "pyroom_extract"

    # A positive value replaces the dataset order for every method.
    max_order_override: int = 0

    # Calculate keep rates from visible candidates or the full frontier.
    budget_relative_to: str = "visible"


# WAV output


def sanitize_name(name: str) -> str:
    """Return a filesystem-friendly name while preserving room IDs."""
    keep = []
    for ch in str(name):
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(keep).strip("_") or "unnamed"


def write_rir_wav(path: Path, rir: np.ndarray, fs: int) -> None:
    """Write one mono RIR as WAV while preserving physical amplitude.

    Floating-point WAV is preferred. If neither soundfile nor scipy is
    available, the standard library writes clipped 16-bit PCM without per-file
    normalization.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(rir, dtype=np.float32).reshape(-1)

    try:
        import soundfile as sf  # type: ignore
        sf.write(str(path), x, int(fs), subtype="FLOAT")
        return
    except Exception:
        pass

    try:
        from scipy.io import wavfile  # type: ignore
        wavfile.write(str(path), int(fs), x)
        return
    except Exception:
        pass

    # Preserve relative level across files when falling back to PCM16.
    y = np.clip(x, -1.0, 1.0)
    y16 = (y * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(fs))
        wf.writeframes(y16.tobytes())


def save_method_rirs_as_wavs(base_dir: Path, method: str, room_name: str, rirs: np.ndarray, fs: int, source_name: str = "S1") -> None:
    """Save multi-microphone RIRs as method/room/source/mic.wav."""
    method_dir = base_dir / sanitize_name(method) / sanitize_name(room_name) / sanitize_name(source_name)
    arr = np.asarray(rirs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    for m in range(arr.shape[0]):
        write_rir_wav(method_dir / f"m{m + 1}.wav", arr[m], fs)

# Models and checkpoints


class TinyPruningMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, depth: int = 3, dropout: float = 0.05) -> None:
        super().__init__()
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


class CompensationMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, depth: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and name != "cuda" and ":" not in name:
        suffix = name[len("cuda") :]
        if suffix.isdigit():
            return torch.device(f"cuda:{suffix}")
    return torch.device(name)


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
        len(feature_names),
        int(cfg.get("hidden_dim", 64)),
        int(cfg.get("depth", 3)),
        float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    aux = {
        "feature_names": feature_names,
        "feature_mean": np.asarray(ckpt["feature_mean"], dtype=np.float32).reshape(1, -1),
        "feature_std": np.asarray(ckpt["feature_std"], dtype=np.float32).reshape(1, -1),
        "log_imp_mean": float(ckpt.get("log_imp_mean", 0.0)),
        "log_imp_std": float(ckpt.get("log_imp_std", 1.0)),
    }
    return model, aux


def load_compensation_model(path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, object]]:
    ckpt = torch_load_checkpoint(path, device)
    cfg = ckpt.get("config", {})
    feature_names = list(ckpt["feature_names"])
    target_mean = np.asarray(ckpt["target_mean"], dtype=np.float32).reshape(1, -1)
    target_std = np.asarray(ckpt["target_std"], dtype=np.float32).reshape(1, -1)
    model = CompensationMLP(
        input_dim=len(feature_names),
        output_dim=target_mean.shape[1],
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        depth=int(cfg.get("depth", 4)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    aux = {
        "feature_names": feature_names,
        "feature_mean": np.asarray(ckpt["feature_mean"], dtype=np.float32).reshape(1, -1),
        "feature_std": np.asarray(ckpt["feature_std"], dtype=np.float32).reshape(1, -1),
        "target_mean": target_mean,
        "target_std": target_std,
        "num_bins": int(target_mean.shape[1]),
        "config": cfg,
    }
    return model, aux


# Dataset loading


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def decode_feature_names(raw: np.ndarray) -> List[str]:
    out: List[str] = []
    for x in raw.tolist():
        out.append(x.decode("utf-8") if isinstance(x, bytes) else str(x))
    return out


def parse_metadata(z: np.lib.npyio.NpzFile) -> Dict[str, object]:
    if "metadata_json" not in z:
        return {}
    raw = z["metadata_json"]
    try:
        return json.loads(str(raw.item() if raw.shape == () else raw.tolist()))
    except Exception:
        return {}


def read_manifest_paths(data_dir: Path) -> List[Path]:
    manifest = data_dir / "manifest.jsonl"
    if not manifest.exists():
        return []
    paths: List[Path] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
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
    if max_files > 0:
        files = files[:max_files]
    return files


def load_room_npz(path: Path, ckpt_feature_names: Sequence[str]) -> Dict[str, object]:
    with np.load(path, allow_pickle=True) as z:
        dataset_names = decode_feature_names(z["feature_names"])
        name_to_idx = {name: i for i, name in enumerate(dataset_names)}
        missing = [name for name in ckpt_feature_names if name not in name_to_idx]
        if missing:
            raise ValueError(f"{path.name}: missing pruning features {missing}")
        feature_idx = np.asarray([name_to_idx[name] for name in ckpt_feature_names], dtype=np.int64)
        metadata = parse_metadata(z)
        data: Dict[str, object] = {
            "path": str(path),
            "metadata": metadata,
            "corners_xy": np.asarray(z["corners_xy"], dtype=np.float32),
            "height": float(np.asarray(z["height"]).reshape(-1)[0]),
            "wall_absorption": np.asarray(z["wall_absorption"], dtype=np.float32),
            "source_position": np.asarray(z["source_position"], dtype=np.float32).reshape(3),
            "mic_positions": np.asarray(z["mic_positions"], dtype=np.float32),
            "full_rirs": np.asarray(z["full_rirs"], dtype=np.float32),
            "node_images": np.asarray(z["node_images"], dtype=np.float32),
            "node_damping": np.asarray(z["node_damping"], dtype=np.float32),
            "node_parent": np.asarray(z["node_parent"], dtype=np.int64).reshape(-1),
            "node_order": np.asarray(z["node_order"], dtype=np.int64).reshape(-1),
            "node_wall": np.asarray(z["node_wall"], dtype=np.int64).reshape(-1),
            "node_features": np.asarray(z["node_features"], dtype=np.float32)[:, feature_idx],
        }
        if "node_visibility" in z:
            data["node_visibility"] = np.asarray(z["node_visibility"], dtype=bool)
        else:
            data["node_visibility"] = np.ones((data["mic_positions"].shape[1], data["node_images"].shape[0]), dtype=bool)  # type: ignore[index]
    return data


def check_parent_tree(data: Dict[str, object]) -> Tuple[bool, str, Dict[str, float]]:
    parents = data["node_parent"]  # type: ignore[assignment]
    orders = data["node_order"]  # type: ignore[assignment]
    n = len(parents)
    total_nonroot = int((orders > 0).sum())
    bad = 0
    for i, p in enumerate(parents):
        if orders[i] <= 0:
            continue
        if not (0 <= int(p) < n and orders[int(p)] == orders[i] - 1):
            bad += 1
    stats = {
        "parent_bad_order_nodes": float(bad),
        "parent_valid_nonroot_frac": float((total_nonroot - bad) / max(total_nonroot, 1)),
    }
    return bad == 0, "ok" if bad == 0 else f"bad parent-order links: {bad}", stats


# Tree and RIR synthesis


def build_children(parents: np.ndarray) -> List[List[int]]:
    ch: List[List[int]] = [[] for _ in range(len(parents))]
    for i, p in enumerate(parents):
        if 0 <= int(p) < len(parents):
            ch[int(p)].append(i)
    return ch


def find_roots(parents: np.ndarray) -> List[int]:
    roots = [i for i, p in enumerate(parents) if int(p) < 0 or int(p) >= len(parents)]
    return roots or [0]


def parse_c_and_frac_delay(metadata: Dict[str, object]) -> Tuple[float, int]:
    return float(metadata.get("c", 343.0)), int(metadata.get("frac_delay_len", 81))


def add_fractional_impulse_(buf: np.ndarray, delay_seconds: float, amplitude: float, fs: int, frac_delay_len: int) -> None:
    if not np.isfinite(delay_seconds) or not np.isfinite(amplitude) or amplitude == 0.0:
        return
    d_samp = delay_seconds * fs
    n0 = int(np.floor(d_samp))
    frac = d_samp - n0
    L = int(frac_delay_len)
    t = np.arange(L, dtype=np.float64)
    kernel = np.hanning(L) * np.sinc(t - (L - 1) / 2.0 - frac)
    kernel = kernel.astype(np.float32) * np.float32(amplitude)
    start = n0
    stop = n0 + L
    if stop <= 0 or start >= buf.shape[0]:
        return
    k0 = max(0, -start)
    k1 = L - max(0, stop - buf.shape[0])
    b0 = max(0, start)
    b1 = b0 + (k1 - k0)
    if b1 > b0:
        buf[b0:b1] += kernel[k0:k1]


def damping_scalar(node_damping: np.ndarray, idx: int) -> float:
    return float(node_damping[idx]) if node_damping.ndim == 1 else float(np.mean(node_damping[:, idx]))


def add_node_contribution_(rir: np.ndarray, idx: int, data: Dict[str, object], fs: int, c: float, frac_delay_len: int) -> None:
    images = data["node_images"]  # type: ignore[assignment]
    damping = data["node_damping"]  # type: ignore[assignment]
    mics = data["mic_positions"]  # type: ignore[assignment]
    vis = data["node_visibility"]  # type: ignore[assignment]
    img = images[idx].astype(np.float64)
    damp = damping_scalar(damping, idx)
    for m in range(mics.shape[1]):
        if not bool(vis[m, idx]):
            continue
        dist = float(np.linalg.norm(img - mics[:, m].astype(np.float64)))
        if dist <= 1e-8:
            continue
        add_fractional_impulse_(rir[m], dist / c, damp / dist, fs, frac_delay_len)


def synthesize_full_self(data: Dict[str, object]) -> Tuple[np.ndarray, Dict[str, float]]:
    full_rirs = data["full_rirs"]  # type: ignore[assignment]
    rir_len = full_rirs.shape[1]
    metadata = data["metadata"]  # type: ignore[assignment]
    fs = int(metadata.get("fs", 8000))
    c, frac_delay_len = parse_c_and_frac_delay(metadata)
    n = data["node_images"].shape[0]  # type: ignore[index]
    rir = np.zeros_like(full_rirs, dtype=np.float32)
    for i in range(n):
        add_node_contribution_(rir, i, data, fs, c, frac_delay_len)
    return rir, {"full_expanded_nodes": float(n), "full_contribution_nodes": float(n)}


def build_pyroomacoustics_room_from_data(data: Dict[str, object]) -> object:
    """Recreate a pyroomacoustics room from saved geometry for end-to-end timing."""
    if pra is None:
        raise RuntimeError("pyroomacoustics is required to measure Pyroom runtime")

    metadata = data["metadata"]  # type: ignore[assignment]
    fs = int(metadata.get("fs", 8000))
    max_order = int(metadata.get("max_order", int(np.max(data["node_order"]))))  # type: ignore[index]
    rir_hpf_enable = bool(metadata.get("rir_hpf_enable", False))
    try:
        pra.constants.set("rir_hpf_enable", rir_hpf_enable)
    except Exception:
        pass

    corners = np.asarray(data["corners_xy"], dtype=np.float32)  # type: ignore[arg-type]
    height = float(data["height"])
    wall_abs = np.asarray(data["wall_absorption"], dtype=np.float32)  # type: ignore[arg-type]
    n_side = corners.shape[0]
    side_abs = wall_abs[:n_side]
    floor_abs = float(wall_abs[n_side]) if len(wall_abs) > n_side else float(np.mean(side_abs))
    ceil_abs = float(wall_abs[n_side + 1]) if len(wall_abs) > n_side + 1 else float(np.mean(side_abs))

    side_materials = [pra.Material(energy_absorption=float(a), scattering=0.0) for a in side_abs]
    floor_ceil = {
        "floor": pra.Material(energy_absorption=floor_abs, scattering=0.0),
        "ceiling": pra.Material(energy_absorption=ceil_abs, scattering=0.0),
    }
    try:
        room = pra.Room.from_corners(corners.T, fs=fs, max_order=max_order, materials=side_materials)
        room.extrude(height, materials=floor_ceil)
    except TypeError:
        room = pra.Room.from_corners(corners.T, fs=fs, max_order=max_order, absorption=side_abs)
        room.extrude(height, absorption=wall_abs)

    room.add_source(np.asarray(data["source_position"], dtype=np.float32))
    room.add_microphone_array(np.asarray(data["mic_positions"], dtype=np.float32))
    return room


@torch.no_grad()
def predict_all_nodes(model: nn.Module, node_features: np.ndarray, aux: Dict[str, object], device: torch.device, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    mean = aux["feature_mean"]  # type: ignore[assignment]
    std = aux["feature_std"]  # type: ignore[assignment]
    x = ((node_features.astype(np.float32) - mean) / std).astype(np.float32)
    probs: List[np.ndarray] = []
    imps: List[np.ndarray] = []
    log_imp_mean = float(aux["log_imp_mean"])
    log_imp_std = float(aux["log_imp_std"])
    for s in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(x[s : s + batch_size]).to(device)
        out = model(xb)
        prob = torch.sigmoid(out["keep_logit"]).detach().cpu().numpy().reshape(-1)
        log_imp_norm = out["importance_log_norm"].detach().cpu().numpy().reshape(-1)
        imp = np.power(10.0, log_imp_norm * log_imp_std + log_imp_mean).astype(np.float32)
        probs.append(prob.astype(np.float32))
        imps.append(imp)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return np.concatenate(probs), np.concatenate(imps)


def raw_decision(prob: np.ndarray, imp: np.ndarray, cfg: EvalConfig) -> np.ndarray:
    if cfg.decision_mode == "prob":
        return prob >= cfg.prob_threshold
    if cfg.decision_mode == "importance":
        return imp >= cfg.importance_threshold
    if cfg.decision_mode == "either":
        return (prob >= cfg.prob_threshold) | (imp >= cfg.importance_threshold)
    if cfg.decision_mode == "both":
        return (prob >= cfg.prob_threshold) & (imp >= cfg.importance_threshold)
    raise ValueError(f"Unknown decision_mode={cfg.decision_mode}")


def apply_order_budget(keep_mask: np.ndarray, scores: np.ndarray, order: int, cfg: EvalConfig, n_ref: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply the order budget from Eq. 6.

    ``n_ref`` sets the candidate count used for the keep rates. When omitted,
    the current candidate count is used.
    """
    n = len(keep_mask)
    if n_ref is None:
        n_ref = n
    if n == 0:
        return keep_mask, {"forced_keep": 0.0, "forced_prune": 0.0, "conflict": 0.0}
    if cfg.order_budget_mode == "none":
        return keep_mask, {"forced_keep": 0.0, "forced_prune": 0.0, "conflict": 0.0}
    if cfg.order_budget_mode != "score":
        raise ValueError(f"Unknown order_budget_mode={cfg.order_budget_mode}")
    if order <= cfg.early_keep_order:
        new_keep = np.ones_like(keep_mask, dtype=bool)
        return new_keep, {"forced_keep": float((new_keep & ~keep_mask).sum()), "forced_prune": 0.0, "conflict": 0.0}

    min_count = max(int(cfg.min_keep_count_after), int(math.ceil(cfg.min_keep_rate_after * n_ref)))
    min_count = min(max(min_count, 0), n)
    max_count = n if cfg.max_keep_rate_after >= 1.0 else int(math.floor(cfg.max_keep_rate_after * n_ref))
    max_count = min(max(max_count, 0), n)
    conflict = 0.0
    if max_count < min_count:
        max_count = min_count
        conflict = 1.0

    current = int(keep_mask.sum())
    final_count = min(max(current, min_count), max_count)
    rank = np.argsort(-scores, kind="stable")
    new_keep = np.zeros_like(keep_mask, dtype=bool)
    new_keep[rank[:final_count]] = True
    return new_keep, {
        "forced_keep": float((new_keep & ~keep_mask).sum()),
        "forced_prune": float((~new_keep & keep_mask).sum()),
        "conflict": conflict,
    }


def synthesize_pruned_bfs(data: Dict[str, object], prob: np.ndarray, imp: np.ndarray, cfg: EvalConfig) -> Tuple[np.ndarray, Dict[str, object]]:
    full_rirs = data["full_rirs"]  # type: ignore[assignment]
    rir_len = full_rirs.shape[1]
    metadata = data["metadata"]  # type: ignore[assignment]
    fs = int(metadata.get("fs", 8000))
    c, frac_delay_len = parse_c_and_frac_delay(metadata)
    parents = data["node_parent"]  # type: ignore[assignment]
    orders = data["node_order"]  # type: ignore[assignment]
    n_nodes = data["node_images"].shape[0]  # type: ignore[index]
    children = build_children(parents)
    roots = find_roots(parents)
    rir = np.zeros_like(full_rirs, dtype=np.float32)
    kept = np.zeros(n_nodes, dtype=bool)
    evaluated = np.zeros(n_nodes, dtype=bool)
    raw_kept = np.zeros(n_nodes, dtype=bool)
    by_order: Dict[int, Dict[str, float]] = {}
    peak_frontier = len(roots)

    frontier = list(roots)
    for r in roots:
        kept[r] = True
        evaluated[r] = True
        raw_kept[r] = True
        add_node_contribution_(rir, r, data, fs, c, frac_delay_len)

    while frontier:
        candidates: List[int] = []
        for idx in frontier:
            candidates.extend(children[idx])
        if not candidates:
            break
        cand = np.asarray(candidates, dtype=np.int64)
        cand_orders = orders[cand]
        evaluated[cand] = True
        next_frontier: List[int] = []
        peak_frontier = max(peak_frontier, len(cand))

        for order in sorted(set(int(o) for o in cand_orders.tolist())):
            mask = cand_orders == order
            group = cand[mask]
            raw = raw_decision(prob[group], imp[group], cfg)
            scores = imp[group] if cfg.budget_score == "importance" else prob[group]
            keep_group, budget = apply_order_budget(raw, scores, order, cfg)

            raw_kept[group[raw]] = True
            kept[group[keep_group]] = True
            next_frontier.extend(group[keep_group].astype(np.int64).tolist())

            if order not in by_order:
                by_order[order] = {
                    "evaluated": 0.0,
                    "raw_kept": 0.0,
                    "kept": 0.0,
                    "pruned": 0.0,
                    "budget_forced_keep": 0.0,
                    "budget_forced_prune": 0.0,
                    "budget_conflict": 0.0,
                }
            by_order[order]["evaluated"] += float(len(group))
            by_order[order]["raw_kept"] += float(raw.sum())
            by_order[order]["kept"] += float(keep_group.sum())
            by_order[order]["pruned"] += float((~keep_group).sum())
            by_order[order]["budget_forced_keep"] += budget["forced_keep"]
            by_order[order]["budget_forced_prune"] += budget["forced_prune"]
            by_order[order]["budget_conflict"] += budget["conflict"]

        for idx in next_frontier:
            add_node_contribution_(rir, int(idx), data, fs, c, frac_delay_len)
        peak_frontier = max(peak_frontier, len(next_frontier))
        frontier = next_frontier

    stats: Dict[str, object] = {
        "kept_mask": kept,
        "evaluated_mask": evaluated,
        "raw_kept_mask": raw_kept,
        "missing_mask": ~kept,
        "pruned_expanded_nodes": float(evaluated.sum()),
        "pruned_model_evaluated_nodes": float(max(evaluated.sum() - len(roots), 0)),
        "pruned_contribution_nodes": float(kept.sum()),
        "pruned_decision_nodes": float(evaluated.sum() - kept.sum()),
        "skipped_by_ancestor_nodes": float(n_nodes - evaluated.sum()),
        "expanded_node_reduction": float(1.0 - evaluated.sum() / max(n_nodes, 1)),
        "contribution_node_reduction": float(1.0 - kept.sum() / max(n_nodes, 1)),
        "pruned_peak_frontier": float(peak_frontier),
        "budget_forced_keep_nodes": float(sum(v["budget_forced_keep"] for v in by_order.values())),
        "budget_forced_prune_nodes": float(sum(v["budget_forced_prune"] for v in by_order.values())),
        "budget_conflict_orders": float(sum(1.0 for v in by_order.values() if v["budget_conflict"] > 0)),
        "by_order": by_order,
    }
    return rir, stats


# Compensation features and stochastic tail


def polygon_area(poly_xy: np.ndarray) -> float:
    x = poly_xy[:, 0]
    y = poly_xy[:, 1]
    return abs(0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def log10_safe(x, eps: float = 1e-20):
    return np.log10(np.maximum(x, 0.0) + eps)


def energy_bins_1d(x: np.ndarray, num_bins: int) -> np.ndarray:
    n = len(x)
    edges = np.linspace(0, n, num_bins + 1, dtype=np.int64)
    e = np.zeros(num_bins, dtype=np.float32)
    xx = x.astype(np.float64) ** 2
    for b in range(num_bins):
        s, t = int(edges[b]), int(edges[b + 1])
        if t > s:
            e[b] = float(np.sum(xx[s:t]))
    return e


def node_delay_bins_for_mic(data: Dict[str, object], mic_index: int, num_bins: int, rir_len: int) -> np.ndarray:
    metadata = data["metadata"]  # type: ignore[assignment]
    fs = int(metadata.get("fs", 8000))
    c, frac_delay_len = parse_c_and_frac_delay(metadata)
    images = data["node_images"]  # type: ignore[assignment]
    mics = data["mic_positions"]  # type: ignore[assignment]
    mic = mics[:, mic_index].astype(np.float64)
    d = np.linalg.norm(images.astype(np.float64) - mic[None, :], axis=1)
    samples = np.round(d / c * fs + frac_delay_len // 2).astype(np.int64)
    bins = np.floor(samples / max(rir_len, 1) * num_bins).astype(np.int64)
    return np.clip(bins, 0, num_bins - 1)


def direct_delay_s(data: Dict[str, object], mic_index: int) -> float:
    metadata = data["metadata"]  # type: ignore[assignment]
    c = float(metadata.get("c", 343.0))
    src = data["source_position"]  # type: ignore[assignment]
    mics = data["mic_positions"]  # type: ignore[assignment]
    return float(np.linalg.norm(src.astype(np.float64) - mics[:, mic_index].astype(np.float64)) / c)


def build_compensation_features(data: Dict[str, object], pruned_rir: np.ndarray, pruned_stats: Dict[str, object], all_imp: np.ndarray, comp_aux: Dict[str, object]) -> np.ndarray:
    num_bins = int(comp_aux["num_bins"])
    comp_cfg = comp_aux.get("config", {})
    metadata = data["metadata"]  # type: ignore[assignment]
    fs = int(metadata.get("fs", 8000))
    max_order = int(metadata.get("max_order", int(np.max(data["node_order"]))))  # type: ignore[index]
    order_feature_max = int(comp_cfg.get("order_feature_max", max_order)) if isinstance(comp_cfg, dict) else max_order
    rir_len = pruned_rir.shape[1]
    duration = rir_len / float(fs)

    corners = data["corners_xy"]  # type: ignore[assignment]
    height = float(data["height"])
    wall_abs = data["wall_absorption"]  # type: ignore[assignment]
    volume = polygon_area(corners) * height
    source = data["source_position"]  # type: ignore[assignment]
    mics = data["mic_positions"]  # type: ignore[assignment]
    orders = data["node_order"]  # type: ignore[assignment]
    visibility = data["node_visibility"]  # type: ignore[assignment]
    n_nodes = len(orders)
    kept = np.asarray(pruned_stats["kept_mask"], dtype=bool)
    evaluated = np.asarray(pruned_stats["evaluated_mask"], dtype=bool)
    missing = ~kept
    imp = np.nan_to_num(np.asarray(all_imp, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    order_feats: List[float] = []
    order_names: List[str] = []
    for o in range(order_feature_max + 1):
        om = orders == o
        denom = max(int(om.sum()), 1)
        vals = {
            "full_frac": float(om.sum() / max(n_nodes, 1)),
            "evaluated_rate": float((evaluated & om).sum() / denom),
            "kept_rate": float((kept & om).sum() / denom),
            "missing_rate": float((missing & om).sum() / denom),
            "log_missing_imp": float(log10_safe(np.sum(imp[missing & om]), 1e-20)),
            "log_kept_imp": float(log10_safe(np.sum(imp[kept & om]), 1e-20)),
        }
        for key, val in vals.items():
            order_feats.append(val)
            order_names.append(f"order{o}_{key}")

    base_feats = [
        float(fs / 16000.0),
        float(duration),
        float(max_order),
        float(n_nodes),
        float(log10_safe(n_nodes, 1.0)),
        float(corners.shape[0]),
        float(polygon_area(corners)),
        float(height),
        float(volume),
        float(log10_safe(volume, 1e-6)),
        float(np.mean(wall_abs)),
        float(np.std(wall_abs)),
        float(np.min(wall_abs)),
        float(np.max(wall_abs)),
        float(pruned_stats["expanded_node_reduction"]),
        float(pruned_stats["contribution_node_reduction"]),
        float(kept.sum() / max(n_nodes, 1)),
        float(evaluated.sum() / max(n_nodes, 1)),
        float(missing.sum() / max(n_nodes, 1)),
        float(log10_safe(np.sum(imp[missing]), 1e-20)),
        float(log10_safe(np.sum(imp[kept]), 1e-20)),
        float(log10_safe(np.sum(imp), 1e-20)),
    ]
    base_names = [
        "fs_over_16000", "rir_duration_s", "max_order", "num_nodes", "log10_num_nodes",
        "num_side_walls", "floor_area", "height", "volume", "log10_volume",
        "wall_abs_mean", "wall_abs_std", "wall_abs_min", "wall_abs_max",
        "expanded_node_reduction", "contribution_node_reduction", "kept_node_frac",
        "evaluated_node_frac", "missing_node_frac", "log10_missing_pred_importance",
        "log10_kept_pred_importance", "log10_total_pred_importance",
    ]

    rows: List[np.ndarray] = []
    generated_names: Optional[List[str]] = None
    for mic in range(pruned_rir.shape[0]):
        bin_idx = node_delay_bins_for_mic(data, mic, num_bins, rir_len)
        missing_imp_bins = np.zeros(num_bins, dtype=np.float32)
        kept_imp_bins = np.zeros(num_bins, dtype=np.float32)
        for i in range(n_nodes):
            if not bool(visibility[mic, i]):
                continue
            b = int(bin_idx[i])
            if missing[i]:
                missing_imp_bins[b] += float(imp[i])
            elif kept[i]:
                kept_imp_bins[b] += float(imp[i])
        missing_imp_log = np.asarray(log10_safe(missing_imp_bins, 1e-20), dtype=np.float32)
        kept_imp_log = np.asarray(log10_safe(kept_imp_bins, 1e-20), dtype=np.float32)
        pruned_log_bins = np.asarray(log10_safe(energy_bins_1d(pruned_rir[mic], num_bins), 1e-20), dtype=np.float32)

        mic_pos = mics[:, mic]
        mic_feats = [
            float(mic),
            float(np.linalg.norm(source.astype(np.float64) - mic_pos.astype(np.float64))),
            direct_delay_s(data, mic),
            float(log10_safe(np.sum(pruned_rir[mic].astype(np.float64) ** 2), 1e-20)),
            0.0,
        ]
        mic_names = ["mic_index", "src_mic_distance", "direct_delay_s", "log10_pruned_rir_energy", "log10_residual_energy_oracle_train_only"]
        row = np.asarray(base_feats + mic_feats + order_feats + missing_imp_log.tolist() + kept_imp_log.tolist() + pruned_log_bins.tolist(), dtype=np.float32)
        rows.append(row)
        if generated_names is None:
            generated_names = base_names + mic_names + order_names + [f"log10_missing_pred_imp_bin{b}" for b in range(num_bins)] + [f"log10_kept_pred_imp_bin{b}" for b in range(num_bins)] + [f"log10_pruned_energy_bin{b}" for b in range(num_bins)]

    assert generated_names is not None
    if comp_aux.get("feature_names") is None:
        # Training needs every generated feature and its column name.
        comp_aux["feature_names"] = list(generated_names)
        return np.stack(rows, axis=0).astype(np.float32)
    ckpt_names = list(comp_aux["feature_names"])
    name_to_idx = {name: i for i, name in enumerate(generated_names)}
    missing_names = [name for name in ckpt_names if name not in name_to_idx]
    if missing_names:
        raise ValueError(f"Compensation checkpoint expects features not generated here: {missing_names[:10]}")
    idx = np.asarray([name_to_idx[name] for name in ckpt_names], dtype=np.int64)
    return np.stack(rows, axis=0)[:, idx].astype(np.float32)


@torch.no_grad()
def predict_compensation_energy_bins(model: nn.Module, aux: Dict[str, object], x: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    mean = aux["feature_mean"]  # type: ignore[assignment]
    std = aux["feature_std"]  # type: ignore[assignment]
    target_mean = aux["target_mean"]  # type: ignore[assignment]
    target_std = aux["target_std"]  # type: ignore[assignment]
    x_norm = ((x.astype(np.float32) - mean) / std).astype(np.float32)
    pred_norm = model(torch.from_numpy(x_norm).to(device)).detach().cpu().numpy()
    pred_log = pred_norm * target_std + target_mean
    pred_energy = np.maximum(np.power(10.0, pred_log).astype(np.float32), 0.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return pred_energy, pred_log.astype(np.float32)


def generate_stochastic_tail(energy_bins: np.ndarray, fs: int, rir_len: int, gain: float, seed: int, start_ms: float, mode: str = "absolute") -> np.ndarray:
    """Generate the stochastic compensation tail.

    Zero-mean Gaussian noise is scaled within each time bin so that its energy
    matches the predicted residual-energy envelope. ``mode="absolute"``
    rectifies the scaled noise, while ``mode="signed"`` preserves its sign.
    """
    mode = str(mode).strip().lower()
    if mode not in {"absolute", "signed"}:
        raise ValueError(f"Unsupported compensation tail mode: {mode!r}")
    rng = np.random.default_rng(seed)
    num_mics, num_bins = energy_bins.shape
    edges = np.linspace(0, rir_len, num_bins + 1, dtype=np.int64)
    start_sample = int(round(max(0.0, start_ms) / 1000.0 * fs))
    tail = np.zeros((num_mics, rir_len), dtype=np.float32)
    for m in range(num_mics):
        for b in range(num_bins):
            s = max(int(edges[b]), start_sample)
            t = int(edges[b + 1])
            if t <= s:
                continue
            e = float(max(energy_bins[m, b], 0.0))
            if e <= 0.0:
                continue
            noise = rng.standard_normal(t - s).astype(np.float64)
            noise -= float(np.mean(noise))
            norm = float(np.sqrt(np.sum(noise ** 2)))
            if norm <= 1e-20:
                continue
            tail[m, s:t] += (gain * math.sqrt(e) * noise / norm).astype(np.float32)
    if mode == "absolute":
        np.abs(tail, out=tail)
    return tail

# ISM geometry and method runners

FEATURE_NAMES_ALL = [
    "order", "wall_id", "parent_order", "node_wall_absorption",
    "image_x", "image_y", "image_z", "damping_mean", "damping_min", "damping_max",
    "dist_min", "dist_mean", "dist_max", "delay_min", "delay_mean", "delay_max",
    "visible_count", "visible_frac", "room_num_walls", "room_floor_area", "room_height", "room_volume",
    "wall_abs_mean", "wall_abs_std", "src_x", "src_y", "src_z",
    "mic_centroid_x", "mic_centroid_y", "mic_centroid_z",
]


def load_geometry_from_npz(path: Path, include_saved_ref: bool = False, max_order_override: int = 0) -> Dict[str, object]:
    """Load scene geometry and simulation settings without saved tree nodes.

    When requested, the stored full RIR is included as the pyroomacoustics
    reference. That reference uses the dataset's original reflection order, so
    it should not be used with ``max_order_override``.
    """
    with np.load(path, allow_pickle=True) as z:
        metadata = parse_metadata(z)
        if max_order_override and int(max_order_override) > 0:
            metadata["max_order"] = int(max_order_override)
        max_order = int(metadata.get("max_order", 10))
        fs = int(metadata.get("fs", 8000))
        rir_duration = float(metadata.get("rir_duration", 0.5))
        rir_len = int(round(fs * rir_duration))
        mics = np.asarray(z["mic_positions"], dtype=np.float32)
        scene: Dict[str, object] = {
            "path": str(path),
            "metadata": metadata,
            "corners_xy": np.asarray(z["corners_xy"], dtype=np.float32),
            "height": float(np.asarray(z["height"]).reshape(-1)[0]),
            "wall_absorption": np.asarray(z["wall_absorption"], dtype=np.float32),
            "source_position": np.asarray(z["source_position"], dtype=np.float32).reshape(3),
            "mic_positions": mics,
            "node_order": np.asarray([max_order], dtype=np.int64),
            "full_rirs": np.zeros((mics.shape[1], rir_len), dtype=np.float32),
        }
        if include_saved_ref and "full_rirs" in z:
            scene["full_rirs"] = np.asarray(z["full_rirs"], dtype=np.float32)
    return scene


def make_node_features_online(nodes: Dict[str, np.ndarray], meta: Dict[str, object], visibility: np.ndarray, c: float) -> np.ndarray:
    images = nodes["node_images"]
    damping = nodes["node_damping"]
    parents = nodes["node_parent"].astype(np.int64)
    walls = nodes["node_wall"].astype(np.int64)
    orders = nodes["node_order"].astype(np.float32)
    num_nodes = images.shape[0]
    mic_positions = np.asarray(meta["mic_positions"], dtype=np.float32)
    source_position = np.asarray(meta["source_position"], dtype=np.float32)
    wall_abs = np.asarray(meta["wall_absorption"], dtype=np.float32)
    corners = np.asarray(meta["corners_xy"], dtype=np.float32)
    height = float(meta["height"])
    floor_area = polygon_area(corners)
    volume = floor_area * height

    d = np.linalg.norm(images[:, None, :] - mic_positions.T[None, :, :], axis=2)
    delays = d / float(c)
    if damping.ndim == 1:
        damping_mean = damping.astype(np.float32)
        damping_min = damping.astype(np.float32)
        damping_max = damping.astype(np.float32)
    else:
        damping_mean = damping.mean(axis=0)
        damping_min = damping.min(axis=0)
        damping_max = damping.max(axis=0)

    parent_order = np.zeros(num_nodes, dtype=np.float32)
    valid_parent = (parents >= 0) & (parents < num_nodes)
    parent_order[valid_parent] = orders[parents[valid_parent].astype(np.int64)]
    parent_order[~valid_parent] = -1.0

    node_wall_abs = np.zeros(num_nodes, dtype=np.float32)
    valid_wall = (walls >= 0) & (walls < len(wall_abs))
    node_wall_abs[valid_wall] = wall_abs[walls[valid_wall]]
    visible_count = visibility.sum(axis=0).astype(np.float32)
    visible_frac = visible_count / max(1, visibility.shape[0])
    mic_centroid = mic_positions.mean(axis=1)

    feats = np.column_stack([
        orders,
        walls.astype(np.float32),
        parent_order,
        node_wall_abs,
        images[:, 0], images[:, 1], images[:, 2],
        damping_mean, damping_min, damping_max,
        d.min(axis=1), d.mean(axis=1), d.max(axis=1),
        delays.min(axis=1), delays.mean(axis=1), delays.max(axis=1),
        visible_count, visible_frac,
        np.full(num_nodes, float(len(wall_abs)), dtype=np.float32),
        np.full(num_nodes, float(floor_area), dtype=np.float32),
        np.full(num_nodes, float(height), dtype=np.float32),
        np.full(num_nodes, float(volume), dtype=np.float32),
        np.full(num_nodes, float(wall_abs.mean()), dtype=np.float32),
        np.full(num_nodes, float(wall_abs.std()), dtype=np.float32),
        np.full(num_nodes, float(source_position[0]), dtype=np.float32),
        np.full(num_nodes, float(source_position[1]), dtype=np.float32),
        np.full(num_nodes, float(source_position[2]), dtype=np.float32),
        np.full(num_nodes, float(mic_centroid[0]), dtype=np.float32),
        np.full(num_nodes, float(mic_centroid[1]), dtype=np.float32),
        np.full(num_nodes, float(mic_centroid[2]), dtype=np.float32),
    ])
    return feats.astype(np.float32)


def select_checkpoint_features(features_all: np.ndarray, ckpt_feature_names: Sequence[str]) -> np.ndarray:
    name_to_idx = {name: i for i, name in enumerate(FEATURE_NAMES_ALL)}
    missing = [name for name in ckpt_feature_names if name not in name_to_idx]
    if missing:
        raise ValueError(f"Online feature generator missing checkpoint features: {missing}")
    idx = np.asarray([name_to_idx[name] for name in ckpt_feature_names], dtype=np.int64)
    return features_all[:, idx].astype(np.float32)

def make_node_dict(tree: ISMTree, visibility: np.ndarray, scene: Dict[str, object], ckpt_feature_names: Sequence[str], build_features: bool = True) -> Dict[str, object]:
    """Convert an ``ISMTree`` into the arrays and features used by the models."""
    node_arrays = {
        "node_images": tree.img_pos.astype(np.float32),
        "node_damping": tree.damping.astype(np.float32)[None, :],
        "node_parent": tree.parent_idx.astype(np.int32),
        "node_wall": tree.gen_wall.astype(np.int32),
        "node_order": tree.order.astype(np.int16),
    }
    data = dict(scene)
    data.update(node_arrays)
    data["node_visibility"] = visibility.astype(bool)
    if build_features:
        metadata = scene["metadata"]  # type: ignore[assignment]
        c, _ = parse_c_and_frac_delay(metadata)
        features_all = make_node_features_online(node_arrays, scene, visibility, c)
        features = select_checkpoint_features(features_all, ckpt_feature_names)
        data["node_features"] = features.astype(np.float32)
    return data


def accumulate_node_rir_(rir: np.ndarray, tree: ISMTree, row: int, vis_col: np.ndarray, mics: np.ndarray, fs: int, c: float, frac_delay_len: int) -> None:
    """Add one image source to the RIR of each microphone that can see it."""
    img = tree.img_pos[row]
    damping = float(tree.damping[row])
    for m in range(mics.shape[1]):
        if not bool(vis_col[m]):
            continue
        dist = float(np.linalg.norm(img - mics[:, m].astype(np.float64)))
        if dist <= 1e-8:
            continue
        add_fractional_impulse_(rir[m], dist / c, damping / dist, fs, frac_delay_len)


def synthesize_nodes_rir(tree: ISMTree, visibility: np.ndarray, scene: Dict[str, object]) -> np.ndarray:
    metadata = scene["metadata"]  # type: ignore[assignment]
    fs = int(metadata.get("fs", 8000))
    rir_len = int(round(float(metadata.get("rir_duration", 0.5)) * fs))
    c, frac_delay_len = parse_c_and_frac_delay(metadata)
    mics = np.asarray(scene["mic_positions"], dtype=np.float32)
    rir = np.zeros((mics.shape[1], rir_len), dtype=np.float32)
    for j in range(len(tree)):
        accumulate_node_rir_(rir, tree, j, visibility[:, j], mics, fs, c, frac_delay_len)
    return rir


def run_full_online_ism_true_expand(scene: Dict[str, object], ckpt_feature_names: Sequence[str], cfg: EvalConfig) -> Tuple[np.ndarray, Dict[str, float], Dict[str, object]]:
    """Run the polygon ISM engine without pruning.

    The tree grows one reflection order at a time. Every generated node remains
    available for expansion, while visibility only controls whether it
    contributes to the RIR.

    The tree grows exponentially, so this mode is intended for reflection
    orders of about 6 or below. Use ``pyroom_extract`` for higher orders.
    """
    corners = np.asarray(scene["corners_xy"], dtype=np.float64)
    height = float(scene["height"])
    wall_abs = np.asarray(scene["wall_absorption"], dtype=np.float64)
    source = np.asarray(scene["source_position"], dtype=np.float64)
    mics = np.asarray(scene["mic_positions"], dtype=np.float32)
    metadata = scene["metadata"]  # type: ignore[assignment]
    max_order = int(metadata.get("max_order", 10))

    room = ExtrudedPolygonRoom(corners, height, wall_abs)
    tree = ISMTree.from_source(source)
    vis_blocks: List[np.ndarray] = [tree.visibility_matrix(room, mics, indices=[0])]
    frontier = np.array([0], dtype=np.int64)
    peak_frontier = 1

    for _order in range(1, max_order + 1):
        new_rows = tree.expand_frontier(room, frontier)
        if new_rows.size == 0:
            break
        vis_blocks.append(tree.visibility_matrix(room, mics, indices=new_rows))
        frontier = new_rows
        peak_frontier = max(peak_frontier, int(frontier.size))

    visibility = np.concatenate(vis_blocks, axis=1).astype(bool)
    data = make_node_dict(tree, visibility, scene, ckpt_feature_names, build_features=False)
    rir = synthesize_nodes_rir(tree, visibility, scene)
    stats = {
        "full_expanded_nodes": float(len(tree)),
        "full_contribution_nodes": float(np.sum(visibility.any(axis=0))),
        "full_peak_frontier": float(peak_frontier),
        "parent_valid_nonroot_frac": 1.0,
        "parent_bad_order_nodes": 0.0,
    }
    return rir, stats, data


def run_full_online_ism_pyroom_extract(scene: Dict[str, object], ckpt_feature_names: Sequence[str], cfg: EvalConfig) -> Tuple[np.ndarray, Dict[str, float], Dict[str, object]]:
    """Build the full image-source set with pyroomacoustics.

    pyroomacoustics supplies the positions, damping, and visibility, and the
    PathRIR backend accumulates their RIR contributions.
    """
    if pra is None:
        raise RuntimeError("pyroomacoustics is required for full_ism_mode='pyroom_extract'")
    corners = np.asarray(scene["corners_xy"], dtype=np.float64)
    height = float(scene["height"])
    wall_abs = np.asarray(scene["wall_absorption"], dtype=np.float64)
    source = np.asarray(scene["source_position"], dtype=np.float64)
    mics = np.asarray(scene["mic_positions"], dtype=np.float64)
    metadata = scene["metadata"]  # type: ignore[assignment]
    fs = int(metadata.get("fs", 8000))
    rir_len = int(round(float(metadata.get("rir_duration", 0.5)) * fs))
    c, frac_delay_len = parse_c_and_frac_delay(metadata)
    max_order = int(metadata.get("max_order", 10))
    n_side = corners.shape[0]

    room = pra.Room.from_corners(
        corners.T, fs=fs, max_order=max_order,
        materials=[pra.Material(energy_absorption=float(a), scattering=0.0) for a in wall_abs[:n_side]],
        ray_tracing=False, air_absorption=False,
    )
    room.extrude(height, materials={
        "floor": pra.Material(energy_absorption=float(wall_abs[n_side]), scattering=0.0),
        "ceiling": pra.Material(energy_absorption=float(wall_abs[n_side + 1]), scattering=0.0),
    })
    room.add_source(source)
    for m in range(mics.shape[1]):
        room.add_microphone(mics[:, m])
    room.image_source_model()

    src0 = room.sources[0]
    images = np.asarray(src0.images, dtype=np.float64).T
    damping = np.asarray(src0.damping, dtype=np.float64)
    if damping.ndim == 2:
        damping = damping[0]
    orders = np.asarray(src0.orders, dtype=np.int64).reshape(-1)
    n_nodes = images.shape[0]
    visibility = np.ones((mics.shape[1], n_nodes), dtype=bool)
    raw_vis = getattr(room, "visibility", None)
    if raw_vis is not None:
        for m in range(mics.shape[1]):
            v = np.asarray(raw_vis[0][m]).astype(bool).reshape(-1)
            if v.shape[0] == n_nodes:
                visibility[m] = v

    rir = np.zeros((mics.shape[1], rir_len), dtype=np.float32)
    for i in range(n_nodes):
        for m in range(mics.shape[1]):
            if not bool(visibility[m, i]):
                continue
            dist = float(np.linalg.norm(images[i] - mics[:, m]))
            if dist > 1e-8:
                add_fractional_impulse_(rir[m], dist / c, float(damping[i]) / dist, fs, frac_delay_len)

    node_arrays = {
        "node_images": images.astype(np.float32),
        "node_damping": damping.astype(np.float32)[None, :],
        "node_parent": np.full(n_nodes, -1, dtype=np.int32),
        "node_wall": np.asarray(src0.walls, dtype=np.int32).reshape(-1),
        "node_order": orders.astype(np.int16),
    }
    data = dict(scene)
    data.update(node_arrays)
    data["node_visibility"] = visibility
    stats = {
        "full_expanded_nodes": float(n_nodes),
        "full_contribution_nodes": float(np.sum(visibility.any(axis=0))),
        "full_peak_frontier": float(np.max(np.bincount(orders.astype(int)))) if n_nodes else 0.0,
        "parent_valid_nonroot_frac": 1.0,
        "parent_bad_order_nodes": 0.0,
    }
    return rir, stats, data


def run_full_online_ism(scene: Dict[str, object], ckpt_feature_names: Sequence[str], cfg: EvalConfig) -> Tuple[np.ndarray, Dict[str, float], Dict[str, object]]:
    if cfg.full_ism_mode == "true_expand":
        return run_full_online_ism_true_expand(scene, ckpt_feature_names, cfg)
    return run_full_online_ism_pyroom_extract(scene, ckpt_feature_names, cfg)


def load_stored_full_nodes(path: Path, cfg: EvalConfig) -> Dict[str, object]:
    """Load the visible image sources used by the ``stored_nodes`` baseline.

    Dataset files also contain invisible connector nodes and negative samples.
    Those rows are removed here. Loading happens before the timed accumulation.
    """
    scene = load_geometry_from_npz(path, include_saved_ref=False, max_order_override=cfg.max_order_override)
    with np.load(path, allow_pickle=True) as z:
        if "node_images" not in z:
            raise RuntimeError(
                f"{path.name} does not contain stored image-source nodes. "
                "Generate the dataset again or use --full-ism-mode pyroom_extract."
            )
        images = np.asarray(z["node_images"], dtype=np.float64)
        damping = np.asarray(z["node_damping"], dtype=np.float64)
        damping = damping[0] if damping.ndim == 2 else damping
        orders = np.asarray(z["node_order"], dtype=np.int64).reshape(-1)
        vis = np.asarray(z["node_visibility"]).astype(bool)
    keep = vis.any(axis=0)
    if cfg.max_order_override and int(cfg.max_order_override) > 0:
        keep &= orders <= int(cfg.max_order_override)
    idx = np.flatnonzero(keep)
    return {
        "scene": scene,
        "images": images[idx],
        "damping": damping[idx],
        "orders": orders[idx],
        "visibility": vis[:, idx],
    }


def run_full_ism_from_stored(stored: Dict[str, object]) -> Tuple[np.ndarray, Dict[str, float], Dict[str, object]]:
    """Accumulate the full-ISM RIR from a preloaded set of visible nodes."""
    scene = stored["scene"]  # type: ignore[assignment]
    metadata = scene["metadata"]  # type: ignore[index]
    fs = int(metadata.get("fs", 8000))
    rir_len = int(round(float(metadata.get("rir_duration", 0.5)) * fs))
    c, frac_delay_len = parse_c_and_frac_delay(metadata)
    mics = np.asarray(scene["mic_positions"], dtype=np.float64)  # type: ignore[index]
    images = np.asarray(stored["images"], dtype=np.float64)
    damping = np.asarray(stored["damping"], dtype=np.float64)
    visibility = np.asarray(stored["visibility"], dtype=bool)
    n_nodes = images.shape[0]

    rir = np.zeros((mics.shape[1], rir_len), dtype=np.float32)
    for i in range(n_nodes):
        for m in range(mics.shape[1]):
            if not bool(visibility[m, i]):
                continue
            dist = float(np.linalg.norm(images[i] - mics[:, m]))
            if dist > 1e-8:
                add_fractional_impulse_(rir[m], dist / c, float(damping[i]) / dist, fs, frac_delay_len)

    data = dict(scene)  # type: ignore[arg-type]
    data.update({
        "node_images": images.astype(np.float32),
        "node_damping": damping.astype(np.float32)[None, :],
        "node_parent": np.full(n_nodes, -1, dtype=np.int32),
        "node_wall": np.full(n_nodes, -1, dtype=np.int32),
        "node_order": np.asarray(stored["orders"], dtype=np.int16),
        "node_visibility": visibility,
    })
    stats = {
        "full_expanded_nodes": float(n_nodes),
        "full_contribution_nodes": float(n_nodes),
        "full_peak_frontier": float(np.max(np.bincount(np.asarray(stored["orders"], dtype=np.int64)))) if n_nodes else 0.0,
        "parent_valid_nonroot_frac": 1.0,
        "parent_bad_order_nodes": 0.0,
    }
    return rir, stats, data


def raw_decision_for_cfg(prob: np.ndarray, imp: np.ndarray, cfg: EvalConfig) -> np.ndarray:
    return raw_decision(prob, imp, cfg)


def run_pruned_online_ism(scene: Dict[str, object], pruning_model: nn.Module, pruning_aux: Dict[str, object], device: torch.device, cfg: EvalConfig) -> Tuple[np.ndarray, Dict[str, object], Dict[str, object], np.ndarray, np.ndarray]:
    corners = np.asarray(scene["corners_xy"], dtype=np.float64)
    height = float(scene["height"])
    wall_abs = np.asarray(scene["wall_absorption"], dtype=np.float64)
    source = np.asarray(scene["source_position"], dtype=np.float64)
    mics = np.asarray(scene["mic_positions"], dtype=np.float32)
    metadata = scene["metadata"]  # type: ignore[assignment]
    fs = int(metadata.get("fs", 8000))
    rir_len = int(round(float(metadata.get("rir_duration", 0.5)) * fs))
    c, frac_delay_len = parse_c_and_frac_delay(metadata)
    max_order = int(metadata.get("max_order", 10))
    ckpt_feature_names = list(pruning_aux["feature_names"])

    room = ExtrudedPolygonRoom(corners, height, wall_abs)
    tree = ISMTree.from_source(source)
    root_vis = tree.visibility_matrix(room, mics, indices=[0])
    vis_blocks: List[np.ndarray] = [root_vis]
    kept_flags: List[bool] = [True]
    evaluated_flags: List[bool] = [True]
    raw_kept_flags: List[bool] = [True]
    prob_values: List[float] = [1.0]
    imp_values: List[float] = [1.0]
    frontier = np.array([0], dtype=np.int64)
    rir = np.zeros((mics.shape[1], rir_len), dtype=np.float32)
    # Add the direct path when it is visible.
    accumulate_node_rir_(rir, tree, 0, root_vis[:, 0], mics, fs, c, frac_delay_len)

    by_order: Dict[int, Dict[str, float]] = {}
    peak_frontier = 1

    for order in range(1, max_order + 1):
        # Expand only the nodes kept at the previous order. Invisible
        # candidates are still considered because their descendants may be
        # visible.
        n_before = len(tree)
        cand_rows = tree.expand_frontier(room, frontier)
        if cand_rows.size == 0:
            break
        cand_vis = tree.visibility_matrix(room, mics, indices=cand_rows)  # Shape: (microphones, candidates).
        vis_blocks.append(cand_vis)

        # Build the features expected by the pruning checkpoint.
        temp_vis = np.concatenate(vis_blocks, axis=1).astype(bool)
        temp_data = make_node_dict(tree, temp_vis, scene, ckpt_feature_names)
        cand_features = np.asarray(temp_data["node_features"], dtype=np.float32)[n_before:]
        prob, imp = predict_all_nodes(pruning_model, cand_features, pruning_aux, device, cfg.batch_size)
        raw = raw_decision_for_cfg(prob, imp, cfg)
        scores = imp if cfg.budget_score == "importance" else prob
        n_ref = int(cand_vis.any(axis=0).sum()) if cfg.budget_relative_to == "visible" else None
        keep, budget = apply_order_budget(raw, scores, order, cfg, n_ref=n_ref)

        # Keep the highest-scoring visible candidates when the order budget
        # falls below the requested minimum.
        forced_visible = 0
        if cfg.min_visible_keep_count > 0 and order > cfg.early_keep_order:
            keep = keep.copy()
            vis_any = cand_vis.any(axis=0)
            n_target = min(int(cfg.min_visible_keep_count), int(vis_any.sum()))
            n_vis_kept = int(np.sum(keep & vis_any))
            if n_vis_kept < n_target:
                addable = np.flatnonzero(vis_any & ~keep)
                addable = addable[np.argsort(-scores[addable], kind="stable")]
                forced_visible = n_target - n_vis_kept
                keep[addable[:forced_visible]] = True

        for j, row in enumerate(cand_rows):
            evaluated_flags.append(True)
            raw_kept_flags.append(bool(raw[j]))
            kept_flags.append(bool(keep[j]))
            prob_values.append(float(prob[j]))
            imp_values.append(float(imp[j]))
            if bool(keep[j]):
                accumulate_node_rir_(rir, tree, int(row), cand_vis[:, j], mics, fs, c, frac_delay_len)

        by_order[order] = {
            "evaluated": float(cand_rows.size),
            "raw_kept": float(np.sum(raw)),
            "kept": float(np.sum(keep)),
            "kept_visible": float(np.sum(keep & cand_vis.any(axis=0))),
            "pruned": float(np.sum(~keep)),
            "budget_forced_keep": float(budget["forced_keep"]),
            "budget_forced_prune": float(budget["forced_prune"]),
            "budget_forced_visible_keep": float(forced_visible),
            "budget_conflict": float(budget["conflict"]),
        }
        frontier = cand_rows[keep]
        peak_frontier = max(peak_frontier, int(frontier.size))
        if frontier.size == 0:
            break

    visibility = np.concatenate(vis_blocks, axis=1).astype(bool)
    data = make_node_dict(tree, visibility, scene, ckpt_feature_names)
    kept = np.asarray(kept_flags, dtype=bool)
    evaluated = np.asarray(evaluated_flags, dtype=bool)
    raw_kept = np.asarray(raw_kept_flags, dtype=bool)
    prob_all = np.asarray(prob_values, dtype=np.float32)
    imp_all = np.asarray(imp_values, dtype=np.float32)
    # An invisible node may be kept so its descendants can still be expanded.
    # Only visible retained nodes contribute to the RIR and to R_img.
    visible_retained = kept & visibility.any(axis=0)
    stats: Dict[str, object] = {
        "kept_mask": kept,
        "evaluated_mask": evaluated,
        "raw_kept_mask": raw_kept,
        "missing_mask": ~kept,
        "pruned_expanded_nodes": float(evaluated.sum()),
        "pruned_model_evaluated_nodes": float(max(evaluated.sum() - 1, 0)),
        # Retained traversal nodes, including invisible connectors.
        "pruned_retained_nodes": float(kept.sum()),
        # Retained nodes visible to at least one microphone.
        "pruned_contribution_nodes": float(visible_retained.sum()),
        "pruned_decision_nodes": float(evaluated.sum() - kept.sum()),
        # Full-tree reductions are filled in after Self-Full-ISM has run.
        "skipped_by_ancestor_nodes": 0.0,
        "expanded_node_reduction": 0.0,
        "contribution_node_reduction": float(1.0 - visible_retained.sum() / max(evaluated.sum(), 1)),
        "pruned_peak_frontier": float(peak_frontier),
        "budget_forced_keep_nodes": float(sum(v["budget_forced_keep"] for v in by_order.values())),
        "budget_forced_prune_nodes": float(sum(v["budget_forced_prune"] for v in by_order.values())),
        "budget_forced_visible_keep_nodes": float(sum(v.get("budget_forced_visible_keep", 0.0) for v in by_order.values())),
        "budget_conflict_orders": float(sum(1.0 for v in by_order.values() if v["budget_conflict"] > 0)),
        "by_order": by_order,
        "parent_valid_nonroot_frac": 1.0,
        "parent_bad_order_nodes": 0.0,
    }
    return rir, stats, data, prob_all, imp_all


def room_to_rir_array(room: object, num_mics: int, rir_len: int) -> np.ndarray:
    out = np.zeros((num_mics, rir_len), dtype=np.float32)
    for m in range(num_mics):
        try:
            r = np.asarray(room.rir[m][0], dtype=np.float32).reshape(-1)
            out[m, : min(rir_len, len(r))] = r[:rir_len]
        except Exception:
            pass
    return out


def run_pyroom_from_geometry(path: Path, cfg: EvalConfig) -> Tuple[np.ndarray, Dict[str, object]]:
    scene = load_geometry_from_npz(path, include_saved_ref=False, max_order_override=cfg.max_order_override)
    room = build_pyroomacoustics_room_from_data(scene)
    room.compute_rir()
    metadata = scene["metadata"]  # type: ignore[assignment]
    fs = int(metadata.get("fs", 8000))
    rir_len = int(round(float(metadata.get("rir_duration", 0.5)) * fs))
    num_mics = int(np.asarray(scene["mic_positions"]).shape[1])
    rir = room_to_rir_array(room, num_mics, rir_len)
    scene["full_rirs"] = rir
    return rir, scene


def run_self_full_from_geometry(path: Path, ckpt_feature_names: Sequence[str], cfg: EvalConfig) -> Tuple[np.ndarray, Dict[str, float], Dict[str, object]]:
    if cfg.full_ism_mode == "stored_nodes":
        return run_full_ism_from_stored(load_stored_full_nodes(path, cfg))
    scene = load_geometry_from_npz(path, include_saved_ref=False, max_order_override=cfg.max_order_override)
    return run_full_online_ism(scene, ckpt_feature_names, cfg)


def run_prune_from_geometry(path: Path, pruning_model: nn.Module, pruning_aux: Dict[str, object], device: torch.device, cfg: EvalConfig) -> Tuple[np.ndarray, Dict[str, object], Dict[str, object], np.ndarray, np.ndarray]:
    scene = load_geometry_from_npz(path, include_saved_ref=False, max_order_override=cfg.max_order_override)
    return run_pruned_online_ism(scene, pruning_model, pruning_aux, device, cfg)


def run_pathrir_from_geometry(path: Path, pruning_model: nn.Module, pruning_aux: Dict[str, object], comp_model: Optional[nn.Module], comp_aux: Optional[Dict[str, object]], device: torch.device, cfg: EvalConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, object], Dict[str, object], Optional[np.ndarray]]:
    pruned, pruned_stats, data, prob, imp = run_prune_from_geometry(path, pruning_model, pruning_aux, device, cfg)
    compensated = pruned
    pred_energy_bins = None
    if comp_model is not None and comp_aux is not None:
        metadata = data["metadata"]  # type: ignore[assignment]
        fs = int(metadata.get("fs", 8000))
        x = build_compensation_features(data, pruned, pruned_stats, imp, comp_aux)
        pred_energy_bins, _ = predict_compensation_energy_bins(comp_model, comp_aux, x, device)
        tail = generate_stochastic_tail(pred_energy_bins, fs, pruned.shape[1], cfg.comp_gain, cfg.comp_seed, cfg.comp_start_ms, cfg.comp_tail_mode)
        compensated = pruned + tail
    return compensated, pruned, pruned_stats, data, pred_energy_bins


# Metrics


def pad_match(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_mics = min(a.shape[0], b.shape[0])
    n = max(a.shape[1], b.shape[1])
    aa = np.zeros((n_mics, n), dtype=np.float32)
    bb = np.zeros((n_mics, n), dtype=np.float32)
    aa[:, : a.shape[1]] = a[:n_mics]
    bb[:, : b.shape[1]] = b[:n_mics]
    return aa, bb


def cosine_distance(ref: np.ndarray, est: np.ndarray) -> float:
    ref, est = pad_match(ref, est)
    vals: List[float] = []
    for m in range(ref.shape[0]):
        x = ref[m].astype(np.float64)
        y = est[m].astype(np.float64)
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        if denom > 1e-20:
            vals.append(float(1.0 - np.dot(x, y) / denom))
    return float(np.mean(vals)) if vals else float("nan")


def nmse_db(ref: np.ndarray, est: np.ndarray) -> float:
    ref, est = pad_match(ref, est)
    num = float(np.sum((ref.astype(np.float64) - est.astype(np.float64)) ** 2))
    den = float(np.sum(ref.astype(np.float64) ** 2))
    return float(10.0 * np.log10(max(num, 1e-30) / max(den, 1e-30)))


def edc_db(h: np.ndarray, floor_db: float = -80.0) -> np.ndarray:
    e = h.astype(np.float64) ** 2
    sch = np.cumsum(e[::-1])[::-1]
    mx = float(np.max(sch))
    if mx <= 1e-30:
        return np.full_like(h, floor_db, dtype=np.float64)
    db = 10.0 * np.log10(np.maximum(sch / mx, 1e-30))
    return np.maximum(db, floor_db)


def edc_rmse_db(ref: np.ndarray, est: np.ndarray, floor_db: float = -80.0) -> float:
    ref, est = pad_match(ref, est)
    return float(np.mean([np.sqrt(np.mean((edc_db(ref[m], floor_db) - edc_db(est[m], floor_db)) ** 2)) for m in range(ref.shape[0])]))


def estimate_rt60(h: np.ndarray, fs: int, mode: str, floor_db: float) -> float:
    curve = edc_db(h, floor_db)
    hi, lo = (-5.0, -35.0) if mode == "t30" else (-5.0, -25.0)
    idx = np.where((curve <= hi) & (curve >= lo))[0]
    if idx.size < 8:
        return float("nan")
    t = idx.astype(np.float64) / float(fs)
    y = curve[idx].astype(np.float64)
    slope, _ = np.linalg.lstsq(np.column_stack([t, np.ones_like(t)]), y, rcond=None)[0]
    return float(-60.0 / slope) if slope < -1e-9 else float("nan")


def rt60_abs_error(ref: np.ndarray, est: np.ndarray, fs: int, mode: str, floor_db: float) -> float:
    ref, est = pad_match(ref, est)
    vals = []
    for m in range(ref.shape[0]):
        r = estimate_rt60(ref[m], fs, mode, floor_db)
        e = estimate_rt60(est[m], fs, mode, floor_db)
        if np.isfinite(r) and np.isfinite(e):
            vals.append(abs(e - r))
    return float(np.mean(vals)) if vals else float("nan")


def drr_db(h: np.ndarray, fs: int, src: np.ndarray, mic: np.ndarray, c: float, direct_window_ms: float, frac_delay_len: int) -> float:
    dist = float(np.linalg.norm(src.astype(np.float64) - mic.astype(np.float64)))
    direct_sample = int(round((dist / c) * fs + frac_delay_len // 2))
    half = max(1, int(round((direct_window_ms / 1000.0) * fs / 2.0)))
    s = max(0, direct_sample - half)
    t = min(len(h), direct_sample + half + 1)
    direct = float(np.sum(h[s:t].astype(np.float64) ** 2))
    total = float(np.sum(h.astype(np.float64) ** 2))
    return float(10.0 * np.log10(max(direct, 1e-30) / max(total - direct, 1e-30)))


def drr_abs_error(ref: np.ndarray, est: np.ndarray, data: Dict[str, object], cfg: EvalConfig) -> float:
    ref, est = pad_match(ref, est)
    metadata = data["metadata"]  # type: ignore[assignment]
    fs = int(metadata.get("fs", 8000))
    c, frac_delay_len = parse_c_and_frac_delay(metadata)
    src = data["source_position"]  # type: ignore[assignment]
    mics = data["mic_positions"]  # type: ignore[assignment]
    vals = []
    for m in range(ref.shape[0]):
        r = drr_db(ref[m], fs, src, mics[:, m], c, cfg.direct_window_ms, frac_delay_len)
        e = drr_db(est[m], fs, src, mics[:, m], c, cfg.direct_window_ms, frac_delay_len)
        vals.append(abs(e - r))
    return float(np.mean(vals)) if vals else float("nan")


def energy_db(x: np.ndarray) -> float:
    return float(10.0 * np.log10(max(float(np.sum(x.astype(np.float64) ** 2)), 1e-30)))


def pair_metrics(ref: np.ndarray, est: np.ndarray, data: Dict[str, object], cfg: EvalConfig, prefix: str) -> Dict[str, float]:
    metadata = data["metadata"]  # type: ignore[assignment]
    fs = int(metadata.get("fs", 8000))
    return {
        f"{prefix}_cosine_distance": cosine_distance(ref, est),
        f"{prefix}_nmse_db": nmse_db(ref, est),
        f"{prefix}_edc_rmse_db": edc_rmse_db(ref, est, cfg.edc_floor_db),
        f"{prefix}_rt60_abs_error_s": rt60_abs_error(ref, est, fs, cfg.rt60_mode, cfg.edc_floor_db),
        f"{prefix}_drr_abs_error_db": drr_abs_error(ref, est, data, cfg),
        f"{prefix}_energy_error_db": energy_db(est) - energy_db(ref),
    }


def method_metrics(rir: np.ndarray, data: Dict[str, object], cfg: EvalConfig, prefix: str) -> Dict[str, float]:
    metadata = data["metadata"]  # type: ignore[assignment]
    fs = int(metadata.get("fs", 8000))
    vals_rt = [estimate_rt60(rir[m], fs, cfg.rt60_mode, cfg.edc_floor_db) for m in range(rir.shape[0])]
    return {
        f"{prefix}_energy_db": energy_db(rir),
        f"{prefix}_rt60_mean_s": float(np.nanmean(vals_rt)),
    }


# Timing, per-room evaluation, summaries


def timed_peak_call(fn, repeats: int, profile_memory: bool = True) -> Tuple[object, float, float]:
    """Measure median wall time and, optionally, peak Python memory.

    Memory tracing adds overhead to Python-heavy code. Set
    ``profile_memory=False`` for runtime comparisons; peak memory is then
    reported as NaN.
    """
    times: List[float] = []
    peaks: List[float] = []
    result = None
    for _ in range(max(1, repeats)):
        if profile_memory:
            tracemalloc.start()
        t0 = time.perf_counter()
        result = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if profile_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks.append(peak / (1024.0 * 1024.0))
        times.append(elapsed)
    peak_mb = float(max(peaks)) if peaks else float("nan")
    return result, float(statistics.median(times)), peak_mb


def evaluate_one_room(
    path: Path,
    pruning_model: nn.Module,
    pruning_aux: Dict[str, object],
    comp_model: Optional[nn.Module],
    comp_aux: Optional[Dict[str, object]],
    device: torch.device,
    cfg: EvalConfig,
    out_rir_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Evaluate all methods for one room.

    Prune-RIR and PathRIR run separately so compensation is excluded from the
    pruning-only timing. The pyroomacoustics reference is loaded from the
    dataset unless end-to-end timing is requested.
    """
    # Load the stored reference unless an end-to-end pyroom run is requested.
    if cfg.measure_pyroom_runtime:
        (pyroom_rir, pyroom_scene), pyroom_sec, pyroom_peak_mb = timed_peak_call(
            lambda: run_pyroom_from_geometry(path, cfg),
            cfg.pyroom_timing_repeats,
            cfg.profile_memory,
        )
    else:
        pyroom_scene = load_geometry_from_npz(
            path,
            include_saved_ref=True,
            max_order_override=cfg.max_order_override,
        )
        pyroom_rir = np.asarray(pyroom_scene["full_rirs"], dtype=np.float32)
        pyroom_sec = float("nan")
        pyroom_peak_mb = float("nan")

    metadata = pyroom_scene["metadata"]  # type: ignore[assignment]
    fs = int(metadata.get("fs", 8000))

    # stored_nodes measures accumulation only; the other modes also build nodes.
    if cfg.full_ism_mode == "stored_nodes":
        stored_nodes = load_stored_full_nodes(path, cfg)
        (self_full, self_full_stats, self_data), self_full_sec, self_full_peak_mb = timed_peak_call(
            lambda: run_full_ism_from_stored(stored_nodes),
            cfg.timing_repeats,
            cfg.profile_memory,
        )
    else:
        (self_full, self_full_stats, self_data), self_full_sec, self_full_peak_mb = timed_peak_call(
            lambda: run_self_full_from_geometry(path, pruning_aux["feature_names"], cfg),  # type: ignore[arg-type]
            cfg.timing_repeats,
            cfg.profile_memory,
        )

    # Time the pruning-only path without compensation features.
    (pruned, pruned_stats, pruned_data, _pruned_prob, _pruned_imp), prunerir_sec, prunerir_peak_mb = timed_peak_call(
        lambda: run_prune_from_geometry(path, pruning_model, pruning_aux, device, cfg),
        cfg.timing_repeats,
        cfg.profile_memory,
    )

    # Time the complete PathRIR path, including compensation when available.
    (
        pathrir,
        _pathrir_pruned,
        _pathrir_pruned_stats,
        pathrir_data,
        pred_energy_bins,
    ), pathrir_sec, pathrir_peak_mb = timed_peak_call(
        lambda: run_pathrir_from_geometry(
            path,
            pruning_model,
            pruning_aux,
            comp_model,
            comp_aux,
            device,
            cfg,
        ),
        cfg.timing_repeats,
        cfg.profile_memory,
    )

    # R_img compares retained visible nodes with the full visible node set.
    # true_expand also reports a reduction against all expanded nodes.
    self_vis = np.asarray(self_data["node_visibility"], dtype=bool)  # type: ignore[index]
    full_nodes = float(self_vis.any(axis=0).sum())
    full_nodes_expanded = float(self_data["node_images"].shape[0])  # type: ignore[index]
    path_nodes = float(pruned_stats.get("pruned_contribution_nodes", full_nodes))
    path_retained = float(pruned_stats.get("pruned_retained_nodes", path_nodes))
    path_expanded = float(pruned_stats.get("pruned_expanded_nodes", path_nodes))
    pruned_stats["contribution_node_reduction"] = float(1.0 - path_nodes / max(full_nodes, 1.0))
    pruned_stats["expanded_node_reduction"] = float(1.0 - path_expanded / max(full_nodes, 1.0))
    pruned_stats["skipped_by_ancestor_nodes"] = float(max(full_nodes - path_expanded, 0.0))
    room_name = path.stem

    rec: Dict[str, object] = {
        "file": path.name,
        "num_nodes": full_nodes,
        "num_nodes_expanded": full_nodes_expanded if cfg.full_ism_mode == "true_expand" else float("nan"),
        "PathRIR_R_img_vs_expanded": (1.0 - path_nodes / max(full_nodes_expanded, 1.0)) if cfg.full_ism_mode == "true_expand" else float("nan"),
        "num_mics": float(pyroom_rir.shape[0]),
        "fs": float(fs),
        "max_order": float(metadata.get("max_order", np.max(pruned_data["node_order"]))),  # type: ignore[index]
        "decision_mode": cfg.decision_mode,
        "prob_threshold": float(cfg.prob_threshold),
        "importance_threshold": float(cfg.importance_threshold),
        "order_budget_mode": cfg.order_budget_mode,
        "early_keep_order": float(cfg.early_keep_order),
        "min_keep_rate_after": float(cfg.min_keep_rate_after),
        "max_keep_rate_after": float(cfg.max_keep_rate_after),
        "min_keep_count_after": float(cfg.min_keep_count_after),
        "budget_score": cfg.budget_score,
        "pyroomacoustics_runtime_ms": 1000.0 * pyroom_sec if np.isfinite(pyroom_sec) else float("nan"),
        "pyroomacoustics_peak_mem_mb": pyroom_peak_mb,
        "self_full_runtime_ms": 1000.0 * self_full_sec,
        "self_full_peak_mem_mb": self_full_peak_mb,
        "Prune-RIR_runtime_ms": 1000.0 * prunerir_sec,
        "Prune-RIR_peak_mem_mb": prunerir_peak_mb,
        "PathRIR_runtime_ms": 1000.0 * pathrir_sec,
        "PathRIR_peak_mem_mb": pathrir_peak_mb,
        "PathRIR_R_img": 1.0 - path_nodes / max(full_nodes, 1.0),
        "PathRIR_S_rt": pyroom_sec / max(pathrir_sec, 1e-12) if np.isfinite(pyroom_sec) else float("nan"),
        "PathRIR_contribution_nodes": path_nodes,
        "PathRIR_retained_nodes": path_retained,
        "Self-Full-ISM_R_img": 0.0,
        "Self-Full-ISM_S_rt": pyroom_sec / max(self_full_sec, 1e-12) if np.isfinite(pyroom_sec) else float("nan"),
        "Prune-RIR_R_img": 1.0 - path_nodes / max(full_nodes, 1.0),
        "Prune-RIR_S_rt": pyroom_sec / max(prunerir_sec, 1e-12) if np.isfinite(pyroom_sec) else float("nan"),
        # Speedup relative to Self-Full-ISM with the same accumulation backend.
        "Self-Full-ISM_S_rt_vs_full": 1.0,
        "Prune-RIR_S_rt_vs_full": self_full_sec / max(prunerir_sec, 1e-12),
        "PathRIR_S_rt_vs_full": self_full_sec / max(pathrir_sec, 1e-12),
        "comp_enabled": bool(comp_model is not None),
        "full_ism_mode": cfg.full_ism_mode,
    }
    rec.update({f"self_full_{k}": v for k, v in self_full_stats.items() if isinstance(v, float)})
    rec.update({k: v for k, v in pruned_stats.items() if isinstance(v, float)})
    rec.update(method_metrics(pyroom_rir, pruned_data, cfg, "pyroomacoustics"))
    rec.update(method_metrics(self_full, self_data, cfg, "self_full"))
    rec.update(method_metrics(pruned, pruned_data, cfg, "self_pruned"))
    rec.update(method_metrics(pathrir, pathrir_data, cfg, "self_compensated"))
    rec.update(pair_metrics(pyroom_rir, self_full, self_data, cfg, "self_full_vs_pyroomacoustics"))
    rec.update(pair_metrics(pyroom_rir, pruned, pruned_data, cfg, "pruned_vs_pyroomacoustics"))
    rec.update(pair_metrics(pyroom_rir, pathrir, pathrir_data, cfg, "PathRIR_vs_pyroomacoustics"))
    if pred_energy_bins is not None:
        rec["comp_ckpt"] = cfg.comp_ckpt
        rec["comp_gain"] = float(cfg.comp_gain)
        rec["comp_start_ms"] = float(cfg.comp_start_ms)
        rec["comp_tail_mode"] = str(cfg.comp_tail_mode)
        rec["comp_predicted_residual_energy_db"] = float(
            10.0 * np.log10(max(float(np.sum(pred_energy_bins)), 1e-30))
        )

    if out_rir_dir is not None:
        out_rir_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_rir_dir / f"{path.stem}_rirs.npz",
            pyroomacoustics_full=pyroom_rir,
            self_full=self_full,
            self_pruned=pruned,
            pathrir=pathrir,
            comp_pred_energy_bins=(
                pred_energy_bins if pred_energy_bins is not None else np.zeros(1, dtype=np.float32)
            ),
        )

    if cfg.save_rir_wavs:
        wav_base = Path(cfg.rir_wav_dir)
        if not wav_base.is_absolute():
            wav_base = Path(cfg.out_dir) / wav_base
        save_method_rirs_as_wavs(wav_base, "Pyroom", room_name, pyroom_rir, fs, source_name="S1")
        save_method_rirs_as_wavs(wav_base, "Self-Full-ISM", room_name, self_full, fs, source_name="S1")
        save_method_rirs_as_wavs(wav_base, "PathRIR", room_name, pathrir, fs, source_name="S1")

    return rec

def is_number(x: object) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(float(x))


def summarize(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    keys = sorted({k for r in records for k in r.keys()})
    metrics: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [float(r[k]) for r in records if k in r and is_number(r[k])]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            metrics[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "p05": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
    return {"num_rooms": len(records), "metrics": metrics}


def paper_rows_from_record(rec: Dict[str, object]) -> List[Dict[str, object]]:
    """Convert one detailed room record into long-format paper metric rows."""
    rows: List[Dict[str, object]] = []

    def add_row(method: str, prefix: str, r_img_key: str, s_rt_key: str, runtime_key: str) -> None:
        rows.append({
            "file": rec.get("file", ""),
            "method": method,
            "R_img": rec.get(r_img_key, float("nan")),
            "S_rt": rec.get(s_rt_key, float("nan")),
            "CD": rec.get(f"{prefix}_cosine_distance", float("nan")),
            "NMSE_dB": rec.get(f"{prefix}_nmse_db", float("nan")),
            "EDC_Err_dB": rec.get(f"{prefix}_edc_rmse_db", float("nan")),
            "RT60_Err_s": rec.get(f"{prefix}_rt60_abs_error_s", float("nan")),
            "RT60_Err_ms": 1000.0 * float(rec.get(f"{prefix}_rt60_abs_error_s", float("nan"))) if is_number(rec.get(f"{prefix}_rt60_abs_error_s")) else float("nan"),
            "DRR_Err_dB": rec.get(f"{prefix}_drr_abs_error_db", float("nan")),
            "runtime_ms": rec.get(runtime_key, float("nan")),
            "num_nodes": rec.get("num_nodes", float("nan")),
            "num_mics": rec.get("num_mics", float("nan")),
        })

    # Paper rows use Self-Full-ISM as the speedup reference. When pyroom timing
    # is enabled, the corresponding ratios remain in the detailed CSV.
    add_row(
        method="Self-Full-ISM",
        prefix="self_full_vs_pyroomacoustics",
        r_img_key="Self-Full-ISM_R_img",
        s_rt_key="Self-Full-ISM_S_rt_vs_full",
        runtime_key="self_full_runtime_ms",
    )
    add_row(
        method="Prune-RIR",
        prefix="pruned_vs_pyroomacoustics",
        r_img_key="Prune-RIR_R_img",
        s_rt_key="Prune-RIR_S_rt_vs_full",
        runtime_key="Prune-RIR_runtime_ms",
    )
    if bool(rec.get("comp_enabled", False)):
        add_row(
            method="PathRIR",
            prefix="PathRIR_vs_pyroomacoustics",
            r_img_key="PathRIR_R_img",
            s_rt_key="PathRIR_S_rt_vs_full",
            runtime_key="PathRIR_runtime_ms",
        )
    return rows


def make_paper_rows(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for rec in records:
        if "error" in rec:
            continue
        rows.extend(paper_rows_from_record(rec))
    return rows


def summarize_paper_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    methods = sorted({str(r.get("method", "")) for r in rows})
    metrics = ["R_img", "S_rt", "CD", "NMSE_dB", "EDC_Err_dB", "RT60_Err_ms", "DRR_Err_dB", "runtime_ms"]
    out: Dict[str, object] = {"num_rows": len(rows), "methods": {}}
    for method in methods:
        method_rows = [r for r in rows if str(r.get("method", "")) == method]
        stats: Dict[str, Dict[str, float]] = {}
        for k in metrics:
            vals = [float(r[k]) for r in method_rows if k in r and is_number(r[k])]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                stats[k] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "median": float(np.median(arr)),
                    "p05": float(np.percentile(arr, 5)),
                    "p95": float(np.percentile(arr, 95)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }
        out["methods"][method] = {"num_rooms": len(method_rows), "metrics": stats}  # type: ignore[index]
    return out


def write_paper_summary_csv(path: Path, paper_summary: Dict[str, object]) -> None:
    methods = paper_summary.get("methods", {})
    fields = ["method", "metric", "mean", "std", "median", "p05", "p95", "min", "max"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        if isinstance(methods, dict):
            for method, payload in methods.items():
                metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
                for metric, stats in metrics.items():
                    row = {"method": method, "metric": metric}
                    row.update(stats)
                    writer.writerow(row)


def write_csv(path: Path, records: Sequence[Dict[str, object]]) -> None:
    if not records:
        return
    fields = sorted({k for r in records for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def print_key_summary(summary: Dict[str, object]) -> None:
    metrics: Dict[str, Dict[str, float]] = summary.get("metrics", {})  # type: ignore[assignment]
    keys = [
        "pyroomacoustics_runtime_ms",
        "Self-Full-ISM_S_rt",
        "PathRIR_R_img",
        "Prune-RIR_S_rt_vs_full",
        "PathRIR_S_rt_vs_full",
        "PathRIR_S_rt",
        "PathRIR_vs_pyroomacoustics_cosine_distance",
        "PathRIR_vs_pyroomacoustics_nmse_db",
        "PathRIR_vs_pyroomacoustics_edc_rmse_db",
        "PathRIR_vs_pyroomacoustics_rt60_abs_error_s",
        "PathRIR_vs_pyroomacoustics_drr_abs_error_db",
        "self_full_vs_pyroomacoustics_cosine_distance",
        "self_full_vs_pyroomacoustics_nmse_db",
        "self_full_vs_pyroomacoustics_edc_rmse_db",
    ]
    print("\nDetailed metric summary (mean ± std, median [p05, p95])")
    for k in keys:
        if k not in metrics:
            continue
        m = metrics[k]
        print(f"{k}: mean={m['mean']:.6g} std={m['std']:.6g} median={m['median']:.6g} [{m['p05']:.6g}, {m['p95']:.6g}]")


def print_paper_summary(paper_summary: Dict[str, object]) -> None:
    print("\nPaper metrics against the pyroomacoustics reference")
    methods = paper_summary.get("methods", {})
    if not isinstance(methods, dict):
        return
    ordered_metrics = ["R_img", "S_rt", "CD", "NMSE_dB", "EDC_Err_dB", "RT60_Err_ms", "DRR_Err_dB"]
    for method in sorted(methods):
        payload = methods[method]
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        print(f"[{method}]")
        for k in ordered_metrics:
            if k not in metrics:
                continue
            m = metrics[k]
            print(f"  {k}: mean={m['mean']:.6g} std={m['std']:.6g} median={m['median']:.6g}")


# Command-line interface


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate PathRIR, Prune-RIR, and Self-Full-ISM on saved rooms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--ckpt", required=True, help="Path to the Pruning-MLP checkpoint.")
    p.add_argument("--comp-ckpt", default="",
                   help="Path to the Compensation-MLP checkpoint; omit for pruning-only evaluation.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8192)

    p.add_argument("--decision-mode", choices=["prob", "importance", "either", "both"], default="prob")
    p.add_argument("--prob-threshold", type=float, default=0.5)
    p.add_argument("--importance-threshold", type=float, default=1e-4)

    p.add_argument("--order-budget-mode", choices=["none", "score"], default="score",
                   help="Use 'score' for the per-order budget or 'none' for threshold decisions only.")
    p.add_argument("--early-keep-order", type=int, default=1)
    p.add_argument("--min-keep-rate-after", type=float, default=0.20)
    p.add_argument("--max-keep-rate-after", type=float, default=0.50)
    p.add_argument("--min-keep-count-after", type=int, default=48)
    p.add_argument("--budget-score", choices=["importance", "prob"], default="importance")
    p.add_argument("--min-visible-keep-count", type=int, default=0,
                   help="Minimum visible candidates kept per order after applying the budget.")

    p.add_argument("--comp-gain", type=float, default=1.0)
    p.add_argument("--comp-seed", type=int, default=0)
    p.add_argument("--comp-start-ms", type=float, default=40.0)
    p.add_argument("--comp-tail-mode", choices=["absolute", "signed"], default="absolute",
                   help="'absolute' rectifies the stochastic tail; 'signed' preserves its sign.")

    p.add_argument("--timing-repeats", type=int, default=3,
                   help="Number of timing runs for Self-Full-ISM and PathRIR.")
    p.add_argument("--no-mem-profiling", dest="profile_memory", action="store_false",
                   help="Disable peak-memory tracing during timing. Use this for runtime comparisons.")
    p.add_argument("--pyroom-timing-repeats", type=int, default=1,
                   help="Number of end-to-end pyroomacoustics timing runs.")
    p.add_argument("--direct-window-ms", type=float, default=2.5)
    p.add_argument("--edc-floor-db", type=float, default=-80.0)
    p.add_argument("--rt60-mode", choices=["t20", "t30"], default="t20")
    p.add_argument("--csv-name", default="detailed_room_metrics.csv",
                   help="Filename for detailed per-room metrics.")
    p.add_argument("--paper-csv-name", default="paper_metrics_long.csv",
                   help="Filename for long-format metrics with one row per room and method.")
    p.add_argument("--paper-summary-csv-name", default="paper_summary.csv",
                   help="Filename for the summary table.")
    p.add_argument("--summary-name", default="summary.json")
    p.add_argument("--skip-pyroom-runtime", dest="measure_pyroom_runtime", action="store_false",
                   help="Use the reference RIR stored in each dataset file.")
    p.add_argument("--measure-pyroom-runtime", dest="measure_pyroom_runtime", action="store_true",
                   help="Rebuild and time the pyroomacoustics RIR from room geometry.")
    p.set_defaults(measure_pyroom_runtime=False)
    p.add_argument("--save-rirs", action="store_true",
                   help="Save each room's RIR arrays under <out-dir>/rirs.")
    p.add_argument("--save-rir-wavs", action="store_true",
                   help="Save pyroomacoustics, Self-Full-ISM, and PathRIR RIRs as WAV files.")
    p.add_argument("--rir-wav-dir", default="Output_RIR",
                   help="WAV output folder. Relative paths are placed under <out-dir>.")
    p.add_argument("--budget-relative-to", choices=["visible", "all"], default="visible",
                   help="Candidate set used to calculate per-order keep rates.")
    p.add_argument("--full-ism-mode", choices=["pyroom_extract", "stored_nodes", "true_expand"], default="pyroom_extract",
                   help="Source of full-ISM nodes: pyroomacoustics, saved dataset nodes, or polygon-engine expansion.")
    p.add_argument("--max-order-override", type=int, default=0,
                   help="Positive values replace the dataset's maximum reflection order for every method.")
    return p.parse_args()


def args_to_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(**vars(args))


def main() -> None:
    args = parse_args()
    cfg = args_to_config(args)
    seed_everything(cfg.seed)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8")

    files = find_npz_files(Path(cfg.data_dir), cfg.max_files, cfg.seed)

    device = resolve_device(cfg.device)
    pruning_model, pruning_aux = load_pruning_model(Path(cfg.ckpt), device)
    comp_model = None
    comp_aux = None
    if cfg.comp_ckpt:
        comp_model, comp_aux = load_compensation_model(Path(cfg.comp_ckpt), device)

    print(f"Dataset: {cfg.data_dir}")
    print(f"Rooms: {len(files)}")
    print(f"Device: {device}")
    print(f"Pruning checkpoint: {cfg.ckpt}")
    print(f"Compensation checkpoint: {cfg.comp_ckpt or 'not used'}")
    print(f"Compensation tail: {cfg.comp_tail_mode if cfg.comp_ckpt else 'not used'}")
    print(
        f"Order budget: mode={cfg.order_budget_mode}, early_keep_order={cfg.early_keep_order}, "
        f"min_rate={cfg.min_keep_rate_after}, max_rate={cfg.max_keep_rate_after}, "
        f"min_count={cfg.min_keep_count_after}"
    )
    print(
        f"pyroomacoustics timing: "
        f"{'enabled' if cfg.measure_pyroom_runtime else 'disabled'}, "
        f"repeats={cfg.pyroom_timing_repeats}"
    )

    records: List[Dict[str, object]] = []
    csv_path = out / cfg.csv_name
    paper_csv_path = out / cfg.paper_csv_name
    paper_summary_csv_path = out / cfg.paper_summary_csv_name
    rir_dir = out / "rirs" if cfg.save_rirs else None
    if cfg.save_rir_wavs:
        wav_base = Path(cfg.rir_wav_dir)
        wav_dir = wav_base if wav_base.is_absolute() else out / wav_base
        wav_dir.mkdir(parents=True, exist_ok=True)
    else:
        wav_dir = None
    iterator: Iterable[Path] = tqdm(files, desc="Evaluating rooms", dynamic_ncols=True) if tqdm is not None else files
    for i, path in enumerate(iterator):
        try:
            rec = evaluate_one_room(
                path,
                pruning_model,
                pruning_aux,
                comp_model,
                comp_aux,
                device,
                cfg,
                rir_dir,
            )
            records.append(rec)
            write_csv(csv_path, records)
            paper_rows = make_paper_rows(records)
            write_csv(paper_csv_path, paper_rows)
            write_paper_summary_csv(paper_summary_csv_path, summarize_paper_rows(paper_rows))
            if tqdm is not None:
                iterator.set_postfix(
                    Rimg=f"{rec.get('PathRIR_R_img', float('nan')):.2f}",
                    Srt=f"{rec.get('PathRIR_S_rt_vs_full', float('nan')):.2f}x",
                    EDC=f"{rec.get('PathRIR_vs_pyroomacoustics_edc_rmse_db', float('nan')):.2g}",
                )
            else:
                print(
                    f"[{i+1}/{len(files)}] {path.name}: "
                    f"R_img={rec.get('PathRIR_R_img', float('nan')):.3f}, "
                    f"S_rt_vs_full={rec.get('PathRIR_S_rt_vs_full', float('nan')):.2f}x"
                )
        except Exception as exc:
            err = {"file": path.name, "error": repr(exc)}
            records.append(err)
            write_csv(csv_path, records)
            print(f"[error] {path.name}: {exc}", file=sys.stderr, flush=True)

    summary = summarize(records)
    paper_rows = make_paper_rows(records)
    paper_summary = summarize_paper_rows(paper_rows)
    (out / cfg.summary_name).write_text(json.dumps({"detailed": summary, "paper": paper_summary}, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(paper_csv_path, paper_rows)
    write_paper_summary_csv(paper_summary_csv_path, paper_summary)
    print_key_summary(summary)
    print_paper_summary(paper_summary)
    print(f"\nSaved detailed room metrics: {csv_path}")
    print(f"Saved long-format paper metrics: {paper_csv_path}")
    print(f"Saved paper summary: {paper_summary_csv_path}")
    print(f"Saved JSON summary: {out / cfg.summary_name}")
    if cfg.save_rir_wavs:
        print(f"Saved WAV RIRs: {wav_dir}")


if __name__ == "__main__":
    main()
