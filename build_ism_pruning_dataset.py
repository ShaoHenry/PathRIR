#!/usr/bin/env python3
"""Create the room datasets used to train and evaluate PathRIR.

Each room starts from a random 2-D polygon and is extruded to a 3-D volume.
pyroomacoustics provides the full-ISM reference RIR and visible image sources.
The script reconstructs their parent chains, inserts any invisible connector
nodes, calculates subtree labels, and saves one compressed ``.npz`` file per
room. It also writes a manifest with the result of each room.

Example:
    python build_ism_pruning_dataset.py \
        --out-dir ./data/train_order10 \
        --num-configs 1000 \
        --max-order 10 \
        --num-workers 1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import pyroomacoustics as pra
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pyroomacoustics is required. Install with: pip install pyroomacoustics"
    ) from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from polygon_ism_engine import ExtrudedPolygonRoom


# Configuration


@dataclass(frozen=True)
class DatasetConfig:
    out_dir: str
    num_configs: int = 1000
    seed: int = 0
    fs: int = 8000
    max_order: int = 15
    rir_duration: float = 0.5
    num_mics: int = 2
    mic_spacing: float = 0.06
    label_eps: float = 1e-4
    min_src_mic_dist: float = 0.75
    min_wall_margin: float = 0.15
    compress: bool = True
    rir_hpf_enable: bool = False

    min_vertices: int = 5
    max_vertices: int = 10
    width_min: float = 3.0
    width_max: float = 12.0
    length_min: float = 3.0
    length_max: float = 12.0
    height_min: float = 2.2
    height_max: float = 4.5

    side_abs_min: float = 0.04
    side_abs_max: float = 0.55
    floor_abs_min: float = 0.05
    floor_abs_max: float = 0.70
    ceiling_abs_min: float = 0.03
    ceiling_abs_max: float = 0.45

    radial_jitter: float = 0.35
    chain_match_tol: float = 1e-4  # Position tolerance used to match image-source chains.
    chain_blind_limit: int = 4  # Maximum run of missing ancestors during chain search.
    dead_negatives_ratio: float = 1.0  # Negative samples per tree node; 0 disables sampling.


FEATURE_NAMES = np.array(
    [
        "order",
        "wall_id",
        "parent_order",
        "node_wall_absorption",
        "image_x",
        "image_y",
        "image_z",
        "damping_mean",
        "damping_min",
        "damping_max",
        "dist_min",
        "dist_mean",
        "dist_max",
        "delay_min",
        "delay_mean",
        "delay_max",
        "visible_count",
        "visible_frac",
        "room_num_walls",
        "room_floor_area",
        "room_height",
        "room_volume",
        "wall_abs_mean",
        "wall_abs_std",
        "src_x",
        "src_y",
        "src_z",
        "mic_centroid_x",
        "mic_centroid_y",
        "mic_centroid_z",
    ],
    dtype=object,
)


# Geometry helpers


def signed_polygon_area(poly_xy: np.ndarray) -> float:
    x = poly_xy[:, 0]
    y = poly_xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(poly_xy: np.ndarray) -> float:
    return abs(signed_polygon_area(poly_xy))


def ensure_ccw(poly_xy: np.ndarray) -> np.ndarray:
    if signed_polygon_area(poly_xy) < 0:
        return poly_xy[::-1].copy()
    return poly_xy.copy()


def point_in_polygon(point_xy: np.ndarray, poly_xy: np.ndarray) -> bool:
    x, y = float(point_xy[0]), float(point_xy[1])
    inside = False
    n = len(poly_xy)
    x0, y0 = poly_xy[-1]
    for i in range(n):
        x1, y1 = poly_xy[i]
        intersects = ((y1 > y) != (y0 > y)) and (
            x < (x0 - x1) * (y - y1) / ((y0 - y1) + 1e-12) + x1
        )
        if intersects:
            inside = not inside
        x0, y0 = x1, y1
    return inside


def min_edge_length(poly_xy: np.ndarray) -> float:
    edges = np.roll(poly_xy, -1, axis=0) - poly_xy
    return float(np.linalg.norm(edges, axis=1).min())


def random_star_polygon(rng: np.random.Generator, cfg: DatasetConfig) -> np.ndarray:
    for _ in range(200):
        n = int(rng.integers(cfg.min_vertices, cfg.max_vertices + 1))
        base = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        angle_jitter = rng.uniform(-0.35, 0.35, size=n) * (2.0 * np.pi / n)
        angles = np.sort((base + angle_jitter) % (2.0 * np.pi))

        radii = 1.0 + cfg.radial_jitter * rng.normal(size=n)
        radii = np.clip(radii, 0.45, 1.65)
        radii = 0.25 * np.roll(radii, 1) + 0.50 * radii + 0.25 * np.roll(radii, -1)

        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts = ensure_ccw(pts)

        width = float(rng.uniform(cfg.width_min, cfg.width_max))
        length = float(rng.uniform(cfg.length_min, cfg.length_max))
        pts[:, 0] -= pts[:, 0].min()
        pts[:, 1] -= pts[:, 1].min()
        pts[:, 0] *= width / max(float(np.ptp(pts[:, 0])), 1e-6)
        pts[:, 1] *= length / max(float(np.ptp(pts[:, 1])), 1e-6)
        pts += np.array([0.25, 0.25])
        pts = ensure_ccw(pts)

        if polygon_area(pts) > 4.0 and min_edge_length(pts) > 0.7:
            return pts.astype(np.float32)

    raise RuntimeError("Failed to generate a valid polygon after many attempts.")


def sample_xy_in_polygon(
    rng: np.random.Generator,
    poly_xy: np.ndarray,
    margin: float = 0.15,
    max_tries: int = 10000,
) -> np.ndarray:
    lo = poly_xy.min(axis=0) + margin
    hi = poly_xy.max(axis=0) - margin
    if np.any(hi <= lo):
        raise ValueError("Polygon bounding box too small for requested margin.")
    for _ in range(max_tries):
        p = rng.uniform(lo, hi)
        if point_in_polygon(p, poly_xy):
            return p.astype(np.float32)
    raise RuntimeError("Failed to sample a point inside polygon.")


def sample_point_in_room(
    rng: np.random.Generator,
    poly_xy: np.ndarray,
    height: float,
    margin: float = 0.15,
) -> np.ndarray:
    xy = sample_xy_in_polygon(rng, poly_xy, margin=margin)
    z_lo = max(0.3, margin)
    z_hi = max(z_lo + 0.1, height - margin)
    z = float(rng.uniform(z_lo, z_hi))
    return np.array([xy[0], xy[1], z], dtype=np.float32)


def sample_microphone_array(
    rng: np.random.Generator,
    poly_xy: np.ndarray,
    height: float,
    num_mics: int,
    spacing: float,
    margin: float,
) -> np.ndarray:
    if num_mics == 1:
        return sample_point_in_room(rng, poly_xy, height, margin)[:, None]

    for _ in range(2000):
        center = sample_point_in_room(rng, poly_xy, height, margin)
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        direction = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float32)
        offsets = (np.arange(num_mics, dtype=np.float32) - (num_mics - 1) / 2.0) * spacing
        mics = center[:, None] + direction[:, None] * offsets[None, :]
        ok = True
        for m in range(num_mics):
            if not point_in_polygon(mics[:2, m], poly_xy):
                ok = False
                break
            if not (margin <= mics[2, m] <= height - margin):
                ok = False
                break
        if ok:
            return mics.astype(np.float32)
    raise RuntimeError("Failed to sample microphone array inside room.")


# Image-source tree reconstruction


def build_true_tree(
    engine: ExtrudedPolygonRoom,
    src_pos: np.ndarray,
    pyroom_images: np.ndarray,
    pyroom_damping: np.ndarray,
    pyroom_orders: np.ndarray,
    pyroom_visibility: np.ndarray,
    num_mics: int,
    tol: float,
    blind_limit: int = 4,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, float],
    List[Tuple[int, ...]],
    Dict[Tuple[int, ...], int],
]:
    """Reconstruct the image-source tree behind pyroomacoustics' visible nodes.

    Invisible ancestors are inserted where needed. Row 0 is the real source,
    and wall indices follow :class:`ExtrudedPolygonRoom`. The returned wall
    sequences and registry are reused when sampling negative candidates.
    """
    src_pos = np.asarray(src_pos, dtype=np.float64).reshape(3)
    n_py = int(pyroom_images.shape[0])

    def poskey(pos: np.ndarray, k: int) -> Tuple[int, int, int, int]:
        return (k,) + tuple(int(round(float(v) / (tol * 0.25))) for v in pos)

    # Wall sequences, rather than positions, identify nodes. Different
    # corner-bounce sequences can lead to the same point and still contribute
    # separately to the RIR.
    images: List[np.ndarray] = [src_pos.copy()]
    walls: List[int] = [-1]
    parents: List[int] = [-1]
    orders: List[int] = [0]
    damping: List[float] = [1.0]
    visibility: List[np.ndarray] = [np.ones(num_mics, dtype=np.uint8)]
    matched: List[int] = [0]
    glued: List[int] = [0]
    seq_of_row: List[Tuple[int, ...]] = [tuple()]
    registry: Dict[Tuple[int, ...], int] = {tuple(): 0}
    # Map each quantized (order, position) cell to its materialized rows.
    pos_registry: Dict[Tuple[int, int, int, int], List[int]] = {poskey(src_pos, 0): [0]}

    def materialize(seq: Tuple[int, ...]) -> int:
        """Insert a wall sequence, reuse existing prefixes, and return its row."""
        cur_row = 0
        cur = src_pos.copy()
        cur_damp = 1.0
        for depth in range(1, len(seq) + 1):
            prefix = seq[:depth]
            w = seq[depth - 1]
            cur = engine.reflect(cur, w)
            cur_damp *= float(engine.wall_beta[w])
            row = registry.get(prefix)
            if row is None:
                images.append(cur.copy())
                walls.append(int(w))
                parents.append(cur_row)
                orders.append(depth)
                damping.append(cur_damp)
                visibility.append(np.zeros(num_mics, dtype=np.uint8))
                matched.append(0)
                glued.append(0)
                seq_of_row.append(prefix)
                row = len(images) - 1
                registry[prefix] = row
                pos_registry.setdefault(poskey(cur, depth), []).append(row)
            cur_row = row
        return cur_row

    def glue_fallback(pos: np.ndarray, k: int) -> Tuple[int, int, float]:
        """Find the closest registered parent when an exact chain is unavailable."""
        cand_rows = [r for r in range(len(images)) if orders[r] == k - 1]
        if not cand_rows:
            return -1, -1, float("inf")
        cand_pos = np.asarray([images[r] for r in cand_rows], dtype=np.float64)
        best = (-1, -1, float("inf"))
        for w in range(engine.n_walls):
            parent_est = engine.reflect(pos, w)
            d = np.linalg.norm(cand_pos - parent_est[None, :], axis=1)
            j = int(np.argmin(d))
            if float(d[j]) < best[2]:
                best = (cand_rows[j], w, float(d[j]))
        return best

    # Search backwards for wall sequences that reach this position at order k.
    # Distance rules remove impossible parents, while known prefixes shorten
    # the search and blind_limit bounds runs of invisible ancestors.
    def find_sequences(pos: np.ndarray, k: int, want: int, blind_limit: int) -> List[Tuple[int, ...]]:
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        dist_tol = max(tol, 1e-9)

        def dfs(p: np.ndarray, kk: int, w_next: int, dist_p: float, blind: int) -> List[Tuple[int, ...]]:
            if kk == 0:
                return [tuple()] if float(np.linalg.norm(p - src_pos)) <= tol else []
            out: List[Tuple[int, ...]] = []
            # Reuse a wall sequence already registered at this position.
            for row in pos_registry.get(poskey(p, kk), []):
                sr = seq_of_row[row]
                if sr and sr[-1] != w_next:
                    out.append(sr)
            if len(out) >= want:
                return out[:want]
            if blind >= blind_limit:
                return out
            seen = set(out)
            for w in range(engine.n_walls):
                if w == w_next:
                    continue
                parent = engine.reflect(p, w)
                if not engine.on_inward_side(parent, w):  # Discard reflections behind the wall.
                    continue
                dist_parent = float(np.linalg.norm(parent - src_pos))
                if dist_parent > dist_p + dist_tol:
                    continue
                for sub in dfs(parent, kk - 1, w, dist_parent, blind + 1):
                    cand = sub + (w,)
                    if cand not in seen:
                        seen.add(cand)
                        out.append(cand)
                    if len(out) >= want:
                        return out[:want]
            return out

        return dfs(pos, int(k), -1, float(np.linalg.norm(pos - src_pos)), 0)

    # A position can represent several corner-bounce paths, so group nodes
    # before assigning their distinct wall sequences.
    groups: Dict[Tuple[int, int, int, int], List[int]] = {}
    for i in range(n_py):
        k = int(pyroom_orders[i])
        if k <= 0:
            visibility[0] = np.asarray(pyroom_visibility[:, i], dtype=np.uint8)
            matched[0] = 1
            continue
        groups.setdefault(poskey(np.asarray(pyroom_images[i], dtype=np.float64), k), []).append(i)

    n_recovered = 0
    n_glued = 0
    n_failed = 0
    n_multiplicity_short = 0
    n_damping_overridden = 0
    n_twin_dup = 0
    max_damp_err = 0.0
    max_glue_err = 0.0

    for pk, members in sorted(groups.items()):
        k = pk[0]
        i0 = members[0]
        pos = np.asarray(pyroom_images[i0], dtype=np.float64)
        # Extra candidates help match coincident paths by damping.
        seqs = find_sequences(pos, k, want=len(members) + 2, blind_limit=blind_limit)
        # Assign coincident wall sequences to pyroom nodes by damping agreement.
        if seqs:
            seq_damp = [float(np.prod(engine.wall_beta[list(s)])) for s in seqs]
            assigned: List[Tuple[int, ...]] = []
            used = [False] * len(seqs)
            for i in members:
                best_j, best_e = -1, float("inf")
                for j, sd in enumerate(seq_damp):
                    if not used[j] and abs(sd - float(pyroom_damping[i])) < best_e:
                        best_j, best_e = j, abs(sd - float(pyroom_damping[i]))
                if best_j >= 0:
                    used[best_j] = True
                    assigned.append(tuple(seqs[best_j]))
            seqs = assigned
        for j, i in enumerate(members):
            if j < len(seqs):
                row = materialize(tuple(seqs[j]))
                if matched[row]:
                    # Keep a separate row when a coincident path has already
                    # claimed this sequence; both paths contribute to the RIR.
                    pos_i = np.asarray(pyroom_images[i], dtype=np.float64)
                    images.append(pos_i.copy())
                    walls.append(int(walls[row]))
                    parents.append(int(parents[row]))
                    orders.append(int(k))
                    damping.append(float(pyroom_damping[i]))
                    visibility.append(np.asarray(pyroom_visibility[:, i], dtype=np.uint8))
                    matched.append(1)
                    glued.append(0)
                    # The primary row already represents this sequence and cell.
                    seq_of_row.append(seq_of_row[row])
                    n_twin_dup += 1
                    n_recovered += 1
                    continue
                visibility[row] = np.asarray(pyroom_visibility[:, i], dtype=np.uint8)
                matched[row] = 1
                n_recovered += 1
                d_err = abs(damping[row] - float(pyroom_damping[i]))
                max_damp_err = max(max_damp_err, d_err)
                if d_err > 1e-3:
                    # Keep pyroomacoustics' damping when the recovered chain
                    # does not distinguish a deeper coincident path.
                    damping[row] = float(pyroom_damping[i])
                    n_damping_overridden += 1
            else:
                # Use the nearest compatible parent if no exact chain was found.
                if j >= 1 and len(seqs) >= 1:
                    n_multiplicity_short += 1
                p_row, p_wall, err = glue_fallback(pos, k)
                if p_row < 0:
                    n_failed += 1
                    continue
                images.append(pos.copy())
                walls.append(int(p_wall))
                parents.append(int(p_row))
                orders.append(int(k))
                damping.append(float(pyroom_damping[i]))
                visibility.append(np.asarray(pyroom_visibility[:, i], dtype=np.uint8))
                matched.append(1)
                glued.append(1)
                seq = seq_of_row[p_row] + (int(p_wall),)
                seq_of_row.append(seq)
                # The approximate sequence does not reproduce the node
                # position, so it is not registered for later matching.
                n_glued += 1
                max_glue_err = max(max_glue_err, float(err))

    nodes = {
        "node_images": np.asarray(images, dtype=np.float32),
        "node_damping_1d": np.asarray(damping, dtype=np.float64),
        "node_parent": np.asarray(parents, dtype=np.int32),
        "node_wall": np.asarray(walls, dtype=np.int32),
        "node_order": np.asarray(orders, dtype=np.int16),
        "node_visibility": np.stack(visibility, axis=1).astype(np.uint8),
        "node_matched_pyroom": np.asarray(matched, dtype=np.uint8),
        "node_parent_glued": np.asarray(glued, dtype=np.uint8),
    }
    n_nodes = len(images)
    n_py_nonroot = int(np.sum(pyroom_orders > 0))
    stats = {
        "chain_recovered_nodes": float(n_recovered),
        "chain_glued_nodes": float(n_glued),
        "chain_failed_nodes": float(n_failed),
        "chain_multiplicity_short_nodes": float(n_multiplicity_short),
        "chain_recovered_frac": float(n_recovered / max(n_py_nonroot, 1)),
        "chain_glued_frac": float(n_glued / max(n_py_nonroot, 1)),
        "chain_connector_nodes": float(n_nodes - int(np.sum(nodes["node_matched_pyroom"]))),
        "chain_max_damping_error": float(max_damp_err),
        "chain_damping_overridden_nodes": float(n_damping_overridden),
        "chain_twin_duplicate_nodes": float(n_twin_dup),
        "chain_max_glue_error_m": float(max_glue_err),
        "parent_valid_nonroot_frac": 1.0,
        "parent_bad_order_nodes": 0.0,
    }
    return nodes, stats, seq_of_row, registry


def sample_dead_end_candidates(
    engine: ExtrudedPolygonRoom,
    nodes: Dict[str, np.ndarray],
    seq_of_row: List[Tuple[int, ...]],
    registry: Dict[Tuple[int, ...], int],
    max_order: int,
    num_mics: int,
    ratio: float,
    rng: np.random.Generator,
    tol: float = 1e-4,
) -> Dict[str, np.ndarray]:
    """Sample valid expansions that do not belong to the reconstructed tree.

    These zero-label nodes resemble the candidates seen during online pruning.
    Wall sequence and position are both checked so coincident paths are not
    mislabeled.
    """
    empty = {
        "node_images": np.zeros((0, 3), dtype=np.float32),
        "node_damping_1d": np.zeros(0, dtype=np.float64),
        "node_parent": np.zeros(0, dtype=np.int32),
        "node_wall": np.zeros(0, dtype=np.int32),
        "node_order": np.zeros(0, dtype=np.int16),
        "node_visibility": np.zeros((num_mics, 0), dtype=np.uint8),
        "node_matched_pyroom": np.zeros(0, dtype=np.uint8),
    }
    if ratio <= 0.0:
        return empty

    cand_img: List[np.ndarray] = []
    cand_wall: List[int] = []
    cand_parent: List[int] = []
    cand_order: List[int] = []
    cand_damp: List[float] = []

    imgs = nodes["node_images"].astype(np.float64)
    walls_arr = nodes["node_wall"]
    orders_arr = nodes["node_order"]
    damps = nodes["node_damping_1d"]
    glued_arr = nodes.get("node_parent_glued")

    # Approximate and coincident rows may be absent from the sequence registry.
    def cell(pos: np.ndarray, k: int) -> Tuple[int, int, int, int]:
        return (k,) + tuple(int(round(float(v) / (tol * 0.25))) for v in pos)

    pos_set = {cell(imgs[r], int(orders_arr[r])) for r in range(imgs.shape[0])}
    for r in range(imgs.shape[0]):
        k = int(orders_arr[r])
        if k >= max_order:
            continue
        if glued_arr is not None and int(glued_arr[r]) == 1:
            # Skip rows whose approximate chain cannot identify negative children.
            continue
        p = imgs[r]
        pw = int(walls_arr[r])
        seq_r = seq_of_row[r]
        for w in range(engine.n_walls):
            if w == pw:
                continue
            if not engine.on_inward_side(p, w):
                continue
            if seq_r + (w,) in registry:
                continue
            child = engine.reflect(p, w)
            if cell(child, k + 1) in pos_set:
                continue
            cand_img.append(child)
            cand_wall.append(w)
            cand_parent.append(r)
            cand_order.append(k + 1)
            cand_damp.append(float(damps[r]) * float(engine.wall_beta[w]))

    if not cand_img:
        return empty

    n_all = len(cand_img)
    n_keep = min(n_all, int(round(ratio * imgs.shape[0])))
    if n_keep < n_all:
        sel = np.sort(rng.choice(n_all, n_keep, replace=False))
    else:
        sel = np.arange(n_all)

    return {
        "node_images": np.asarray([cand_img[j] for j in sel], dtype=np.float32).reshape(-1, 3),
        "node_damping_1d": np.asarray([cand_damp[j] for j in sel], dtype=np.float64),
        "node_parent": np.asarray([cand_parent[j] for j in sel], dtype=np.int32),
        "node_wall": np.asarray([cand_wall[j] for j in sel], dtype=np.int32),
        "node_order": np.asarray([cand_order[j] for j in sel], dtype=np.int16),
        "node_visibility": np.zeros((num_mics, len(sel)), dtype=np.uint8),
        "node_matched_pyroom": np.zeros(len(sel), dtype=np.uint8),
    }


# RIR contribution helpers


def get_pra_constant(name: str, default: float) -> float:
    try:
        return float(pra.constants.get(name))
    except Exception:
        return float(default)


def pad_or_truncate(x: np.ndarray, n: int) -> np.ndarray:
    y = np.zeros(n, dtype=np.float32)
    k = min(n, int(x.shape[-1]))
    y[:k] = np.asarray(x[:k], dtype=np.float32)
    return y


def add_fractional_impulse_(
    buf: np.ndarray,
    delay_seconds: float,
    amplitude: float,
    fs: int,
    frac_delay_len: int,
) -> None:
    if not np.isfinite(delay_seconds) or not np.isfinite(amplitude):
        return
    if amplitude == 0.0:
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


def build_children(parents: np.ndarray) -> List[List[int]]:
    children: List[List[int]] = [[] for _ in range(len(parents))]
    for i, p in enumerate(parents):
        if 0 <= int(p) < len(parents):
            children[int(p)].append(i)
    return children


def compute_node_own_rirs(
    images_xyz: np.ndarray,
    damping: np.ndarray,
    mic_positions: np.ndarray,
    visibility: np.ndarray,
    fs: int,
    rir_len: int,
    c: float,
    frac_delay_len: int,
) -> np.ndarray:
    num_nodes = images_xyz.shape[0]
    num_mics = mic_positions.shape[1]
    own = np.zeros((num_nodes, num_mics, rir_len), dtype=np.float32)

    damping_1d = np.asarray(damping)
    if damping_1d.ndim == 2:
        damping_1d = damping_1d.mean(axis=0)
    damping_1d = damping_1d.astype(np.float64)

    for i in range(num_nodes):
        img = images_xyz[i].astype(np.float64)
        for m in range(num_mics):
            if not bool(visibility[m, i]):
                continue
            mic = mic_positions[:, m].astype(np.float64)
            dist = float(np.linalg.norm(img - mic))
            if dist <= 1e-8:
                continue
            # pyroomacoustics uses damping divided by propagation distance.
            amp = float(damping_1d[i]) / dist
            delay = dist / c
            add_fractional_impulse_(own[i, m], delay, amp, fs, frac_delay_len)

    return own


def compute_subtree_labels(
    parents: np.ndarray,
    own_rirs: np.ndarray,
    label_eps: float,
) -> Dict[str, np.ndarray]:
    sys.setrecursionlimit(max(10000, len(parents) + 100))

    num_nodes = len(parents)
    children = build_children(parents)
    roots = [i for i, p in enumerate(parents) if p < 0 or p >= num_nodes]
    if not roots:
        roots = [0]

    subtree_energy = np.zeros(num_nodes, dtype=np.float64)
    subtree_peak = np.zeros(num_nodes, dtype=np.float32)
    subtree_size = np.ones(num_nodes, dtype=np.int32)
    visited = np.zeros(num_nodes, dtype=bool)

    def dfs(i: int) -> np.ndarray:
        visited[i] = True
        acc = own_rirs[i].copy()
        size = 1
        for ch in children[i]:
            child_acc = dfs(ch)
            acc += child_acc
            size += int(subtree_size[ch])
        subtree_size[i] = size
        subtree_energy[i] = float(np.sum(acc.astype(np.float64) ** 2))
        subtree_peak[i] = float(np.max(np.abs(acc))) if acc.size else 0.0
        return acc

    full_reconstructed = np.zeros_like(own_rirs[0])
    for r in roots:
        full_reconstructed += dfs(r)
    for i in range(num_nodes):
        if not visited[i]:
            full_reconstructed += dfs(i)

    full_energy = float(np.sum(full_reconstructed.astype(np.float64) ** 2))
    denom = full_energy + 1e-20
    energy_ratio = subtree_energy / denom
    l2_ratio = np.sqrt(energy_ratio)
    keep = (energy_ratio >= float(label_eps)).astype(np.uint8)

    return {
        "label_subtree_energy": subtree_energy.astype(np.float32),
        "label_subtree_energy_ratio": energy_ratio.astype(np.float32),
        "label_subtree_l2_ratio": l2_ratio.astype(np.float32),
        "label_subtree_peak": subtree_peak.astype(np.float32),
        "label_keep": keep,
        "label_full_reconstructed_energy": np.array(full_energy, dtype=np.float32),
        "subtree_size": subtree_size,
        "reconstructed_full_rirs": full_reconstructed.astype(np.float32),
    }


# pyroomacoustics room construction and node extraction


def build_room(
    rng: np.random.Generator, cfg: DatasetConfig
) -> Tuple[object, Dict[str, np.ndarray | float]]:
    poly_xy = random_star_polygon(rng, cfg)
    height = float(rng.uniform(cfg.height_min, cfg.height_max))
    n_side_walls = poly_xy.shape[0]

    side_abs = rng.uniform(cfg.side_abs_min, cfg.side_abs_max, size=n_side_walls).astype(np.float32)
    floor_abs = float(rng.uniform(cfg.floor_abs_min, cfg.floor_abs_max))
    ceiling_abs = float(rng.uniform(cfg.ceiling_abs_min, cfg.ceiling_abs_max))
    wall_abs = np.concatenate([side_abs, [floor_abs, ceiling_abs]]).astype(np.float32)

    side_materials = [pra.Material(energy_absorption=float(a), scattering=0.0) for a in side_abs]
    floor_ceiling_materials = {
        "floor": pra.Material(energy_absorption=floor_abs, scattering=0.0),
        "ceiling": pra.Material(energy_absorption=ceiling_abs, scattering=0.0),
    }

    try:
        room = pra.Room.from_corners(
            poly_xy.T,
            fs=cfg.fs,
            max_order=cfg.max_order,
            materials=side_materials,
        )
        room.extrude(height, materials=floor_ceiling_materials)
    except TypeError:
        room = pra.Room.from_corners(
            poly_xy.T,
            fs=cfg.fs,
            max_order=cfg.max_order,
            absorption=side_abs,
        )
        room.extrude(height, absorption=wall_abs)

    mic_positions = sample_microphone_array(
        rng,
        poly_xy,
        height,
        num_mics=cfg.num_mics,
        spacing=cfg.mic_spacing,
        margin=cfg.min_wall_margin,
    )

    for _ in range(2000):
        source_position = sample_point_in_room(rng, poly_xy, height, margin=cfg.min_wall_margin)
        dists = np.linalg.norm(mic_positions.T - source_position[None, :], axis=1)
        if float(dists.min()) >= cfg.min_src_mic_dist:
            break
    else:
        raise RuntimeError("Failed to sample source sufficiently far from microphones.")

    room.add_source(source_position)
    room.add_microphone_array(mic_positions)

    meta = {
        "corners_xy": poly_xy,
        "height": np.array(height, dtype=np.float32),
        "wall_absorption": wall_abs,
        "source_position": source_position.astype(np.float32),
        "mic_positions": mic_positions.astype(np.float32),
        "floor_area": np.array(polygon_area(poly_xy), dtype=np.float32),
        "volume": np.array(polygon_area(poly_xy) * height, dtype=np.float32),
    }
    return room, meta


def safe_visibility(room: object, src_idx: int, num_mics: int, num_nodes: int) -> np.ndarray:
    vis = np.ones((num_mics, num_nodes), dtype=bool)
    raw = getattr(room, "visibility", None)
    if raw is None:
        return vis
    for m in range(num_mics):
        try:
            v = np.asarray(raw[src_idx][m]).astype(bool).reshape(-1)
            if v.shape[0] == num_nodes:
                vis[m] = v
        except Exception:
            pass
    return vis


def extract_source_nodes(
    room: object,
    meta: Dict[str, "np.ndarray | float"],
    cfg: DatasetConfig,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Build the training node set from a pyroomacoustics room.

    The result contains the reconstructed tree, visibility, match flags, and
    sampled negative candidates, all using the polygon engine's wall indices.
    """
    src = room.sources[0]
    images = np.asarray(src.images, dtype=np.float64).T
    damping = np.asarray(src.damping, dtype=np.float64)
    if damping.ndim == 2:
        damping = damping[0]
    orders = np.asarray(src.orders, dtype=np.int64).reshape(-1)
    if not (images.shape[0] == damping.shape[0] == orders.shape[0]):
        raise RuntimeError("Inconsistent SoundSource node arrays from pyroomacoustics.")

    num_mics = int(np.asarray(meta["mic_positions"]).shape[1])
    py_vis = safe_visibility(room, src_idx=0, num_mics=num_mics, num_nodes=images.shape[0])

    engine = ExtrudedPolygonRoom(
        np.asarray(meta["corners_xy"], dtype=np.float64),
        float(np.asarray(meta["height"])),
        np.asarray(meta["wall_absorption"], dtype=np.float64),
    )
    src_pos = np.asarray(meta["source_position"], dtype=np.float64).reshape(3)

    tree, stats, seq_of_row, registry = build_true_tree(
        engine, src_pos, images, damping, orders, py_vis,
        num_mics=num_mics, tol=cfg.chain_match_tol,
        blind_limit=cfg.chain_blind_limit,
    )
    dead = sample_dead_end_candidates(
        engine, tree, seq_of_row, registry, max_order=cfg.max_order,
        num_mics=num_mics, ratio=cfg.dead_negatives_ratio, rng=rng,
        tol=cfg.chain_match_tol,
    )
    n_tree = tree["node_images"].shape[0]
    n_dead = dead["node_images"].shape[0]

    nodes = {
        "node_images": np.concatenate([tree["node_images"], dead["node_images"]], axis=0).astype(np.float32),
        "node_damping": np.concatenate([tree["node_damping_1d"], dead["node_damping_1d"]])[None, :].astype(np.float32),
        "node_parent": np.concatenate([tree["node_parent"], dead["node_parent"]]).astype(np.int32),
        "node_wall": np.concatenate([tree["node_wall"], dead["node_wall"]]).astype(np.int32),
        "node_order": np.concatenate([tree["node_order"], dead["node_order"]]).astype(np.int16),
        "node_visibility": np.concatenate([tree["node_visibility"], dead["node_visibility"]], axis=1).astype(np.uint8),
        "node_matched_pyroom": np.concatenate([tree["node_matched_pyroom"], dead["node_matched_pyroom"]]).astype(np.uint8),
        "node_parent_glued": np.concatenate([
            tree["node_parent_glued"], np.zeros(n_dead, dtype=np.uint8)
        ]).astype(np.uint8),
        "node_is_dead_candidate": np.concatenate([
            np.zeros(n_tree, dtype=np.uint8), np.ones(n_dead, dtype=np.uint8)
        ]),
    }
    stats["dead_candidate_nodes"] = float(n_dead)
    stats["true_tree_nodes"] = float(n_tree)
    stats["pyroom_visible_nodes"] = float(images.shape[0])
    return nodes, stats


def make_node_features(
    nodes: Dict[str, np.ndarray],
    meta: Dict[str, np.ndarray | float],
    visibility: np.ndarray,
    c: float,
) -> np.ndarray:
    images = nodes["node_images"]
    damping = nodes["node_damping"]
    parents = nodes["node_parent"].astype(np.int64)
    walls = nodes["node_wall"].astype(np.int64)
    orders = nodes["node_order"].astype(np.float32)
    num_nodes = images.shape[0]

    mic_positions = np.asarray(meta["mic_positions"], dtype=np.float32)
    source_position = np.asarray(meta["source_position"], dtype=np.float32)
    wall_abs = np.asarray(meta["wall_absorption"], dtype=np.float32)

    d = np.linalg.norm(images[:, None, :] - mic_positions.T[None, :, :], axis=2)
    delays = d / float(c)
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
    room_num_walls = float(len(wall_abs))
    floor_area = float(np.asarray(meta["floor_area"]))
    height = float(np.asarray(meta["height"]))
    volume = float(np.asarray(meta["volume"]))

    feats = np.column_stack(
        [
            orders,
            walls.astype(np.float32),
            parent_order,
            node_wall_abs,
            images[:, 0],
            images[:, 1],
            images[:, 2],
            damping_mean,
            damping_min,
            damping_max,
            d.min(axis=1),
            d.mean(axis=1),
            d.max(axis=1),
            delays.min(axis=1),
            delays.mean(axis=1),
            delays.max(axis=1),
            visible_count,
            visible_frac,
            np.full(num_nodes, room_num_walls, dtype=np.float32),
            np.full(num_nodes, floor_area, dtype=np.float32),
            np.full(num_nodes, height, dtype=np.float32),
            np.full(num_nodes, volume, dtype=np.float32),
            np.full(num_nodes, float(wall_abs.mean()), dtype=np.float32),
            np.full(num_nodes, float(wall_abs.std()), dtype=np.float32),
            np.full(num_nodes, float(source_position[0]), dtype=np.float32),
            np.full(num_nodes, float(source_position[1]), dtype=np.float32),
            np.full(num_nodes, float(source_position[2]), dtype=np.float32),
            np.full(num_nodes, float(mic_centroid[0]), dtype=np.float32),
            np.full(num_nodes, float(mic_centroid[1]), dtype=np.float32),
            np.full(num_nodes, float(mic_centroid[2]), dtype=np.float32),
        ]
    )
    return feats.astype(np.float32)


# Room worker


def pad_2d(x: np.ndarray, n: int) -> np.ndarray:
    y = np.zeros((x.shape[0], n), dtype=np.float32)
    k = min(x.shape[1], n)
    y[:, :k] = x[:, :k]
    return y


def process_one(index: int, cfg: DatasetConfig) -> Dict[str, object]:
    start_time = time.time()
    out_dir = Path(cfg.out_dir)
    out_path = out_dir / f"room_{index:07d}.npz"
    if out_path.exists():
        return {
            "index": index,
            "status": "skipped_exists",
            "path": str(out_path),
            "elapsed_sec": 0.0,
        }

    rng = np.random.default_rng(cfg.seed + 1000003 * index)

    try:
        try:
            pra.constants.set("rir_hpf_enable", bool(cfg.rir_hpf_enable))
        except Exception:
            pass
        try:
            pra.random.seed(int(cfg.seed + index))
        except Exception:
            pass

        room, meta = build_room(rng, cfg)
        room.compute_rir()

        nodes, parent_stats = extract_source_nodes(room, meta, cfg, rng)
        num_nodes = int(nodes["node_images"].shape[0])
        num_mics = int(meta["mic_positions"].shape[1])
        rir_len = int(round(cfg.rir_duration * cfg.fs))

        full_rirs = np.stack(
            [pad_or_truncate(room.rir[m][0], rir_len) for m in range(num_mics)], axis=0
        )
        visibility = nodes["node_visibility"].astype(bool)

        c = get_pra_constant("c", 343.0)
        frac_delay_len = int(get_pra_constant("frac_delay_length", 81))

        own_rirs = compute_node_own_rirs(
            images_xyz=nodes["node_images"],
            damping=nodes["node_damping"],
            mic_positions=np.asarray(meta["mic_positions"], dtype=np.float32),
            visibility=visibility,
            fs=cfg.fs,
            rir_len=rir_len,
            c=c,
            frac_delay_len=frac_delay_len,
        )
        labels = compute_subtree_labels(
            parents=nodes["node_parent"],
            own_rirs=own_rirs,
            label_eps=cfg.label_eps,
        )

        features = make_node_features(nodes, meta, visibility, c=c)

        full_energy = np.array(np.sum(full_rirs.astype(np.float64) ** 2), dtype=np.float32)
        recon = labels.pop("reconstructed_full_rirs")
        recon_energy = labels["label_full_reconstructed_energy"]
        recon_mse = np.array(np.mean((pad_2d(full_rirs, recon.shape[-1]) - recon) ** 2), dtype=np.float32)

        metadata = {
            "index": index,
            "seed": cfg.seed + 1000003 * index,
            "fs": cfg.fs,
            "max_order": cfg.max_order,
            "rir_duration": cfg.rir_duration,
            "num_mics": num_mics,
            "num_nodes": num_nodes,
            "num_kept_at_label_eps": int(labels["label_keep"].sum()),
            "label_eps": cfg.label_eps,
            "c": c,
            "frac_delay_len": frac_delay_len,
            "rir_hpf_enable": bool(cfg.rir_hpf_enable),
            "full_rir_energy_pyroomacoustics": float(full_energy),
            "full_rir_energy_reconstructed": float(recon_energy),
            "full_rir_reconstruction_mse": float(recon_mse),
            **parent_stats,
        }

        arrays = {
            **{k: np.asarray(v) for k, v in meta.items()},
            **nodes,
            **labels,
            "node_visibility": visibility.astype(np.uint8),
            "node_features": features,
            "feature_names": FEATURE_NAMES,
            "full_rirs": full_rirs.astype(np.float32),
            "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
        }

        if cfg.compress:
            np.savez_compressed(out_path, **arrays)
        else:
            np.savez(out_path, **arrays)

        elapsed = time.time() - start_time
        return {
            "index": index,
            "status": "ok",
            "path": str(out_path),
            "elapsed_sec": elapsed,
            "num_nodes": num_nodes,
            "num_kept": int(labels["label_keep"].sum()),
            "full_rir_reconstruction_mse": float(recon_mse),
            **parent_stats,
        }

    except Exception as exc:
        return {
            "index": index,
            "status": "error",
            "path": str(out_path),
            "elapsed_sec": time.time() - start_time,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


# Progress reporting


class TerminalProgress:
    def __init__(self, total: int, desc: str = "rooms", every: int = 1):
        self.total = max(1, int(total))
        self.desc = desc
        self.every = max(1, int(every))
        self.n = 0
        self.ok = 0
        self.skipped = 0
        self.errors = 0
        self.total_nodes = 0
        self.start = time.time()
        self._pbar = tqdm(total=self.total, desc=self.desc, dynamic_ncols=True) if tqdm is not None else None
        if self._pbar is None:
            self._render()

    def update(self, rec: Dict[str, object]) -> None:
        self.n += 1
        status = rec.get("status")
        self.ok += int(status == "ok")
        self.skipped += int(status == "skipped_exists")
        self.errors += int(status == "error")
        self.total_nodes += int(rec.get("num_nodes", 0) or 0)
        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix(ok=self.ok, err=self.errors, skip=self.skipped, nodes=self.total_nodes, refresh=False)
        elif self.n % self.every == 0 or self.n >= self.total:
            self._render()

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
        else:
            self._render()
            print(file=sys.stderr, flush=True)

    def _render(self) -> None:
        elapsed = max(time.time() - self.start, 1e-9)
        rate = self.n / elapsed
        frac = min(1.0, self.n / self.total)
        width = 32
        filled = int(round(width * frac))
        bar = "#" * filled + "-" * (width - filled)
        msg = (
            f"\r{self.desc}: |{bar}| {self.n}/{self.total} "
            f"({100.0 * frac:5.1f}%) ok={self.ok} err={self.errors} "
            f"skip={self.skipped} nodes={self.total_nodes} rate={rate:.2f}/s elapsed={elapsed:.1f}s"
        )
        print(msg, end="", file=sys.stderr, flush=True)


# Command-line interface


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate PathRIR datasets for extruded polygon rooms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--num-configs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fs", type=int, default=8000)
    p.add_argument("--max-order", type=int, default=15)
    p.add_argument("--rir-duration", type=float, default=0.5)
    p.add_argument("--num-mics", type=int, default=2)
    p.add_argument("--mic-spacing", type=float, default=0.06)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--label-eps", type=float, default=1e-4)
    p.add_argument("--min-src-mic-dist", type=float, default=0.75)
    p.add_argument("--min-wall-margin", type=float, default=0.15)
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--rir-hpf-enable", action="store_true", help="Enable pyroomacoustics' RIR high-pass filter.")
    p.add_argument("--chain-match-tol", type=float, default=1e-4,
                   help="Position tolerance in metres for matching image-source chains.")
    p.add_argument("--chain-blind-limit", type=int, default=4,
                   help="Maximum run of missing ancestors allowed during chain search.")
    p.add_argument("--dead-negatives-ratio", type=float, default=1.0,
                   help="Negative candidates sampled per reconstructed tree node; set 0 to skip them.")

    p.add_argument("--min-vertices", type=int, default=5)
    p.add_argument("--max-vertices", type=int, default=10)
    p.add_argument("--width-min", type=float, default=3.0)
    p.add_argument("--width-max", type=float, default=12.0)
    p.add_argument("--length-min", type=float, default=3.0)
    p.add_argument("--length-max", type=float, default=12.0)
    p.add_argument("--height-min", type=float, default=2.2)
    p.add_argument("--height-max", type=float, default=4.5)
    p.add_argument("--radial-jitter", type=float, default=0.35)

    p.add_argument("--side-abs-min", type=float, default=0.04)
    p.add_argument("--side-abs-max", type=float, default=0.55)
    p.add_argument("--floor-abs-min", type=float, default=0.05)
    p.add_argument("--floor-abs-max", type=float, default=0.70)
    p.add_argument("--ceiling-abs-min", type=float, default=0.03)
    p.add_argument("--ceiling-abs-max", type=float, default=0.45)
    return p.parse_args()


def args_to_config(args: argparse.Namespace) -> DatasetConfig:
    return DatasetConfig(
        out_dir=args.out_dir,
        num_configs=args.num_configs,
        seed=args.seed,
        fs=args.fs,
        max_order=args.max_order,
        rir_duration=args.rir_duration,
        num_mics=args.num_mics,
        mic_spacing=args.mic_spacing,
        label_eps=args.label_eps,
        min_src_mic_dist=args.min_src_mic_dist,
        min_wall_margin=args.min_wall_margin,
        compress=not args.no_compress,
        rir_hpf_enable=args.rir_hpf_enable,
        chain_match_tol=args.chain_match_tol,
        chain_blind_limit=args.chain_blind_limit,
        dead_negatives_ratio=args.dead_negatives_ratio,
        min_vertices=args.min_vertices,
        max_vertices=args.max_vertices,
        width_min=args.width_min,
        width_max=args.width_max,
        length_min=args.length_min,
        length_max=args.length_max,
        height_min=args.height_min,
        height_max=args.height_max,
        radial_jitter=args.radial_jitter,
        side_abs_min=args.side_abs_min,
        side_abs_max=args.side_abs_max,
        floor_abs_min=args.floor_abs_min,
        floor_abs_max=args.floor_abs_max,
        ceiling_abs_min=args.ceiling_abs_min,
        ceiling_abs_max=args.ceiling_abs_max,
    )


def append_jsonl(path: Path, record: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    cfg = args_to_config(args)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    config_path = out_dir / "dataset_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8")

    indices = list(range(cfg.num_configs))
    progress = TerminalProgress(total=len(indices), desc="rooms", every=args.progress_every)

    try:
        if args.num_workers <= 1:
            for idx in indices:
                rec = process_one(idx, cfg)
                append_jsonl(manifest_path, rec)
                progress.update(rec)
                if rec.get("status") == "error":
                    print(file=sys.stderr, flush=True)
                    print(f"[error] index={idx}: {rec.get('error')}", file=sys.stderr, flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
                futures = {ex.submit(process_one, idx, cfg): idx for idx in indices}
                for fut in as_completed(futures):
                    rec = fut.result()
                    append_jsonl(manifest_path, rec)
                    progress.update(rec)
                    if rec.get("status") == "error":
                        print(file=sys.stderr, flush=True)
                        print(f"[error] index={rec.get('index')}: {rec.get('error')}", file=sys.stderr, flush=True)
    finally:
        progress.close()

    summary = {"ok": progress.ok, "skipped": progress.skipped, "errors": progress.errors, "out_dir": str(out_dir)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
