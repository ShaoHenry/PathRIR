"""Python interface for generating room impulse responses with PathRIR.

Example:
    from pathrir import PathRIR

    simulator = PathRIR()
    rir = simulator.simulate(
        corners=[[0, 0], [5, 0], [6, 3], [3, 5], [0, 4]],
        height=3.0,
        absorption=0.3,
        source=[2.0, 2.0, 1.5],
        mics=[[3.0, 3.0, 1.2], [1.5, 3.2, 1.6]],
        fs=8000,
        duration=0.5,
        max_order=10,
    )
    # rir has shape (number of microphones, round(fs * duration))
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

import evaluate_pathrir as _backend

__all__ = ["PathRIR", "simulate"]

_PKG_CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_PRUNING_CKPT = _PKG_CKPT_DIR / "pruning_mlp.pt"
DEFAULT_COMPENSATION_CKPT = _PKG_CKPT_DIR / "compensation_mlp.pt"


def _ensure_ccw(corners: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Return counter-clockwise corners and whether their order was reversed."""
    if corners.shape[0] < 3:
        raise ValueError("corners must contain at least three vertices")
    x, y = corners[:, 0], corners[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if abs(area) <= 1e-12:
        raise ValueError("corners must define a polygon with non-zero area")
    if area > 0:
        return corners, False
    return corners[::-1].copy(), True


def _normalize_absorption(
    absorption,
    num_walls: int,
    reverse_wall_order: bool = False,
) -> np.ndarray:
    """Return one absorption value per wall, followed by floor and ceiling."""
    if np.isscalar(absorption):
        values = np.full(num_walls + 2, float(absorption), dtype=np.float64)
    else:
        arr = np.asarray(absorption, dtype=np.float64).reshape(-1)
        if arr.size == num_walls:
            mean = float(arr.mean())
            values = np.concatenate([arr, [mean, mean]])
        elif arr.size == num_walls + 2:
            values = arr.copy()
        else:
            raise ValueError(
                f"absorption must be a scalar, length-{num_walls} (walls only), "
                f"or length-{num_walls + 2} (walls, floor, ceiling); got {arr.size} values"
            )

    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("absorption values must be finite and lie between 0 and 1")

    if reverse_wall_order and num_walls > 0:
        values[:num_walls] = np.roll(values[:num_walls][::-1], -1)
    return values


class PathRIR:
    """Generate RIRs with ISM path pruning and late-tail compensation.

    The bundled order-10 checkpoints are loaded when no checkpoint paths are
    provided.
    """

    def __init__(
        self,
        pruning_ckpt: Union[str, Path, None] = None,
        compensation_ckpt: Union[str, Path, None] = None,
        device: str = "auto",
    ) -> None:
        self.device = _backend.resolve_device(device)
        p_path = Path(pruning_ckpt) if pruning_ckpt else DEFAULT_PRUNING_CKPT
        c_path = Path(compensation_ckpt) if compensation_ckpt else DEFAULT_COMPENSATION_CKPT
        self.pruning_model, self.pruning_aux = _backend.load_pruning_model(p_path, self.device)
        self.comp_model, self.comp_aux = _backend.load_compensation_model(c_path, self.device)

    def simulate(
        self,
        corners: Sequence[Sequence[float]],
        height: float,
        absorption,
        source: Sequence[float],
        mics: Sequence[Sequence[float]],
        fs: int = 8000,
        duration: float = 0.5,
        max_order: int = 10,
        compensate: bool = True,
        tail_mode: str = "absolute",
        sound_speed: float = 343.0,
        seed: int = 0,
        return_pruned: bool = False,
    ):
        """Generate one RIR per microphone in an extruded polygon room.

        Args:
            corners: Polygon floor plan in metres with shape ``(W, 2)``.
                Clockwise and counter-clockwise vertex orders are accepted.
            height: Room height in metres.
            absorption: A scalar, one value per polygon wall, or one value per
                wall followed by floor and ceiling values. Wall values follow
                the edge order in ``corners``.
            source: Source position in metres with shape ``(3,)``.
            mics: Microphone positions in metres with shape ``(M, 3)``.
            fs: Sampling rate in Hz.
            duration: RIR length in seconds.
            max_order: Maximum reflection order.
            compensate: Add the stochastic late tail predicted by
                the Compensation-MLP.
            tail_mode: ``"absolute"`` rectifies the tail; ``"signed"``
                preserves the sign of the noise.
            sound_speed: Speed of sound in metres per second.
            seed: Random seed for the stochastic tail.
            return_pruned: Return the pruning-only RIR as well.

        Returns:
            A ``float32`` array with shape ``(M, round(fs * duration))``. If
            ``return_pruned`` is true, the result is
            ``(compensated_rir, pruned_rir)``.
        """
        if height <= 0:
            raise ValueError("height must be positive")
        if fs <= 0:
            raise ValueError("fs must be positive")
        if duration <= 0:
            raise ValueError("duration must be positive")
        if max_order < 0:
            raise ValueError("max_order must be non-negative")
        if sound_speed <= 0:
            raise ValueError("sound_speed must be positive")

        corners_input = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        corners_arr, corners_reversed = _ensure_ccw(corners_input)
        mics_arr = np.asarray(mics, dtype=np.float32)
        if mics_arr.ndim == 1:
            mics_arr = mics_arr.reshape(1, 3)
        if mics_arr.shape[1] != 3:
            raise ValueError(f"mics must be (M, 3); got {mics_arr.shape}")
        mics_t = mics_arr.T.astype(np.float32)  # The backend stores microphones as columns.
        rir_len = int(round(fs * duration))

        scene = {
            "path": "<api>",
            "metadata": {
                "fs": int(fs),
                "rir_duration": float(duration),
                "max_order": int(max_order),
                "c": float(sound_speed),
                "frac_delay_len": 81,
                "rir_hpf_enable": False,
                "num_mics": int(mics_t.shape[1]),
            },
            "corners_xy": corners_arr,
            "height": float(height),
            "wall_absorption": _normalize_absorption(
                absorption,
                corners_arr.shape[0],
                reverse_wall_order=corners_reversed,
            ),
            "source_position": np.asarray(source, dtype=np.float32).reshape(3),
            "mic_positions": mics_t,
            "node_order": np.asarray([int(max_order)], dtype=np.int64),
            "full_rirs": np.zeros((mics_t.shape[1], rir_len), dtype=np.float32),
        }
        cfg = _backend.EvalConfig(
            data_dir="", ckpt="", out_dir="",
            comp_seed=int(seed),
            comp_tail_mode=str(tail_mode),
        )

        pruned, pruned_stats, data, _prob_all, imp_all = _backend.run_pruned_online_ism(
            scene, self.pruning_model, self.pruning_aux, self.device, cfg
        )
        if not compensate:
            return (pruned, pruned) if return_pruned else pruned

        x = _backend.build_compensation_features(data, pruned, pruned_stats, imp_all, self.comp_aux)
        pred_energy_bins, _ = _backend.predict_compensation_energy_bins(
            self.comp_model, self.comp_aux, x, self.device
        )
        tail = _backend.generate_stochastic_tail(
            pred_energy_bins, int(fs), pruned.shape[1], cfg.comp_gain, cfg.comp_seed, cfg.comp_start_ms, cfg.comp_tail_mode
        )
        compensated = (pruned + tail).astype(np.float32)
        return (compensated, pruned) if return_pruned else compensated


_default_simulator: Optional[PathRIR] = None


def simulate(*args, **kwargs):
    """Call :meth:`PathRIR.simulate` with a shared simulator created on first use."""
    global _default_simulator
    if _default_simulator is None:
        _default_simulator = PathRIR()
    return _default_simulator.simulate(*args, **kwargs)
