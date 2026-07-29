"""Incremental image-source expansion for extruded polygon rooms.

Polygon edges are numbered ``0`` through ``W - 1``. The floor is wall ``W``
and the ceiling is wall ``W + 1``. Floor-plan vertices must be ordered
counter-clockwise.

Expansion skips the wall that produced a parent and rejects reflections from
the back of a wall. Visibility is checked by tracing each specular path back
from a microphone, including finite wall extents and occlusion in non-convex
rooms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np


# Room


def signed_polygon_area(poly_xy: np.ndarray) -> float:
    x = poly_xy[:, 0]
    y = poly_xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


class ExtrudedPolygonRoom:
    """Represent an extruded polygon room with one absorption value per surface."""

    def __init__(
        self,
        corners_xy: np.ndarray,
        height: float,
        wall_absorption: Optional[np.ndarray] = None,
    ) -> None:
        corners_xy = np.asarray(corners_xy, dtype=np.float64)
        if corners_xy.ndim != 2 or corners_xy.shape[1] != 2:
            raise ValueError("corners_xy must have shape (W, 2)")
        if signed_polygon_area(corners_xy) <= 0.0:
            raise ValueError(
                "Polygon vertices must be listed counter-clockwise because wall "
                "indices follow the vertex order."
            )
        self.polygon = corners_xy
        self.height = float(height)
        self.W = int(corners_xy.shape[0])
        self.n_walls = self.W + 2  # Polygon walls, floor, and ceiling.

        if wall_absorption is None:
            alpha = np.zeros(self.n_walls, dtype=np.float64)
        else:
            alpha = np.asarray(wall_absorption, dtype=np.float64).reshape(-1)
            if alpha.shape[0] != self.n_walls:
                raise ValueError(
                    f"wall_absorption must have {self.n_walls} entries "
                    f"(W={self.W} sides + floor + ceiling), got {alpha.shape[0]}"
                )
        self.wall_absorption = alpha
        # Match pyroomacoustics' amplitude-reflection convention.
        self.wall_beta = np.sqrt(np.clip(1.0 - alpha, 0.0, 1.0))

        # Store each wall plane as an anchor point and inward unit normal.
        anchors = np.zeros((self.n_walls, 3), dtype=np.float64)
        normals = np.zeros((self.n_walls, 3), dtype=np.float64)
        for w in range(self.W):
            a, b = self.polygon[w], self.polygon[(w + 1) % self.W]
            n = np.array([-(b[1] - a[1]), b[0] - a[0], 0.0])
            normals[w] = n / np.linalg.norm(n)
            anchors[w] = np.array([a[0], a[1], 0.0])
        normals[self.W] = np.array([0.0, 0.0, 1.0])       # Floor normal.
        anchors[self.W] = np.array([0.0, 0.0, 0.0])
        normals[self.W + 1] = np.array([0.0, 0.0, -1.0])  # Ceiling normal.
        anchors[self.W + 1] = np.array([0.0, 0.0, self.height])
        self.wall_anchor = anchors
        self.wall_normal = normals

    # Wall geometry

    def reflect(self, point: np.ndarray, w: int) -> np.ndarray:
        p = self.wall_anchor[w]
        n = self.wall_normal[w]
        d = float((point - p) @ n)
        return point - 2.0 * d * n

    def on_inward_side(self, point: np.ndarray, w: int) -> bool:
        """Return whether a point lies strictly on the inside of a wall."""
        return float((point - self.wall_anchor[w]) @ self.wall_normal[w]) > 0.0

    def _point_in_polygon(self, pt2: np.ndarray, tol: float = 1e-9) -> bool:
        x, y = float(pt2[0]), float(pt2[1])
        poly = self.polygon
        W = self.W
        tol_sq = tol * tol
        for i in range(W):
            a = poly[i]
            b = poly[(i + 1) % W]
            dx, dy = b[0] - a[0], b[1] - a[1]
            L_sq = dx * dx + dy * dy
            if L_sq < 1e-24:
                continue
            t = ((x - a[0]) * dx + (y - a[1]) * dy) / L_sq
            t_c = max(0.0, min(1.0, t))
            cx = a[0] + t_c * dx
            cy = a[1] + t_c * dy
            if (x - cx) ** 2 + (y - cy) ** 2 < tol_sq:
                return True
        inside = False
        j = W - 1
        for i in range(W):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside

    def on_wall_extent(self, R: np.ndarray, w: int, tol: float = 1e-7) -> bool:
        """Return whether a 3-D point lies within a wall's finite extent."""
        H = self.height
        if 0 <= w < self.W:
            if not (-tol <= R[2] <= H + tol):
                return False
            a, b = self.polygon[w], self.polygon[(w + 1) % self.W]
            edge = b - a
            d2 = float(edge @ edge)
            if d2 < 1e-12:
                return False
            t = float((R[:2] - a) @ edge) / d2
            return -tol <= t <= 1.0 + tol
        if w == self.W:
            return abs(R[2]) <= tol and self._point_in_polygon(R[:2])
        if w == self.W + 1:
            return abs(R[2] - H) <= tol and self._point_in_polygon(R[:2])
        return False

    # Image-source expansion

    def expand_one_order(
        self,
        parent_images: np.ndarray,
        parent_gen_walls: np.ndarray,
        parent_damping: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate one reflection order from a set of parent nodes.

        Parameters
        ----------
        parent_images
            Parent image positions with shape ``(P, 3)``.
        parent_gen_walls
            Wall that generated each parent, or ``-1`` for the real source.
        parent_damping
            Cumulative amplitude-reflection product for each parent.

        Returns
        -------
        images, gen_walls, parent_loc, damping
            Child positions, generating walls, indices into the parent arrays,
            and cumulative damping values.

        Expansion does not filter by visibility because an image source hidden
        from every microphone may still have visible descendants.
        """
        parent_images = np.asarray(parent_images, dtype=np.float64).reshape(-1, 3)
        parent_gen_walls = np.asarray(parent_gen_walls, dtype=np.int64).reshape(-1)
        parent_damping = np.asarray(parent_damping, dtype=np.float64).reshape(-1)
        P = parent_images.shape[0]

        images, gen_walls, parent_loc, damping = [], [], [], []
        for k in range(P):
            p = parent_images[k]
            pw = int(parent_gen_walls[k])
            for w in range(self.n_walls):
                if w == pw:
                    continue
                if not self.on_inward_side(p, w):
                    continue
                images.append(self.reflect(p, w))
                gen_walls.append(w)
                parent_loc.append(k)
                damping.append(parent_damping[k] * self.wall_beta[w])

        if not images:
            return (np.zeros((0, 3)), np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64), np.zeros(0))
        return (np.asarray(images), np.asarray(gen_walls, dtype=np.int64),
                np.asarray(parent_loc, dtype=np.int64), np.asarray(damping))

    # Full tree

    def build_full_tree(self, src_pos: np.ndarray, max_order: int) -> "ISMTree":
        """Build a tree by expanding every frontier through ``max_order``."""
        tree = ISMTree.from_source(src_pos)
        frontier = np.array([0], dtype=np.int64)
        for _ in range(1, max_order + 1):
            frontier = tree.expand_frontier(self, frontier)
            if frontier.size == 0:
                break
        return tree

    # Specular visibility

    def _segment_crosses_wall_extent(self, img, target, w, tol=1e-7):
        direction = target - img
        den = float(direction @ self.wall_normal[w])
        if abs(den) < 1e-12:
            return False, None
        t = float((self.wall_anchor[w] - img) @ self.wall_normal[w]) / den
        if not (-tol <= t <= 1.0 + tol):
            return False, None
        R = img + t * direction
        if not self.on_wall_extent(R, w, tol):
            return False, None
        return True, R

    def _leg_occluded(self, p, q, excluded, eps=1e-9):
        direction = q - p
        for w in range(self.n_walls):
            if w in excluded:
                continue
            den = float(direction @ self.wall_normal[w])
            if abs(den) < 1e-12:
                continue
            t = float((self.wall_anchor[w] - p) @ self.wall_normal[w]) / den
            if not (eps < t < 1.0 - eps):
                continue
            R = p + t * direction
            if self.on_wall_extent(R, w, tol=1e-7):
                return True
        return False

    def visibility(
        self,
        mic_pos: np.ndarray,
        img_pos: np.ndarray,
        gen_wall: np.ndarray,
        parent_idx: np.ndarray,
        indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Check which image sources are visible from one microphone.

        The node arrays describe the full tree, with ``parent_idx == -1`` at
        the source. If ``indices`` is given, only those rows are returned, but
        their complete ancestor chains are still traced.
        """
        img_pos = np.asarray(img_pos, dtype=np.float64)
        mic_pos = np.asarray(mic_pos, dtype=np.float64).reshape(3)
        gen_wall = np.asarray(gen_wall, dtype=np.int64)
        parent_idx = np.asarray(parent_idx, dtype=np.int64)
        rows = np.arange(img_pos.shape[0]) if indices is None else np.asarray(indices, dtype=np.int64)

        out = np.zeros(rows.shape[0], dtype=bool)
        for j, i in enumerate(rows):
            i = int(i)
            if parent_idx[i] == -1:  # Direct path from the real source.
                out[j] = not self._leg_occluded(mic_pos, img_pos[i], set())
                continue

            point = mic_pos
            point_on_wall = -1
            k = i
            visible = True
            while parent_idx[k] != -1:
                wi = int(gen_wall[k])
                ok, R = self._segment_crosses_wall_extent(img_pos[k], point, wi)
                if not ok:
                    visible = False
                    break
                excluded = {wi}
                if point_on_wall != -1:
                    excluded.add(point_on_wall)
                if self._leg_occluded(point, R, excluded):
                    visible = False
                    break
                point = R
                point_on_wall = wi
                k = int(parent_idx[k])
            if visible:
                excluded = set()
                if point_on_wall != -1:
                    excluded.add(point_on_wall)
                if self._leg_occluded(point, img_pos[k], excluded):
                    visible = False
            out[j] = visible
        return out


# Growable node store


@dataclass
class ISMTree:
    """Flat, growable image-source node store.

    Row 0 is the real source, with order 0, no generating wall or parent, and
    unit damping. ``expand_frontier`` appends one order of children for any
    selected set of existing rows.
    """

    img_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    gen_wall: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    parent_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    order: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    damping: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def from_source(cls, src_pos: np.ndarray) -> "ISMTree":
        src = np.asarray(src_pos, dtype=np.float64).reshape(1, 3)
        return cls(
            img_pos=src,
            gen_wall=np.array([-1], dtype=np.int64),
            parent_idx=np.array([-1], dtype=np.int64),
            order=np.array([0], dtype=np.int64),
            damping=np.array([1.0]),
        )

    def __len__(self) -> int:
        return int(self.img_pos.shape[0])

    def expand_frontier(self, room: ExtrudedPolygonRoom, frontier_rows: np.ndarray) -> np.ndarray:
        """Append one order of children and return their row indices."""
        frontier_rows = np.asarray(frontier_rows, dtype=np.int64).reshape(-1)
        if frontier_rows.size == 0:
            return np.zeros(0, dtype=np.int64)
        imgs, walls, ploc, damp = room.expand_one_order(
            self.img_pos[frontier_rows],
            self.gen_wall[frontier_rows],
            self.damping[frontier_rows],
        )
        n_new = imgs.shape[0]
        if n_new == 0:
            return np.zeros(0, dtype=np.int64)
        base = len(self)
        new_rows = np.arange(base, base + n_new, dtype=np.int64)
        self.img_pos = np.concatenate([self.img_pos, imgs], axis=0)
        self.gen_wall = np.concatenate([self.gen_wall, walls])
        self.parent_idx = np.concatenate([self.parent_idx, frontier_rows[ploc]])
        self.order = np.concatenate([self.order, self.order[frontier_rows[ploc]] + 1])
        self.damping = np.concatenate([self.damping, damp])
        return new_rows

    def visibility_matrix(self, room: ExtrudedPolygonRoom, mic_positions: np.ndarray,
                          indices: Optional[Sequence[int]] = None) -> np.ndarray:
        """Return a microphone-by-node visibility matrix."""
        mic_positions = np.asarray(mic_positions, dtype=np.float64)
        M = mic_positions.shape[1]
        cols = []
        for m in range(M):
            cols.append(room.visibility(mic_positions[:, m], self.img_pos,
                                        self.gen_wall, self.parent_idx, indices))
        return np.stack(cols, axis=0)
