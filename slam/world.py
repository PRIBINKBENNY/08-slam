"""A 2D occupancy-grid world, a simulated 2D LiDAR (ray casting) and a noisy odometry model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    """Occupancy grid. data[row, col]: row = y index, col = x index. Origin at (0, 0) metres."""

    data: np.ndarray          # bool (truth) or float (probabilities / log-odds)
    resolution: float = 0.05  # metres per cell

    @property
    def shape(self):
        return self.data.shape

    @property
    def size_m(self) -> tuple[float, float]:
        return self.data.shape[1] * self.resolution, self.data.shape[0] * self.resolution

    def to_cell(self, x, y):
        return (np.floor(np.asarray(y) / self.resolution).astype(int),
                np.floor(np.asarray(x) / self.resolution).astype(int))

    def inside(self, r, c):
        return (r >= 0) & (r < self.data.shape[0]) & (c >= 0) & (c < self.data.shape[1])


def make_world(seed: int = 0, w_m: float = 12.0, h_m: float = 9.0, res: float = 0.05) -> Grid:
    """An apartment-like floor plan: outer walls, interior walls with doorways, furniture blocks."""
    W, H = int(w_m / res), int(h_m / res)
    g = np.zeros((H, W), dtype=bool)

    def box(x0, y0, x1, y1):
        g[int(y0 / res):int(y1 / res), int(x0 / res):int(x1 / res)] = True

    box(0, 0, w_m, 0.1); box(0, h_m - 0.1, w_m, h_m); box(0, 0, 0.1, h_m); box(w_m - 0.1, 0, w_m, h_m)
    box(4.0, 0, 4.1, 3.2); box(4.0, 4.2, 4.1, h_m)            # wall with a doorway at y 3.2-4.2
    box(4.1, 5.0, 7.0, 5.1); box(8.0, 5.0, w_m, 5.1)          # wall with a doorway at x 7-8
    box(8.0, 0, 8.1, 2.2); box(8.0, 3.2, 8.1, 5.0)            # doorway at y 2.2-3.2
    box(1.0, 6.5, 2.6, 7.5)    # sofa
    box(2.0, 1.2, 2.8, 2.0)    # table
    box(5.5, 1.5, 6.5, 2.1)    # desk
    box(9.5, 6.5, 10.8, 8.0)   # bed
    box(10.5, 1.0, 11.2, 1.7)  # cabinet
    rng = np.random.default_rng(seed)
    placed = 0
    while placed < 6:          # a few small clutter items (chair legs, plants), kept clear of the robot's route
        x, y = rng.uniform(1, w_m - 1), rng.uniform(1, h_m - 1)
        if _dist_to_route(x + 0.1, y + 0.1, TOUR) < 0.6:
            continue
        box(x, y, x + 0.2, y + 0.2)
        placed += 1
    return Grid(g, res)


def _dist_to_route(x: float, y: float, pts) -> float:
    best = np.inf
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        vx, vy = x1 - x0, y1 - y0
        t = np.clip(((x - x0) * vx + (y - y0) * vy) / max(vx * vx + vy * vy, 1e-12), 0, 1)
        best = min(best, np.hypot(x - (x0 + t * vx), y - (y0 + t * vy)))
    return best


@dataclass
class Lidar:
    n_beams: int = 180
    fov: float = 2 * np.pi
    max_range: float = 8.0
    sigma: float = 0.02      # range noise (m)
    p_dropout: float = 0.01  # beams that return nothing

    def angles(self) -> np.ndarray:
        if self.fov >= 2 * np.pi - 1e-9:
            return np.linspace(-np.pi, np.pi, self.n_beams, endpoint=False)
        return np.linspace(-self.fov / 2, self.fov / 2, self.n_beams)

    def cast(self, grid: Grid, pose, noise: bool = True, rng: np.random.Generator | None = None) -> np.ndarray:
        """Ray-march every beam through the true map (vectorised). Returns ranges; max_range = no hit."""
        x, y, th = pose
        step = grid.resolution * 0.5
        dists = np.arange(step, self.max_range + step, step)
        ang = th + self.angles()
        xs = x + np.cos(ang)[:, None] * dists[None, :]
        ys = y + np.sin(ang)[:, None] * dists[None, :]
        r, c = grid.to_cell(xs, ys)
        inside = grid.inside(r, c)
        hit = np.zeros_like(inside)
        hit[inside] = grid.data[r[inside], c[inside]]
        hit |= ~inside
        first = np.where(hit.any(1), hit.argmax(1), len(dists) - 1)
        ranges = np.where(hit.any(1), dists[first], self.max_range)
        if noise:
            rng = rng or np.random.default_rng()
            ranges = ranges + rng.normal(0, self.sigma, ranges.shape)
            ranges[rng.random(ranges.shape) < self.p_dropout] = self.max_range
        return np.clip(ranges, 0.0, self.max_range)


@dataclass
class Odometry:
    """Odometry motion model (rot1, trans, rot2) with the classic alpha noise parameters."""

    a1: float = 0.05  # rotation noise from rotation
    a2: float = 0.01  # rotation noise from translation
    a3: float = 0.05  # translation noise from translation
    a4: float = 0.01  # translation noise from rotation

    @staticmethod
    def delta(p0, p1):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        trans = np.hypot(dx, dy)
        rot1 = np.arctan2(dy, dx) - p0[2] if trans > 1e-6 else 0.0
        rot2 = p1[2] - p0[2] - rot1
        return wrap(rot1), trans, wrap(rot2)

    def sample(self, poses: np.ndarray, u, rng: np.random.Generator) -> np.ndarray:
        """Apply odometry u=(rot1, trans, rot2) to N poses with sampled noise. poses: [N,3]."""
        rot1, trans, rot2 = u
        n = len(poses)
        r1 = rot1 - rng.normal(0, np.sqrt(self.a1 * rot1**2 + self.a2 * trans**2) + 1e-9, n)
        tr = trans - rng.normal(0, np.sqrt(self.a3 * trans**2 + self.a4 * (rot1**2 + rot2**2)) + 1e-9, n)
        r2 = rot2 - rng.normal(0, np.sqrt(self.a1 * rot2**2 + self.a2 * trans**2) + 1e-9, n)
        out = poses.copy()
        out[:, 0] += tr * np.cos(poses[:, 2] + r1)
        out[:, 1] += tr * np.sin(poses[:, 2] + r1)
        out[:, 2] = wrap(poses[:, 2] + r1 + r2)
        return out


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def waypoint_path(points, step: float = 0.08, turn_step: float = 0.15) -> np.ndarray:
    """Ground-truth poses that drive through waypoints: turn on the spot, then drive straight."""
    poses = []
    x, y = points[0]
    th = np.arctan2(points[1][1] - y, points[1][0] - x)
    poses.append((x, y, th))
    for tx, ty in points[1:]:
        target = np.arctan2(ty - y, tx - x)
        while abs(wrap(target - th)) > 1e-6:
            th = th + np.clip(wrap(target - th), -turn_step, turn_step)
            poses.append((x, y, float(wrap(th))))
        d = np.hypot(tx - x, ty - y)
        n = max(1, int(np.ceil(d / step)))
        for i in range(1, n + 1):
            poses.append((x + (tx - x) * i / n, y + (ty - y) * i / n, float(wrap(th))))
        x, y = tx, ty
    return np.array(poses)


# a tour through every room, using the doorways (tests check it never comes within 25 cm of a wall)
TOUR = [(1.5, 4.5), (3.3, 4.5), (3.3, 3.7), (6.0, 3.7), (7.5, 3.4), (7.5, 4.3), (7.5, 6.5),
        (6.0, 7.5), (8.6, 7.5), (7.5, 6.5), (7.5, 2.7), (10.0, 2.7), (10.0, 4.2), (11.0, 3.5)]
