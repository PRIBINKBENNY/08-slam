"""Occupancy grid mapping (log-odds) and the likelihood-field sensor model shared by MCL and SLAM."""
from __future__ import annotations

import numpy as np

from .world import Grid, Lidar

L_OCC, L_FREE, L_MIN, L_MAX = 0.85, -0.4, -4.0, 4.0


def logodds_to_prob(L: np.ndarray) -> np.ndarray:
    return 1.0 - 1.0 / (1.0 + np.exp(L))


class OccupancyMap:
    def __init__(self, shape: tuple[int, int], resolution: float):
        self.L = np.zeros(shape, dtype=np.float32)
        self.res = resolution

    @property
    def grid(self) -> Grid:
        return Grid(self.L, self.res)

    def prob(self) -> np.ndarray:
        return logodds_to_prob(self.L)

    def update(self, pose, ranges: np.ndarray, lidar: Lidar) -> None:
        """Inverse sensor model: cells along each beam become freer, the endpoint cell more occupied.
        Beams at max range mark free space only."""
        x, y, th = pose
        ang = th + lidar.angles()
        H, W = self.L.shape
        free_r, free_c, occ_r, occ_c = [], [], [], []
        step = self.res * 0.7
        for a, rng in zip(ang, ranges):
            hit = rng < lidar.max_range - 1e-6
            d = np.arange(0, max(rng - self.res, 0), step)
            fx, fy = x + np.cos(a) * d, y + np.sin(a) * d
            free_r.append((fy / self.res).astype(int))
            free_c.append((fx / self.res).astype(int))
            if hit:
                occ_r.append(int((y + np.sin(a) * rng) / self.res))
                occ_c.append(int((x + np.cos(a) * rng) / self.res))
        fr, fc = np.concatenate(free_r), np.concatenate(free_c)
        ok = (fr >= 0) & (fr < H) & (fc >= 0) & (fc < W)
        free = np.unique(fr[ok] * W + fc[ok])  # each cell updated once per scan
        self.L.flat[free] += L_FREE
        if occ_r:
            orr, occ = np.array(occ_r), np.array(occ_c)
            ok = (orr >= 0) & (orr < H) & (occ >= 0) & (occ < W)
            cells = np.unique(orr[ok] * W + occ[ok])
            self.L.flat[cells] += L_OCC - L_FREE  # cancels the free update if the endpoint was on a free ray
        np.clip(self.L, L_MIN, L_MAX, out=self.L)


def distance_transform(occupied: np.ndarray, resolution: float, max_dist: float = 2.0) -> np.ndarray:
    """Metric distance from every cell to the nearest occupied cell (two-pass chamfer 3-4 transform)."""
    INF = 1e9
    d = np.where(occupied, 0.0, INF)
    H, W = d.shape
    a, b = 1.0, np.sqrt(2.0)
    # forward pass
    for r in range(H):
        row = d[r]
        if r > 0:
            up = d[r - 1]
            row = np.minimum(row, up + a)
            row[1:] = np.minimum(row[1:], up[:-1] + b)
            row[:-1] = np.minimum(row[:-1], up[1:] + b)
        for c in range(1, W):  # left-to-right dependency
            if row[c - 1] + a < row[c]:
                row[c] = row[c - 1] + a
        d[r] = row
    # backward pass
    for r in range(H - 1, -1, -1):
        row = d[r]
        if r < H - 1:
            dn = d[r + 1]
            row = np.minimum(row, dn + a)
            row[1:] = np.minimum(row[1:], dn[:-1] + b)
            row[:-1] = np.minimum(row[:-1], dn[1:] + b)
        for c in range(W - 2, -1, -1):
            if row[c + 1] + a < row[c]:
                row[c] = row[c + 1] + a
        d[r] = row
    return np.minimum(d * resolution, max_dist)


class LikelihoodField:
    """p(z | pose, map) from the distance of each beam endpoint to the nearest obstacle."""

    def __init__(self, occupied: np.ndarray, resolution: float, sigma: float = 0.08, z_rand: float = 0.05,
                 max_range: float = 8.0):
        self.res = resolution
        self.dist = distance_transform(occupied, resolution)
        self.sigma, self.z_rand, self.max_range = sigma, z_rand, max_range
        self.shape = occupied.shape

    def log_likelihood(self, poses: np.ndarray, ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """poses: [N,3]; ranges, angles: [B]. Returns log p(z|x) per pose, vectorised over N x B."""
        use = ranges < self.max_range - 1e-6
        r, a = ranges[use], angles[use]
        if r.size == 0:
            return np.zeros(len(poses))
        ex = poses[:, 0:1] + r[None] * np.cos(poses[:, 2:3] + a[None])
        ey = poses[:, 1:2] + r[None] * np.sin(poses[:, 2:3] + a[None])
        rr, cc = (ey / self.res).astype(int), (ex / self.res).astype(int)
        H, W = self.shape
        inside = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        d = np.full(rr.shape, self.dist.max())
        d[inside] = self.dist[rr[inside], cc[inside]]
        p = (1 - self.z_rand) * np.exp(-0.5 * (d / self.sigma) ** 2) + self.z_rand / self.max_range
        return np.log(p).sum(1)
