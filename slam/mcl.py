"""Monte Carlo Localization: a particle filter that finds the robot on a known map,
including global localization from scratch and recovery from kidnapping."""
from __future__ import annotations

import numpy as np

from .mapping import LikelihoodField
from .world import Grid, Odometry, wrap


class MCL:
    def __init__(self, grid: Grid, field: LikelihoodField, n: int = 2000, odom: Odometry | None = None,
                 beam_skip: int = 2, seed: int = 0, inject_frac: float = 0.02):
        self.grid, self.field = grid, field
        self.n = n
        self.odom = odom or Odometry()
        self.beam_skip = beam_skip
        self.rng = np.random.default_rng(seed)
        self.inject_frac = inject_frac
        self.free_cells = np.argwhere(field.dist > 0.15)  # keep random particles away from walls
        self.particles = self.uniform(n)
        self.weights = np.full(n, 1.0 / n)
        self.w_slow = self.w_fast = None  # Augmented MCL averages (Thrun et al.)
        self.last = None

    def uniform(self, n: int) -> np.ndarray:
        idx = self.rng.integers(0, len(self.free_cells), n)
        rc = self.free_cells[idx]
        x = (rc[:, 1] + self.rng.random(n)) * self.grid.resolution
        y = (rc[:, 0] + self.rng.random(n)) * self.grid.resolution
        return np.column_stack([x, y, self.rng.uniform(-np.pi, np.pi, n)])

    def predict(self, u) -> None:
        self.particles = self.odom.sample(self.particles, u, self.rng)

    def correct(self, ranges: np.ndarray, angles: np.ndarray) -> None:
        sel = slice(None, None, self.beam_skip)
        ll = self.field.log_likelihood(self.particles, ranges[sel], angles[sel])
        # off-map or inside-wall particles are impossible
        r, c = self.grid.to_cell(self.particles[:, 0], self.particles[:, 1])
        inside = self.grid.inside(r, c)
        ll[~inside] = -np.inf
        ll[inside] += np.where(self.field.dist[r[inside], c[inside]] < 0.05, -50.0, 0.0)
        m = ll.max()
        w = np.exp(ll - m)
        avg = float(np.mean(w)) * np.exp(m / max(1, ranges[sel].size))  # scale-free-ish average for recovery
        self.weights = w / w.sum()
        # Augmented MCL: if the short-term likelihood falls below the long-term one, the robot is lost
        self.w_slow = avg if self.w_slow is None else self.w_slow + 0.01 * (avg - self.w_slow)
        self.w_fast = avg if self.w_fast is None else self.w_fast + 0.3 * (avg - self.w_fast)

    def neff(self) -> float:
        return 1.0 / np.sum(self.weights ** 2)

    def resample(self) -> None:
        """Low-variance (systematic) resampling, plus random particles when the filter looks lost."""
        p_inject = self.inject_frac
        if self.w_slow:
            p_inject = max(p_inject, min(0.2, 1.0 - self.w_fast / self.w_slow))
        n_rand = int(self.n * p_inject)
        n_keep = self.n - n_rand
        positions = (self.rng.random() + np.arange(n_keep)) / n_keep
        idx = np.searchsorted(np.cumsum(self.weights), positions)
        idx = np.minimum(idx, self.n - 1)
        kept = self.particles[idx]
        self.particles = np.vstack([kept, self.uniform(n_rand)]) if n_rand else kept
        self.weights = np.full(self.n, 1.0 / self.n)

    def step(self, u, ranges, angles) -> np.ndarray:
        self.predict(u)
        self.correct(ranges, angles)
        if self.neff() < self.n / 2:
            self.resample()
        return self.estimate()

    def estimate(self) -> np.ndarray:
        """Weighted mean of the region holding the most probability mass. (The mean of a multimodal
        cloud is meaningless, and after resampling all weights are equal, so 'best particle' is too.)"""
        cell = 0.5
        ij = np.floor(self.particles[:, :2] / cell).astype(int)
        keys = ij[:, 0] * 100_000 + ij[:, 1]
        uniq, inv = np.unique(keys, return_inverse=True)
        mass = np.bincount(inv, weights=self.weights)
        # smooth over the 3x3 neighbourhood so a cluster split across bin edges still wins
        lookup = dict(zip(uniq.tolist(), mass.tolist()))
        best_key, best_mass = None, -1.0
        for k in uniq.tolist():
            bx, by = divmod(k, 100_000)
            m = sum(lookup.get((bx + dx) * 100_000 + (by + dy), 0.0) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
            if m > best_mass:
                best_key, best_mass = k, m
        bx, by = divmod(best_key, 100_000)
        centre = np.array([(bx + 0.5) * cell, (by + 0.5) * cell])
        # hysteresis: stay with the cluster we were tracking unless another holds clearly more mass
        if self.last is not None:
            d_prev = np.hypot(*(self.particles[:, :2] - self.last[:2]).T)
            if self.weights[d_prev < 1.5 * cell].sum() >= 0.5 * best_mass:
                centre = self.last[:2]
        near = np.hypot(*(self.particles[:, :2] - centre).T) < 1.5 * cell
        w = self.weights[near]
        p = self.particles[near]
        x, y = np.average(p[:, 0], weights=w), np.average(p[:, 1], weights=w)
        th = np.arctan2(np.average(np.sin(p[:, 2]), weights=w), np.average(np.cos(p[:, 2]), weights=w))
        self.last = np.array([x, y, th])
        return self.last

    def spread(self) -> float:
        return float(np.sqrt(np.cov(self.particles[:, :2].T, aweights=self.weights).trace()))


def pose_error(est, truth) -> tuple[float, float]:
    return float(np.hypot(est[0] - truth[0], est[1] - truth[1])), float(abs(wrap(est[2] - truth[2])))
