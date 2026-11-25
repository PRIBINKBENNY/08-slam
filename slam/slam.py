"""Grid SLAM with correlative scan matching.

For each new scan: predict the pose from odometry, then search a small window of (dx, dy, dtheta)
around it for the pose whose scan best fits the map built so far (likelihood field score),
refine it with a finer search, and insert the scan into the map at the corrected pose.
Odometry drift accumulates without bound; scan matching keeps the trajectory and map consistent.
"""
from __future__ import annotations

import numpy as np

from .mapping import LikelihoodField, OccupancyMap
from .world import Lidar, Odometry, wrap


class ScanMatcher:
    def __init__(self, window_xy: float = 0.15, window_th: float = 0.08, beam_skip: int = 2, sigma: float = 0.06):
        self.wxy, self.wth, self.beam_skip, self.sigma = window_xy, window_th, beam_skip, sigma

    def candidates(self, pose, wxy, wth, nxy, nth):
        dx = np.linspace(-wxy, wxy, nxy)
        dth = np.linspace(-wth, wth, nth)
        X, Y, T = np.meshgrid(dx, dx, dth, indexing="ij")
        return np.column_stack([pose[0] + X.ravel(), pose[1] + Y.ravel(), wrap(pose[2] + T.ravel())])

    def match(self, field: LikelihoodField, pose, ranges, angles) -> tuple[np.ndarray, float]:
        sel = slice(None, None, self.beam_skip)
        r, a = ranges[sel], angles[sel]
        best = np.asarray(pose, float)
        # coarse-to-fine: 3 levels, each 4x finer around the previous best
        wxy, wth = self.wxy, self.wth
        score = -np.inf
        for _ in range(3):
            cand = self.candidates(best, wxy, wth, 9, 9)
            ll = field.log_likelihood(cand, r, a)
            i = int(np.argmax(ll))
            best, score = cand[i], float(ll[i])
            wxy, wth = wxy / 4, wth / 4
        return best, score


class GridSLAM:
    def __init__(self, shape, resolution: float, lidar: Lidar, rebuild_every: int = 5, matcher: ScanMatcher | None = None):
        self.map = OccupancyMap(shape, resolution)
        self.lidar = lidar
        self.matcher = matcher or ScanMatcher()
        self.rebuild_every = rebuild_every
        self.pose = None
        self.trajectory: list[np.ndarray] = []
        self._field = None
        self._since = 0

    def field(self) -> LikelihoodField:
        if self._field is None or self._since >= self.rebuild_every:
            occ = self.map.L > 1.5  # needs >= 2 consistent hits: one noisy endpoint must not make a wall
            self._field = LikelihoodField(occ, self.map.res, sigma=self.matcher.sigma, max_range=self.lidar.max_range)
            self._since = 0
        return self._field

    def step(self, odom_delta, ranges: np.ndarray) -> np.ndarray:
        angles = self.lidar.angles()
        if self.pose is None:
            raise RuntimeError("call start(pose, ranges) first")
        rot1, trans, rot2 = odom_delta
        guess = np.array([self.pose[0] + trans * np.cos(self.pose[2] + rot1),
                          self.pose[1] + trans * np.sin(self.pose[2] + rot1),
                          wrap(self.pose[2] + rot1 + rot2)])
        corrected, _ = self.matcher.match(self.field(), guess, ranges, angles)
        self.pose = corrected
        self.map.update(self.pose, ranges, self.lidar)
        self._since += 1
        self.trajectory.append(self.pose.copy())
        return self.pose

    def start(self, pose, ranges) -> None:
        self.pose = np.asarray(pose, float)
        for _ in range(3):  # a confident first scan anchors the map
            self.map.update(self.pose, ranges, self.lidar)
        self.trajectory = [self.pose.copy()]


def dead_reckon(start, odom_deltas) -> np.ndarray:
    p = np.asarray(start, float)
    out = [p.copy()]
    for rot1, trans, rot2 in odom_deltas:
        p = np.array([p[0] + trans * np.cos(p[2] + rot1), p[1] + trans * np.sin(p[2] + rot1), wrap(p[2] + rot1 + rot2)])
        out.append(p)
    return np.array(out)


def noisy_odometry(truth: np.ndarray, odom: Odometry, rng: np.random.Generator) -> list[tuple]:
    """What wheel encoders would report: the true motion between poses, corrupted by the noise model."""
    out = []
    for p0, p1 in zip(truth[:-1], truth[1:]):
        rot1, trans, rot2 = Odometry.delta(p0, p1)
        n = odom.sample(np.zeros((1, 3)), (rot1, trans, rot2), rng)[0]
        out.append(Odometry.delta(np.zeros(3), n))
    return out
