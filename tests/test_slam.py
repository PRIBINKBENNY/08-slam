import numpy as np
import pytest

from slam import (MCL, TOUR, GridSLAM, LikelihoodField, Lidar, OccupancyMap, Odometry, ScanMatcher, dead_reckon,
                  distance_transform, make_world, noisy_odometry, pose_error, waypoint_path, wrap)


@pytest.fixture(scope="module")
def world():
    return make_world()


@pytest.fixture(scope="module")
def truth():
    return waypoint_path(TOUR)


def test_route_is_collision_free(world, truth):
    d = distance_transform(world.data, world.resolution)
    r, c = world.to_cell(truth[:, 0], truth[:, 1])
    assert d[r, c].min() >= 0.25


def test_raycast_matches_geometry(world):
    lid = Lidar(n_beams=360)
    r = lid.cast(world, (2.0, 4.5, np.pi), noise=False)
    a = lid.angles()
    # facing -x: the outer wall's inner face is at x = 0.1 -> 1.9 m (within one ray-march step)
    assert r[np.argmin(abs(a))] == pytest.approx(1.9, abs=0.05)
    assert (r <= lid.max_range).all() and (r > 0).all()


def test_distance_transform_exact_on_simple_case():
    occ = np.zeros((21, 21), bool)
    occ[10, 10] = True
    d = distance_transform(occ, 0.1, max_dist=10)
    assert d[10, 15] == pytest.approx(0.5) and d[10, 10] == 0
    assert d[13, 14] == pytest.approx(np.hypot(0.3, 0.4), rel=0.1)  # chamfer approximates Euclidean


def test_mapping_with_true_poses_reproduces_walls(world, truth):
    lid, rng = Lidar(), np.random.default_rng(0)
    m = OccupancyMap(world.shape, world.resolution)
    for p in truth[::3]:
        m.update(p, lid.cast(world, p, rng=rng), lid)
    occ = m.L > 1.5
    d = distance_transform(world.data, world.resolution)
    assert occ.sum() > 500
    assert d[occ].mean() < 0.03  # mapped walls sit on real walls
    assert (m.prob()[~world.data] < 0.35).sum() > 0.5 * (~world.data).sum()  # most free space observed as free


def test_odometry_noise_model_is_unbiased():
    rng = np.random.default_rng(0)
    o = Odometry()
    out = o.sample(np.zeros((20000, 3)), (0.1, 0.5, -0.05), rng)
    assert out[:, 0].mean() == pytest.approx(0.5 * np.cos(0.1), abs=0.01)
    assert wrap(out[:, 2]).mean() == pytest.approx(0.05, abs=0.01)


def test_scan_matcher_recovers_offset(world):
    lid = Lidar()
    field = LikelihoodField(world.data, world.resolution, sigma=0.06)
    p = np.array([6.0, 3.7, 0.4])
    ranges = lid.cast(world, p, rng=np.random.default_rng(1))
    best, _ = ScanMatcher().match(field, p + [0.1, -0.08, 0.05], ranges, lid.angles())
    assert pose_error(best, p)[0] < 0.02 and pose_error(best, p)[1] < 0.01


@pytest.mark.parametrize("seed", [1, 2])
def test_slam_beats_odometry(world, truth, seed):
    lid, rng = Lidar(), np.random.default_rng(seed)
    od = noisy_odometry(truth, Odometry(0.02, 0.005, 0.02, 0.005), rng)
    dr = dead_reckon(truth[0], od)
    s = GridSLAM(world.shape, world.resolution, lid)
    s.start(truth[0], lid.cast(world, truth[0], rng=rng))
    for i, u in enumerate(od):
        s.step(u, lid.cast(world, truth[i + 1], rng=rng))
    est = np.array(s.trajectory)
    e_slam = np.hypot(*(est[:, :2] - truth[:, :2]).T)
    e_odo = np.hypot(*(dr[:, :2] - truth[:, :2]).T)
    assert e_slam.mean() < 0.06 and e_slam.max() < 0.15
    assert e_odo.max() > 5 * e_slam.max()


def test_mcl_global_localization_and_kidnap_recovery(world, truth):
    lid = Lidar()
    P = np.vstack([truth[:150], truth[380:]])  # teleported 3.7 m at step 150
    assert np.hypot(*(P[150, :2] - P[149, :2])) > 3
    rng = np.random.default_rng(1)
    od = noisy_odometry(P, Odometry(0.02, 0.005, 0.02, 0.005), rng)
    od[149] = (0.0, 0.0, 0.0)  # wheels report nothing during the kidnap
    m = MCL(world, LikelihoodField(world.data, world.resolution, sigma=0.1), n=3000, seed=1)
    errs = []
    for i, u in enumerate(od):
        est = m.step(u, lid.cast(world, P[i + 1], rng=rng), lid.angles())
        errs.append(pose_error(est, P[i + 1])[0])
    errs = np.array(errs)
    assert errs[0] > 1.0                       # starts genuinely lost
    conv = int(np.argmax(errs < 0.15))
    assert conv < 140 and np.median(errs[conv + 5:149]) < 0.05
    assert errs[150] > 1.0                     # kidnapping is detected as a big error...
    rec = 150 + int(np.argmax(errs[150:] < 0.15))
    assert rec - 150 < 40 and (errs[rec:] < 0.15).mean() > 0.95  # ...and recovered from, staying locked
