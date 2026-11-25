"""Figures and animations for the README: python -m slam.viz  ->  docs/*.png, docs/*.gif"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from .mapping import LikelihoodField, OccupancyMap
from .mcl import MCL, pose_error
from .slam import GridSLAM, dead_reckon, noisy_odometry
from .world import TOUR, Lidar, Odometry, make_world, waypoint_path

INK, TRUTH, ODOM, EST, PART = "#1c1f26", "#2a9d8f", "#e76f51", "#264bd8", "#f4a261"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11})
    return plt


def _show_map(ax, img, extent, title):
    ax.imshow(img, cmap="Greys", origin="lower", extent=extent, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, loc="left")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def run_slam(seed: int = 1):
    g = make_world()
    lid = Lidar()
    truth = waypoint_path(TOUR)
    rng = np.random.default_rng(seed)
    odom = Odometry(0.02, 0.005, 0.02, 0.005)
    od = noisy_odometry(truth, odom, rng)
    scans = [lid.cast(g, p, rng=rng) for p in truth]
    dr = dead_reckon(truth[0], od)
    s = GridSLAM(g.shape, g.resolution, lid)
    s.start(truth[0], scans[0])
    for i, u in enumerate(od):
        s.step(u, scans[i + 1])
    odo_map = OccupancyMap(g.shape, g.resolution)
    for p, r in zip(dr, scans):
        odo_map.update(p, r, lid)
    return g, truth, dr, np.array(s.trajectory), s.map, odo_map


def fig_slam(out: Path, seed: int = 1):
    plt = _plt()
    g, truth, dr, est, smap, omap = run_slam(seed)
    ext = (0, g.size_m[0], 0, g.size_m[1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    _show_map(axes[0], g.data.astype(float), ext, "Ground truth + route")
    axes[0].plot(truth[:, 0], truth[:, 1], color=TRUTH, lw=2)
    _show_map(axes[1], omap.prob(), ext, "Mapping from raw odometry (drifts)")
    axes[1].plot(dr[:, 0], dr[:, 1], color=ODOM, lw=1.6, label="odometry")
    axes[1].plot(truth[:, 0], truth[:, 1], color=TRUTH, lw=1, ls="--", label="truth")
    _show_map(axes[2], smap.prob(), ext, "Grid SLAM with scan matching")
    axes[2].plot(est[:, 0], est[:, 1], color=EST, lw=1.6, label="SLAM")
    axes[2].plot(truth[:, 0], truth[:, 1], color=TRUTH, lw=1, ls="--", label="truth")
    for ax in axes[1:]:
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "slam_maps.png", dpi=110, facecolor="white")
    plt.close(fig)

    e_odo = np.hypot(*(dr[:, :2] - truth[:, :2]).T)
    e_slam = np.hypot(*(est[:, :2] - truth[:, :2]).T)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(e_odo, color=ODOM, lw=2, label=f"odometry only (final {e_odo[-1]:.2f} m)")
    ax.plot(e_slam, color=EST, lw=2, label=f"SLAM (mean {e_slam.mean() * 100:.1f} cm)")
    ax.set(xlabel="step", ylabel="position error (m)", title="Position error along the route")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "slam_error.png", dpi=110, facecolor="white")
    plt.close(fig)
    return {"odom_final_m": round(float(e_odo[-1]), 3), "odom_max_m": round(float(e_odo.max()), 3),
            "slam_mean_cm": round(float(e_slam.mean() * 100), 2), "slam_max_cm": round(float(e_slam.max() * 100), 2)}


def mcl_run(seed: int = 1, n: int = 3000, kidnap_from: int = 150, kidnap_to: int = 380):
    g = make_world()
    lid = Lidar()
    truth = waypoint_path(TOUR)
    P = np.vstack([truth[:kidnap_from], truth[kidnap_to:]])
    rng = np.random.default_rng(seed)
    od = noisy_odometry(P, Odometry(0.02, 0.005, 0.02, 0.005), rng)
    od[kidnap_from - 1] = (0.0, 0.0, 0.0)
    field = LikelihoodField(g.data, g.resolution, sigma=0.1)
    m = MCL(g, field, n=n, seed=seed)
    frames, errs = [(m.particles.copy(), P[0], None)], []
    for i, u in enumerate(od):
        est = m.step(u, lid.cast(g, P[i + 1], rng=rng), lid.angles())
        errs.append(pose_error(est, P[i + 1])[0])
        frames.append((m.particles.copy(), P[i + 1], est))
    return g, P, frames, np.array(errs), kidnap_from


def fig_mcl(out: Path, seed: int = 1):
    plt = _plt()
    g, P, frames, errs, k = mcl_run(seed)
    ext = (0, g.size_m[0], 0, g.size_m[1])
    conv = int(np.argmax(errs < 0.15)) + 1          # frames[i] is the state after step i-1
    rec = k + int(np.argmax(errs[k:] < 0.15)) + 1
    picks = [0, max(1, conv // 3), conv, k - 1, k + 1, min(len(frames) - 1, rec + 3)]
    names = ["start: no idea where it is", f"step {picks[1]}: hypotheses narrowing", f"step {conv}: localized",
             f"step {k - 1}: tracking", f"step {k + 1}: kidnapped", f"step {picks[5]}: recovered"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, i, name in zip(axes.ravel(), picks, names):
        parts, tp, est = frames[i]
        _show_map(ax, g.data.astype(float), ext, name)
        ax.scatter(parts[:, 0], parts[:, 1], s=5, color=PART, alpha=0.55, lw=0, zorder=3)
        ax.plot(*tp[:2], "o", ms=11, mfc="none", mec=TRUTH, mew=2.5, zorder=4)
        if est is not None:
            ax.plot(*est[:2], "x", ms=9, color=EST, mew=2.5, zorder=5)
    from matplotlib.lines import Line2D

    fig.legend([Line2D([], [], ls="", marker="o", color=PART), Line2D([], [], ls="", marker="o", mfc="none", mec=TRUTH, mew=2),
                Line2D([], [], ls="", marker="x", color=EST, mew=2)], ["particles", "true pose", "estimate"],
               loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.2)
    fig.savefig(out / "mcl_snapshots.png", dpi=100, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(errs, color=EST, lw=2)
    ax.axvline(k - 1, color=ODOM, ls="--", lw=1.5)
    ax.text(k + 2, ax.get_ylim()[1] * 0.85, "robot kidnapped", color=ODOM)
    ax.set_yscale("log")
    ax.set(xlabel="step", ylabel="position error (m, log)", title="Monte Carlo localization: global start and kidnap recovery")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "mcl_error.png", dpi=110, facecolor="white")
    plt.close(fig)

    # animated GIF
    from PIL import Image

    imgs = []
    idx = list(range(0, conv + 8, 2)) + list(range(conv + 8, k - 3, 8)) + list(range(k - 3, min(len(frames), rec + 20)))
    for i in idx:
        parts, tp, est = frames[i]
        fig, ax = plt.subplots(figsize=(6, 4.6))
        _show_map(ax, g.data.astype(float), ext, f"Particle filter localization · step {i}")
        ax.scatter(parts[:, 0], parts[:, 1], s=5, color=PART, alpha=0.55, lw=0, zorder=3)
        ax.plot(*tp[:2], "o", ms=11, mfc="none", mec=TRUTH, mew=2.5, zorder=4)
        if est is not None:
            ax.plot(*est[:2], "x", ms=9, color=EST, mew=2.5, zorder=5)
        if k <= i <= k + 6:
            ax.text(0.5, 0.5, "KIDNAPPED", transform=ax.transAxes, ha="center", fontsize=20, color=ODOM, weight="bold")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, dpi=72, facecolor="white")
        plt.close(fig)
        imgs.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=64))
    imgs[0].save(out / "mcl.gif", save_all=True, append_images=imgs[1:], duration=90, loop=0, optimize=True)
    return {"converged_step": int(np.argmax(errs < 0.15)), "recovered_after_steps": int(np.argmax(errs[k:] < 0.15)),
            "tracking_error_cm": round(float(np.median(errs[60:k - 1]) * 100), 2)}


def main() -> int:
    out = Path("docs")
    out.mkdir(exist_ok=True)
    print("slam:", fig_slam(out))
    print("mcl:", fig_mcl(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
