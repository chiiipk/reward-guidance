"""Plotting for checkerboard reward guidance experiments."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from model import checkerboard_density, reward_fn, DEFAULT_REWARD_CENTER
from sample import best_of_n, softmax_of_n

REPO = Path(__file__).resolve().parents[1]
for ttf in (REPO / "assets" / "fonts").glob("*.ttf"):
    fm.fontManager.addfont(str(ttf))

plt.style.use(str(REPO / "assets" / "default.mplstyle"))
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "DejaVu Sans"],
    "font.size": 18,
    "axes.labelsize": 19,
    "axes.titlesize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "text.usetex": False,
})

DISPLAY_N = 5000  # Subsample target for paper-figure scatter plots

GRID_RANGE = (-3, 3)
GRID_RES = 300


def _make_grid():
    x = np.linspace(*GRID_RANGE, GRID_RES)
    y = np.linspace(*GRID_RANGE, GRID_RES)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X, Y], axis=-1)  # (RES, RES, 2)
    return X, Y, points


def _checkerboard_background(ax, alpha=0.15):
    """Draw the checkerboard pattern as a faint background."""
    X, Y, points = _make_grid()
    density = checkerboard_density(points)
    ax.contourf(X, Y, density, levels=[0.5, 1.5], colors=["gray"], alpha=alpha)


def _guided_data_path(output_dir, k, n, lam, sigma_damp=None):
    damp_suffix = f"_damp{sigma_damp}" if sigma_damp is not None else ""
    if n > 1:
        return os.path.join(output_dir, f"bon_k{k}_n{n}_lam{lam}{damp_suffix}.npz")
    return os.path.join(output_dir, f"guided_k{k}_lam{lam}{damp_suffix}.npz")


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------


def plot_unguided(output_dir, image_dir):
    data = np.load(os.path.join(output_dir, "unguided.npz"))
    samples = data["samples"]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    _checkerboard_background(ax)
    ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5, c="C0", edgecolors="none")
    ax.set_xlim(*GRID_RANGE)
    ax.set_ylim(*GRID_RANGE)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    out = os.path.join(image_dir, "unguided_samples.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_guided(output_dir, image_dir, k, n, lam, sigma_damp=None):
    path = _guided_data_path(output_dir, k, n, lam, sigma_damp)

    if not os.path.exists(path):
        print(f"No cached data at {path}. Run sample.py first.")
        return

    data = np.load(path)
    samples = data["samples"]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    _checkerboard_background(ax)
    ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5, c="C1", edgecolors="none")

    # Mark reward center
    center = DEFAULT_REWARD_CENTER.numpy()
    ax.scatter(*center, s=80, c="red", marker="*", zorder=5)

    ax.set_xlim(*GRID_RANGE)
    ax.set_ylim(*GRID_RANGE)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    out = os.path.join(image_dir, f"guided_k{k}_n{n}.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_analytic_tilt(image_dir, lam=5.0, sigma_r=0.5):
    """Plot the analytic tilted density rho_1(x) * exp(lambda * r(x))."""
    X, Y, points = _make_grid()
    points_tensor = torch.tensor(points, dtype=torch.float32)

    # Checkerboard density (piecewise constant)
    rho = checkerboard_density(points)

    # Reward
    flat_points = points_tensor.reshape(-1, 2)
    center = DEFAULT_REWARD_CENTER
    r = reward_fn(flat_points, center, sigma_r).numpy().reshape(GRID_RES, GRID_RES)

    # Tilted density (unnormalized)
    tilt = rho * np.exp(lam * r)
    # Normalize for visualization
    if tilt.max() > 0:
        tilt = tilt / tilt.max()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    c = ax.pcolormesh(X, Y, tilt, shading="auto", cmap="viridis", rasterized=True)
    fig.colorbar(c, ax=ax, shrink=0.8)

    # Mark reward center
    center_np = center.numpy()
    ax.scatter(*center_np, s=80, c="red", marker="*", zorder=5)

    ax.set_xlim(*GRID_RANGE)
    ax.set_ylim(*GRID_RANGE)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    fig.savefig(os.path.join(image_dir, "analytic_tilt.pdf"))
    plt.close(fig)
    print(f"Saved {os.path.join(image_dir, 'analytic_tilt.pdf')}")


def plot_analytic_tilt_3d(image_dir, lam=5.0, sigma_r=0.5):
    """Surface plot of the analytic tilted density."""
    X, Y, points = _make_grid()
    points_tensor = torch.tensor(points, dtype=torch.float32)

    rho = checkerboard_density(points)
    flat_points = points_tensor.reshape(-1, 2)
    center = DEFAULT_REWARD_CENTER
    r = reward_fn(flat_points, center, sigma_r).numpy().reshape(GRID_RES, GRID_RES)

    tilt = rho * np.exp(lam * r)
    if tilt.max() > 0:
        tilt = tilt / tilt.max()

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, tilt, cmap="viridis", rasterized=True, alpha=0.9)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel("Tilted density")
    fig.savefig(os.path.join(image_dir, "analytic_tilt_3d.pdf"))
    plt.close(fig)
    print(f"Saved {os.path.join(image_dir, 'analytic_tilt_3d.pdf')}")


def sample_analytic_tilt(n_samples, lam, sigma_r):
    """Draw exact samples from ρ̃₁(x) ∝ exp(λ r(x)) ρ₁(x) via rejection sampling.

    Algorithm:
      1. Sample a filled square proportional to its tilted mass ∫ exp(λ r(x)) dx.
      2. Sample uniformly within that square.
      3. Accept with probability exp(λ (r(x) − 1))  [tight bound since r ≤ 1].
    """
    squares = _enumerate_squares()
    # Sample squares uniformly — ρ₁ is uniform on all filled squares, so the
    # acceptance step alone (exp(λ(r(x)−1))) handles the tilt correctly.
    square_probs = np.ones(len(squares)) / len(squares)

    # Oversample per batch to amortize overhead; worst-case accept rate is exp(-λ)
    batch_size = max(n_samples * 4, int(n_samples * np.exp(lam) + 1))

    collected = []
    while sum(len(c) for c in collected) < n_samples:
        sq_idx = np.random.choice(len(squares), size=batch_size, p=square_probs)
        x_lo = np.array([squares[i][2] for i in sq_idx], dtype=np.float32)
        y_lo = np.array([squares[i][3] for i in sq_idx], dtype=np.float32)
        pts = np.stack([
            x_lo + np.random.uniform(0, 1, batch_size).astype(np.float32),
            y_lo + np.random.uniform(0, 1, batch_size).astype(np.float32),
        ], axis=-1)

        r = reward_fn(torch.tensor(pts), DEFAULT_REWARD_CENTER, sigma_r).numpy()
        accept = np.random.uniform(0, 1, batch_size) < np.exp(lam * (r - 1.0))
        collected.append(pts[accept])

    return np.concatenate(collected, axis=0)[:n_samples]


def plot_analytic_tilt_samples(output_dir, image_dir, lam, sigma_r, n_samples=5000):
    """Scatter plot of exact samples from the analytic tilted density."""
    cache = os.path.join(output_dir, f"analytic_tilt_lam{lam}.npz")
    if os.path.exists(cache):
        samples = np.load(cache)["samples"]
        print(f"Loaded {len(samples)} analytic tilt samples from cache.")
    else:
        print(f"Sampling {n_samples} from analytic tilt (λ={lam})...")
        samples = sample_analytic_tilt(n_samples, lam, sigma_r)
        np.savez(cache, samples=samples)
        print(f"Cached to {cache}")

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    _checkerboard_background(ax)
    ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5, c="C2", edgecolors="none")
    center_np = DEFAULT_REWARD_CENTER.numpy()
    ax.scatter(*center_np, s=80, c="red", marker="*", zorder=5)
    ax.set_xlim(*GRID_RANGE)
    ax.set_ylim(*GRID_RANGE)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    out = os.path.join(image_dir, "analytic_tilt_samples.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_reward(image_dir, sigma_r=1.5):
    """Plot the reward function r(x) = exp(-||x - c||^2 / (2 sigma_r^2))."""
    X, Y, points = _make_grid()
    points_tensor = torch.tensor(points, dtype=torch.float32)
    flat_points = points_tensor.reshape(-1, 2)
    center = DEFAULT_REWARD_CENTER
    r = reward_fn(flat_points, center, sigma_r).numpy().reshape(GRID_RES, GRID_RES)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    _checkerboard_background(ax, alpha=0.10)
    c = ax.pcolormesh(X, Y, r, shading="auto", cmap="viridis", rasterized=True, alpha=0.85)
    fig.colorbar(c, ax=ax, shrink=0.8)

    center_np = center.numpy()
    ax.scatter(*center_np, s=80, c="red", marker="*", zorder=5)

    ax.set_xlim(*GRID_RANGE)
    ax.set_ylim(*GRID_RANGE)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    out = os.path.join(image_dir, "reward.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _enumerate_squares():
    """Return list of (i, j, x_lo, y_lo) for the 18 filled squares."""
    squares = []
    for i in range(6):
        for j in range(6):
            if (i + j) % 2 == 0:
                squares.append((i, j, i - 3, j - 3))
    return squares


def _analytic_square_masses(lam, sigma_r):
    """Compute the analytic tilted mass for each filled square.

    Returns dict mapping (i, j) -> normalized probability under the tilt.
    Uses numerical integration (midpoint rule on a fine grid within each square).
    """
    center = DEFAULT_REWARD_CENTER.numpy()
    squares = _enumerate_squares()
    masses = {}
    res = 50  # grid points per square side
    for i, j, x_lo, y_lo in squares:
        xs = np.linspace(x_lo, x_lo + 1, res, endpoint=False) + 0.5 / res
        ys = np.linspace(y_lo, y_lo + 1, res, endpoint=False) + 0.5 / res
        Xg, Yg = np.meshgrid(xs, ys)
        pts = np.stack([Xg, Yg], axis=-1)
        dist_sq = np.sum((pts - center) ** 2, axis=-1)
        r = np.exp(-dist_sq / (2 * sigma_r ** 2))
        # Integral of exp(lam * r(x)) over the square (uniform base density)
        masses[(i, j)] = np.mean(np.exp(lam * r))
    total = sum(masses.values())
    return {k: v / total for k, v in masses.items()}


def _empirical_square_masses(samples):
    """Compute the empirical fraction of samples in each filled square."""
    squares = _enumerate_squares()
    counts = {}
    n = len(samples)
    for i, j, x_lo, y_lo in squares:
        mask = (
            (samples[:, 0] >= x_lo) & (samples[:, 0] < x_lo + 1) &
            (samples[:, 1] >= y_lo) & (samples[:, 1] < y_lo + 1)
        )
        counts[(i, j)] = mask.sum() / n
    return counts


def _analytic_tilt_moments(lam, sigma_r):
    """Compute mean, covariance, and mean reward under the analytic tilted density.

    Uses numerical integration over each filled square.
    """
    center = DEFAULT_REWARD_CENTER.numpy()
    squares = _enumerate_squares()
    res = 50

    # Gather all weighted points
    all_pts = []
    all_weights = []
    all_rewards = []
    for i, j, x_lo, y_lo in squares:
        xs = np.linspace(x_lo, x_lo + 1, res, endpoint=False) + 0.5 / res
        ys = np.linspace(y_lo, y_lo + 1, res, endpoint=False) + 0.5 / res
        Xg, Yg = np.meshgrid(xs, ys)
        pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)  # (res^2, 2)
        dist_sq = np.sum((pts - center) ** 2, axis=-1)
        r = np.exp(-dist_sq / (2 * sigma_r ** 2))
        w = np.exp(lam * r)
        all_pts.append(pts)
        all_weights.append(w)
        all_rewards.append(r)

    pts = np.concatenate(all_pts, axis=0)
    weights = np.concatenate(all_weights)
    rewards = np.concatenate(all_rewards)
    weights /= weights.sum()

    mean = np.average(pts, weights=weights, axis=0)
    diff = pts - mean
    cov = np.einsum("i,ij,ik->jk", weights, diff, diff)
    mean_reward = np.average(rewards, weights=weights)

    return mean, cov, mean_reward


def compute_diagnostics(output_dir, k, n, lam, sigma_r, sigma_damp=None):
    """Print diagnostic statistics comparing guided samples to analytic tilt."""
    path = _guided_data_path(output_dir, k, n, lam, sigma_damp)
    unguided_path = os.path.join(output_dir, "unguided.npz")

    center = DEFAULT_REWARD_CENTER.numpy()

    print("=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)

    # Analytic tilt statistics
    analytic_masses = _analytic_square_masses(lam, sigma_r)

    # Unguided
    u_masses = None
    if os.path.exists(unguided_path):
        u = np.load(unguided_path)["samples"]
        u_tensor = torch.tensor(u, dtype=torch.float32)
        u_rewards = reward_fn(u_tensor, DEFAULT_REWARD_CENTER, sigma_r).numpy()
        u_masses = _empirical_square_masses(u)
        print(f"\nUnguided ({len(u)} samples):")
        print(f"  Mean reward:    {u_rewards.mean():.4f}")

    # Analytic tilt moments
    analytic_mean, analytic_cov, analytic_mean_reward = _analytic_tilt_moments(lam, sigma_r)
    print(f"\nAnalytic tilt (λ={lam}, σ_r={sigma_r}):")
    print(f"  Mean reward:    {analytic_mean_reward:.4f}")
    print(f"  Mean position:  ({analytic_mean[0]:.3f}, {analytic_mean[1]:.3f})")
    print(f"  Covariance:")
    print(f"    [{analytic_cov[0,0]:7.4f}  {analytic_cov[0,1]:7.4f}]")
    print(f"    [{analytic_cov[1,0]:7.4f}  {analytic_cov[1,1]:7.4f}]")

    # Guided
    if os.path.exists(path):
        data = np.load(path)
        g = data["samples"]
        g_rewards = data["rewards"] if "rewards" in data else None
        if g_rewards is None:
            g_tensor = torch.tensor(g, dtype=torch.float32)
            g_rewards = reward_fn(g_tensor, DEFAULT_REWARD_CENTER, sigma_r).numpy()
        g_masses = _empirical_square_masses(g)
        g_valid = g[~np.isnan(g).any(axis=1)]
        g_cov = np.cov(g_valid.T)
        damp_str = f", σ_damp={sigma_damp}" if sigma_damp is not None else ""
        label = f"Guided k={k}" + (f", best-of-{n}" if n > 1 else "") + damp_str
        print(f"\n{label} ({len(g_valid)}/{len(g)} valid samples):")
        print(f"  Mean reward:    {np.nanmean(g_rewards):.4f}")
        print(f"  Mean position:  ({g_valid.mean(0)[0]:.3f}, {g_valid.mean(0)[1]:.3f})")
        print(f"  Covariance:")
        print(f"    [{g_cov[0,0]:7.4f}  {g_cov[0,1]:7.4f}]")
        print(f"    [{g_cov[1,0]:7.4f}  {g_cov[1,1]:7.4f}]")

        # Per-square comparison
        print(f"\nPer-square mass (top 6 by analytic mass):")
        print(f"  {'Square':>10s}  {'Analytic':>8s}  {'Guided':>8s}  {'Unguided':>8s}")
        sorted_squares = sorted(analytic_masses.keys(),
                                key=lambda sq: analytic_masses[sq], reverse=True)
        for sq in sorted_squares[:6]:
            i, j = sq
            a = analytic_masses[sq]
            gm = g_masses.get(sq, 0.0)
            um = u_masses.get(sq, 0.0) if u_masses is not None else float('nan')
            print(f"  ({i-3},{j-3})-({i-2},{j-2})  {a:8.4f}  {gm:8.4f}  {um:8.4f}")
    else:
        print(f"\nNo guided samples found at {path}. Run sample.py first.")

    print("=" * 60)


def plot_diagnostics(output_dir, image_dir, k, n, lam, sigma_r, sigma_damp=None):
    """Bar chart comparing per-square mass: guided vs analytic tilt."""
    path = _guided_data_path(output_dir, k, n, lam, sigma_damp)

    if not os.path.exists(path):
        print(f"No guided samples at {path}. Run sample.py first.")
        return

    g = np.load(path)["samples"]
    g_masses = _empirical_square_masses(g)
    analytic_masses = _analytic_square_masses(lam, sigma_r)

    # Sort squares by analytic mass
    squares = sorted(analytic_masses.keys(), key=lambda sq: analytic_masses[sq], reverse=True)
    labels = [f"({i-3},{j-3})" for i, j in squares]
    analytic_vals = [analytic_masses[sq] for sq in squares]
    guided_vals = [g_masses.get(sq, 0.0) for sq in squares]

    x_pos = np.arange(len(squares))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(x_pos - width / 2, analytic_vals, width, label="Analytic tilt")
    ax.bar(x_pos + width / 2, guided_vals, width, label=f"Guided $k={k}$")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mass")
    ax.set_xlabel("Square (lower-left corner)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(image_dir, f"square_masses_k{k}_n{n}.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Lambda sweep
# ---------------------------------------------------------------------------


def plot_damp_sweep(output_dir, sweep_dir, k, n, lam, sigma_damp_values, sigma_r):
    """Plot mean reward vs sigma_damp, comparing guided samples to analytic tilt."""
    center = DEFAULT_REWARD_CENTER

    # Analytic target (independent of sigma_damp)
    _, _, analytic_mr = _analytic_tilt_moments(lam, sigma_r)
    analytic_mean, analytic_cov, _ = _analytic_tilt_moments(lam, sigma_r)
    center_np = center.numpy()

    guided_rewards = []
    guided_dists = []
    guided_vars = []
    valid_sigmas = []

    for sd in sigma_damp_values:
        path = _guided_data_path(output_dir, k, n, lam, sigma_damp=sd)
        if not os.path.exists(path):
            print(f"  No data for σ_damp={sd} at {path}")
            continue
        data = np.load(path)
        g = data["samples"]
        g_valid = g[~np.isnan(g).any(axis=1)]
        if "rewards" in data:
            mr = np.nanmean(data["rewards"])
        else:
            g_tensor = torch.tensor(g, dtype=torch.float32)
            mr = np.nanmean(reward_fn(g_tensor, center, sigma_r).numpy())
        guided_rewards.append(mr)
        guided_dists.append(np.linalg.norm(np.nanmean(g, axis=0) - center_np))
        guided_vars.append(np.trace(np.cov(g_valid.T)) / 2)
        valid_sigmas.append(sd)

    if not valid_sigmas:
        print("No data for sigma_damp sweep.")
        return

    # Also include undamped (sigma_damp=None) if available
    undamped_path = _guided_data_path(output_dir, k, n, lam, sigma_damp=None)
    undamped_mr = None
    undamped_dist = None
    undamped_var = None
    if os.path.exists(undamped_path):
        data = np.load(undamped_path)
        g = data["samples"]
        g_valid = g[~np.isnan(g).any(axis=1)]
        if "rewards" in data:
            undamped_mr = np.nanmean(data["rewards"])
        else:
            g_tensor = torch.tensor(g, dtype=torch.float32)
            undamped_mr = np.nanmean(reward_fn(g_tensor, center, sigma_r).numpy())
        undamped_dist = np.linalg.norm(np.nanmean(g, axis=0) - center_np)
        undamped_var = np.trace(np.cov(g_valid.T)) / 2

    # Mean reward vs sigma_damp
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(valid_sigmas, guided_rewards, "s-", label=f"Guided $k={k}$", markersize=4)
    ax.axhline(analytic_mr, color="C0", ls="--", label="Analytic tilt")
    if undamped_mr is not None:
        ax.axhline(undamped_mr, color="C2", ls=":", label="Undamped")
    ax.set_xlabel(r"$\sigma_{\mathrm{damp}}$")
    ax.set_ylabel("Mean reward")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(sweep_dir, "reward_vs_damp.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    # Mean position distance vs sigma_damp
    analytic_dist = np.linalg.norm(analytic_mean - center_np)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(valid_sigmas, guided_dists, "s-", label=f"Guided $k={k}$", markersize=4)
    ax.axhline(analytic_dist, color="C0", ls="--", label="Analytic tilt")
    if undamped_dist is not None:
        ax.axhline(undamped_dist, color="C2", ls=":", label="Undamped")
    ax.set_xlabel(r"$\sigma_{\mathrm{damp}}$")
    ax.set_ylabel(r"$\|\mathrm{mean} - c\|$")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(sweep_dir, "dist_vs_damp.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    # Mean eigenvalue of covariance vs sigma_damp
    analytic_var = np.trace(analytic_cov) / 2
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(valid_sigmas, guided_vars, "s-", label=f"Guided $k={k}$", markersize=4)
    ax.axhline(analytic_var, color="C0", ls="--", label="Analytic tilt")
    if undamped_var is not None:
        ax.axhline(undamped_var, color="C2", ls=":", label="Undamped")
    ax.set_xlabel(r"$\sigma_{\mathrm{damp}}$")
    ax.set_ylabel(r"Mean eigenvalue of $\Sigma$")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(sweep_dir, "var_vs_damp.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_lam_sweep(output_dir, sweep_dir, k, n, lam_values, sigma_r):
    """Plot mean reward vs lambda, comparing guided samples to analytic tilt."""
    analytic_rewards = []
    guided_rewards = []
    valid_lams = []

    center = DEFAULT_REWARD_CENTER

    for lam in lam_values:
        # Analytic
        _, _, analytic_mr = _analytic_tilt_moments(lam, sigma_r)
        analytic_rewards.append(analytic_mr)

        # Guided
        path = _guided_data_path(output_dir, k, n, lam)
        if os.path.exists(path):
            data = np.load(path)
            g = data["samples"]
            if "rewards" in data:
                mr = np.nanmean(data["rewards"])
            else:
                g_tensor = torch.tensor(g, dtype=torch.float32)
                mr = np.nanmean(reward_fn(g_tensor, center, sigma_r).numpy())
            guided_rewards.append(mr)
            valid_lams.append(lam)
        else:
            print(f"  No data for λ={lam} at {path}")

    # Mean reward vs lambda
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(lam_values, analytic_rewards, "o-", label="Analytic tilt", markersize=4)
    if valid_lams:
        ax.plot(valid_lams, guided_rewards, "s--", label=f"Guided $k={k}$", markersize=4)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Mean reward")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(sweep_dir, "reward_vs_lam.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    # Mean position distance and variance vs lambda
    if valid_lams:
        center_np = center.numpy()
        analytic_dists = []
        analytic_vars = []
        guided_dists = []
        guided_vars = []
        for lam in valid_lams:
            a_mean, a_cov, _ = _analytic_tilt_moments(lam, sigma_r)
            analytic_dists.append(np.linalg.norm(a_mean - center_np))
            analytic_vars.append(np.trace(a_cov) / 2)

            path = _guided_data_path(output_dir, k, n, lam)
            g = np.load(path)["samples"]
            g_valid = g[~np.isnan(g).any(axis=1)]
            guided_dists.append(np.linalg.norm(np.nanmean(g, axis=0) - center_np))
            guided_vars.append(np.trace(np.cov(g_valid.T)) / 2)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(valid_lams, analytic_dists, "o-", label="Analytic tilt", markersize=4)
        ax.plot(valid_lams, guided_dists, "s--", label=f"Guided $k={k}$", markersize=4)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\|\mathrm{mean} - c\|$")
        ax.legend()
        fig.tight_layout()
        out = os.path.join(sweep_dir, "dist_vs_lam.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(valid_lams, analytic_vars, "o-", label="Analytic tilt", markersize=4)
        ax.plot(valid_lams, guided_vars, "s--", label=f"Guided $k={k}$", markersize=4)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"Mean eigenvalue of $\Sigma$")
        ax.legend()
        fig.tight_layout()
        out = os.path.join(sweep_dir, "var_vs_lam.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Best-of-n paper plots
# ---------------------------------------------------------------------------


def _load_guided(output_dir, k, lam):
    """Load guided samples as torch tensors, return (samples, rewards) or (None, None)."""
    path = os.path.join(output_dir, f"guided_k{k}_lam{lam}.npz")
    if not os.path.exists(path):
        print(f"No guided data at {path}")
        return None, None
    data = np.load(path)
    return torch.tensor(data["samples"]), torch.tensor(data["rewards"])


def plot_bon_scatter(output_dir, image_dir, k, lam, n, sigma_r, seed=0):
    """Scatter plot of best-of-n guided samples."""
    samples, rewards = _load_guided(output_dir, k, lam)
    if samples is None:
        return
    torch.manual_seed(seed)
    selected, _ = best_of_n(samples, rewards, n)
    selected = selected.numpy()

    if len(selected) > DISPLAY_N:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(selected), DISPLAY_N, replace=False)
        selected = selected[idx]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    _checkerboard_background(ax)
    ax.scatter(selected[:, 0], selected[:, 1], s=2, alpha=0.5, c="C1", edgecolors="none")
    center = DEFAULT_REWARD_CENTER.numpy()
    ax.scatter(*center, s=80, c="red", marker="*", zorder=5)
    ax.set_xlim(*GRID_RANGE)
    ax.set_ylim(*GRID_RANGE)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    out = os.path.join(image_dir, f"bon_n{n}.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_analytic_tilt_samples_named(output_dir, image_dir, lam, sigma_r, n_samples=5000,
                                     outname="analytic_tilt_samples.pdf", seed=0,
                                     max_display=None):
    """Scatter plot of exact samples from the analytic tilted density, saved under outname."""
    np.random.seed(seed)
    cache = os.path.join(output_dir, f"analytic_tilt_lam{lam}.npz")
    if os.path.exists(cache):
        samples = np.load(cache)["samples"]
        print(f"Loaded {len(samples)} analytic tilt samples from cache.")
    else:
        print(f"Sampling {n_samples} from analytic tilt (λ={lam})...")
        samples = sample_analytic_tilt(n_samples, lam, sigma_r)
        np.savez(cache, samples=samples)
        print(f"Cached to {cache}")

    if max_display is not None and len(samples) > max_display:
        idx = np.random.choice(len(samples), max_display, replace=False)
        samples = samples[idx]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    _checkerboard_background(ax)
    ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5, c="C2", edgecolors="none")
    center_np = DEFAULT_REWARD_CENTER.numpy()
    ax.scatter(*center_np, s=80, c="red", marker="*", zorder=5)
    ax.set_xlim(*GRID_RANGE)
    ax.set_ylim(*GRID_RANGE)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    out = os.path.join(image_dir, outname)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_guided_named(output_dir, image_dir, k, lam, outname="guided_k1.pdf",
                       seed=0):
    """Scatter plot of k=1 guided samples (n=1), saved under outname."""
    path = os.path.join(output_dir, f"guided_k{k}_lam{lam}.npz")
    if not os.path.exists(path):
        print(f"No cached data at {path}.")
        return
    samples = np.load(path)["samples"]

    if len(samples) > DISPLAY_N:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(samples), DISPLAY_N, replace=False)
        samples = samples[idx]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    _checkerboard_background(ax)
    ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5, c="C1", edgecolors="none")
    center = DEFAULT_REWARD_CENTER.numpy()
    ax.scatter(*center, s=80, c="red", marker="*", zorder=5)
    ax.set_xlim(*GRID_RANGE)
    ax.set_ylim(*GRID_RANGE)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    out = os.path.join(image_dir, outname)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_bon_softmax_reward_vs_n(output_dir, image_dir, k, lam, sigma_r,
                                  n_values=None, seed=0):
    """Mean reward vs n for best-of-n and softmax-of-n, with analytic tilt reference."""
    if n_values is None:
        n_values = [1, 2, 4, 8, 16, 32]

    samples, rewards = _load_guided(output_dir, k, lam)
    if samples is None:
        return
    torch.manual_seed(seed)

    _, _, analytic_mr = _analytic_tilt_moments(lam, sigma_r)

    bon_mrs = []
    softmax_mrs = []
    valid_n = []
    for n in n_values:
        if n > samples.shape[0]:
            break
        s_bon, r_bon = best_of_n(samples, rewards, n)
        bon_mrs.append(float(torch.nanmean(r_bon).item()))
        s_soft, r_soft = softmax_of_n(samples, rewards, n, lam)
        softmax_mrs.append(float(torch.nanmean(r_soft).item()))
        valid_n.append(n)

    fig, ax = plt.subplots()
    ax.plot(valid_n, bon_mrs, "o-", label=r"Best-of-$n$", markersize=5)
    ax.plot(valid_n, softmax_mrs, "s--", label=r"Softmax-of-$n$", markersize=5)
    ax.axhline(analytic_mr, color="C2", ls=":", lw=1.5, label="Analytic tilt")
    ax.set_xlabel("$n$")
    ax.set_ylabel("Mean reward")
    ax.set_xscale("log", base=2)
    ax.set_xticks(valid_n)
    ax.set_xticklabels([str(n) for n in valid_n])
    ax.legend()
    out = os.path.join(image_dir, "bon_reward_vs_n.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def print_cov_table(output_dir, k, lam, sigma_r):
    """Print covariance trace and mean reward for guided (n=1) vs analytic tilt."""
    samples, rewards = _load_guided(output_dir, k, lam)
    if samples is None:
        return
    samples_np = samples.numpy()
    valid = samples_np[~np.isnan(samples_np).any(axis=1)]
    cov = np.cov(valid.T)
    print(f"k={k} guided (lam={lam}):")
    print(f"  Mean reward:   {rewards.mean().item():.4f}")
    print(f"  Cov trace:     {np.trace(cov):.4f}")
    print(f"  Cov:\n    [{cov[0,0]:.4f}  {cov[0,1]:.4f}]\n    [{cov[1,0]:.4f}  {cov[1,1]:.4f}]")

    analytic_mean, analytic_cov, analytic_mr = _analytic_tilt_moments(lam, sigma_r)
    print(f"Analytic tilt (lam={lam}):")
    print(f"  Mean reward:   {analytic_mr:.4f}")
    print(f"  Cov trace:     {np.trace(analytic_cov):.4f}")
    print(f"  Cov:\n    [{analytic_cov[0,0]:.4f}  {analytic_cov[0,1]:.4f}]\n    [{analytic_cov[1,0]:.4f}  {analytic_cov[1,1]:.4f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_dir(image_root, lam, sigma_damp=None):
    """Per-run subdirectory under images/guided/."""
    name = f"lam_{lam}"
    if sigma_damp is not None:
        name += f"_damp_{sigma_damp}"
    return os.path.join(image_root, "guided", name)


def plot_bon_damp_sweep_stats(output_dir, sweep_dir, k, lam, n, sigma_damps, sigma_r, seed=0):
    """Mean reward, distance to center, and mean eigenvalue of cov vs sigma_damp for best-of-n."""
    torch.manual_seed(seed)
    center = DEFAULT_REWARD_CENTER
    center_np = center.numpy()
    analytic_mean, analytic_cov, analytic_mr = _analytic_tilt_moments(lam, sigma_r)
    analytic_dist = np.linalg.norm(analytic_mean - center_np)
    analytic_var = np.trace(analytic_cov) / 2

    # Undamped best-of-n baseline
    undamped_path = os.path.join(output_dir, f"guided_k{k}_lam{lam}.npz")
    undamped_mr = undamped_dist = undamped_var = None
    if os.path.exists(undamped_path):
        data = np.load(undamped_path)
        s, r = torch.tensor(data["samples"]), torch.tensor(data["rewards"])
        sel, sel_r = best_of_n(s, r, n)
        sel = sel.numpy()
        sel_valid = sel[~np.isnan(sel).any(axis=1)]
        undamped_mr = float(torch.nanmean(sel_r).item())
        undamped_dist = float(np.linalg.norm(np.nanmean(sel, axis=0) - center_np))
        undamped_var = float(np.trace(np.cov(sel_valid.T)) / 2)

    valid_sigmas, bon_rewards, bon_dists, bon_vars = [], [], [], []
    for sd in sigma_damps:
        path = os.path.join(output_dir, f"guided_k{k}_lam{lam}_damp{sd}.npz")
        if not os.path.exists(path):
            print(f"  No data for σ_damp={sd} at {path}")
            continue
        data = np.load(path)
        s, r = torch.tensor(data["samples"]), torch.tensor(data["rewards"])
        sel, sel_r = best_of_n(s, r, n)
        sel = sel.numpy()
        sel_valid = sel[~np.isnan(sel).any(axis=1)]
        valid_sigmas.append(sd)
        bon_rewards.append(float(torch.nanmean(sel_r).item()))
        bon_dists.append(float(np.linalg.norm(np.nanmean(sel, axis=0) - center_np)))
        bon_vars.append(float(np.trace(np.cov(sel_valid.T)) / 2))

    if not valid_sigmas:
        print("No data for bon damp sweep stats.")
        return

    os.makedirs(sweep_dir, exist_ok=True)

    for values, ylabel, fname in [
        (bon_rewards, "Mean reward",               "reward_vs_damp.pdf"),
        (bon_dists,   r"$\|\mathrm{mean} - c\|$",  "dist_vs_damp.pdf"),
        (bon_vars,    r"Mean eigenvalue of $\Sigma$", "var_vs_damp.pdf"),
    ]:
        ref = analytic_mr if fname == "reward_vs_damp.pdf" else \
              analytic_dist if fname == "dist_vs_damp.pdf" else analytic_var
        undamped_ref = undamped_mr if fname == "reward_vs_damp.pdf" else \
                       undamped_dist if fname == "dist_vs_damp.pdf" else undamped_var
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(valid_sigmas, values, "s-", label=f"Best-of-{n} (damped)", markersize=4)
        ax.axhline(ref, color="C0", ls="--", label="Analytic tilt")
        if undamped_ref is not None:
            ax.axhline(undamped_ref, color="C2", ls=":", label=f"Best-of-{n} (undamped)")
        ax.set_xlabel(r"$\sigma_{\mathrm{damp}}$")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(sweep_dir, fname)
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


def plot_bon_damping_sweep(output_dir, image_dir, k, lam, n, sigma_damps, seed=0):
    """Scatter plots of best-of-n guided samples for a sweep of sigma_damp values."""
    torch.manual_seed(seed)
    center = DEFAULT_REWARD_CENTER.numpy()
    for sigma_damp in sigma_damps:
        path = os.path.join(output_dir, f"guided_k{k}_lam{lam}_damp{sigma_damp}.npz")
        if not os.path.exists(path):
            print(f"No cached data at {path}. Run sample.py first.")
            continue
        data = np.load(path)
        samples = torch.tensor(data["samples"])
        rewards = torch.tensor(data["rewards"])
        selected, _ = best_of_n(samples, rewards, n)
        selected = selected.numpy()

        if len(selected) > DISPLAY_N:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(selected), DISPLAY_N, replace=False)
            selected = selected[idx]

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        _checkerboard_background(ax)
        ax.scatter(selected[:, 0], selected[:, 1], s=2, alpha=0.5, c="C1", edgecolors="none")
        ax.scatter(*center, s=80, c="red", marker="*", zorder=5)
        ax.set_xlim(*GRID_RANGE)
        ax.set_ylim(*GRID_RANGE)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_aspect("equal")
        out = os.path.join(image_dir, f"bon_n{n}_damp{sigma_damp}.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot checkerboard guidance results")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--image-root", type=str, default="./images")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--lam", type=float, default=5.0)
    parser.add_argument("--sigma-r", type=float, default=1.5)
    parser.add_argument("--plots", type=str, nargs="+",
                        default=["unguided", "guided", "analytic_samples", "tilt", "tilt3d", "reward", "diagnostics"],
                        help="Which plots to generate: also supports 'bon_paper' for best-of-n paper figures")
    parser.add_argument("--sigma-damp", type=float, default=None,
                        help="Reward damping sigma (None = no damping)")
    parser.add_argument("--lam-sweep", type=float, nargs="+", default=None,
                        help="Lambda values for sweep experiment (e.g. --lam-sweep 0.5 1.0 2.0 5.0 10.0)")
    parser.add_argument("--sigma-damp-sweep", type=float, nargs="+", default=None,
                        help="Sigma_damp values for sweep (e.g. --sigma-damp-sweep 0.1 0.5 1.0 2.0 5.0)")
    args = parser.parse_args()

    base_dir = os.path.join(args.image_root, "base")
    guided_dir = _run_dir(args.image_root, args.lam, args.sigma_damp)
    sweep_dir = os.path.join(args.image_root, "guided", "sweep")
    reward_dir = os.path.join(args.image_root, "guided")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(guided_dir, exist_ok=True)

    if "unguided" in args.plots:
        plot_unguided(args.output_dir, base_dir)
    if "guided" in args.plots:
        plot_guided(args.output_dir, guided_dir, args.k, args.n, args.lam, args.sigma_damp)
    if "analytic_samples" in args.plots:
        plot_analytic_tilt_samples(args.output_dir, guided_dir, args.lam, args.sigma_r)
    if "tilt" in args.plots:
        plot_analytic_tilt(guided_dir, args.lam, args.sigma_r)
    if "tilt3d" in args.plots:
        plot_analytic_tilt_3d(guided_dir, args.lam, args.sigma_r)
    if "reward" in args.plots:
        plot_reward(reward_dir, args.sigma_r)
    if "diagnostics" in args.plots:
        compute_diagnostics(args.output_dir, args.k, args.n, args.lam, args.sigma_r, args.sigma_damp)
        plot_diagnostics(args.output_dir, guided_dir, args.k, args.n, args.lam, args.sigma_r, args.sigma_damp)

    if "bon_paper" in args.plots:
        bon_dir = os.path.join(args.image_root, "bon")
        os.makedirs(bon_dir, exist_ok=True)
        print_cov_table(args.output_dir, args.k, args.lam, args.sigma_r)
        plot_analytic_tilt_samples_named(args.output_dir, bon_dir, args.lam, args.sigma_r,
                                         outname="analytic_tilt_samples.pdf", max_display=DISPLAY_N)
        plot_guided_named(args.output_dir, bon_dir, args.k, args.lam,
                            outname=f"guided_k{args.k}.pdf")
        for n_val in [2, 3, 4]:
            plot_bon_scatter(args.output_dir, bon_dir, args.k, args.lam, n_val, args.sigma_r)
            plot_analytic_tilt_samples_named(args.output_dir, bon_dir, args.lam, args.sigma_r,
                                             outname=f"analytic_tilt_samples_n{n_val}.pdf",
                                             max_display=DISPLAY_N)
        plot_bon_softmax_reward_vs_n(args.output_dir, bon_dir, args.k, args.lam, args.sigma_r)

    if "bon_damp_sweep" in args.plots:
        bon_dir = os.path.join(args.image_root, "bon")
        os.makedirs(bon_dir, exist_ok=True)
        sigma_damps = [0.1, 0.2, 0.5, 1.0]
        for n_val in [2, 3, 4]:
            bon_sweep_dir = os.path.join(bon_dir, f"sweep_n{n_val}")
            plot_bon_damping_sweep(args.output_dir, bon_dir, args.k, args.lam, n=n_val,
                                   sigma_damps=sigma_damps)
            plot_bon_damp_sweep_stats(args.output_dir, bon_sweep_dir, args.k, args.lam, n=n_val,
                                      sigma_damps=sigma_damps, sigma_r=args.sigma_r)

    # Lambda sweep
    if args.lam_sweep:
        os.makedirs(sweep_dir, exist_ok=True)
        for lam_val in args.lam_sweep:
            run_dir = _run_dir(args.image_root, lam_val)
            os.makedirs(run_dir, exist_ok=True)
            if "guided" in args.plots:
                plot_guided(args.output_dir, run_dir, args.k, args.n, lam_val)
            if "analytic_samples" in args.plots:
                plot_analytic_tilt_samples(args.output_dir, run_dir, lam_val, args.sigma_r)
            if "tilt" in args.plots:
                plot_analytic_tilt(run_dir, lam_val, args.sigma_r)
            if "diagnostics" in args.plots:
                compute_diagnostics(args.output_dir, args.k, args.n, lam_val, args.sigma_r)
                plot_diagnostics(args.output_dir, run_dir, args.k, args.n, lam_val, args.sigma_r)
        plot_lam_sweep(args.output_dir, sweep_dir, args.k, args.n, args.lam_sweep, args.sigma_r)

    # Sigma damping sweep
    if args.sigma_damp_sweep:
        os.makedirs(sweep_dir, exist_ok=True)
        for sd in args.sigma_damp_sweep:
            run_dir = _run_dir(args.image_root, args.lam, sigma_damp=sd)
            os.makedirs(run_dir, exist_ok=True)
            if "guided" in args.plots:
                plot_guided(args.output_dir, run_dir, args.k, args.n, args.lam, sigma_damp=sd)
            if "diagnostics" in args.plots:
                compute_diagnostics(args.output_dir, args.k, args.n, args.lam, args.sigma_r, sigma_damp=sd)
                plot_diagnostics(args.output_dir, run_dir, args.k, args.n, args.lam, args.sigma_r, sigma_damp=sd)
        plot_damp_sweep(args.output_dir, sweep_dir, args.k, args.n, args.lam, args.sigma_damp_sweep, args.sigma_r)
