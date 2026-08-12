"""Demo: Feature-Space Second-Order Reward Guidance.

Compares three levels of approximation on a 2D toy problem:
  1. First-order  (mean shift only)
  2. Full second-order (Laplace with D×D Hessian)
  3. Feature-space second-order (Gauss-Newton with d×d Hessian)

Also benchmarks computational cost as D grows (scalability test).

Usage:
    python demo.py                    # Run 2D visualization
    python demo.py --scalability      # Run scalability benchmark
    python demo.py --all              # Run both
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_second_order.core import (
    DecomposableReward,
    exact_tilted_log_density,
    feature_space_second_order_correction,
    first_order_correction,
    full_second_order_correction,
    sample_from_correction,
)


# ============================================================
# Reward Models for Demo
# ============================================================


class ToyFeatureExtractor(nn.Module):
    """f: R^D → R^d, a small MLP with bottleneck.

    For the 2D demo: D=2 → hidden → d=4
    For scalability: D=large → hidden → d=small
    """

    def __init__(self, input_dim, feature_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x):
        return self.net(x)


class ToyRewardHead(nn.Module):
    """g: R^d → R, a simple quadratic-like head.

    g(z) = -||z - target||² + linear_term
    This makes H_g = -2I (constant, negative definite) → reward is concave.
    """

    def __init__(self, feature_dim):
        super().__init__()
        # Learnable target in feature space
        self.target = nn.Parameter(torch.randn(feature_dim) * 0.5)
        self.linear = nn.Linear(feature_dim, 1, bias=False)
        # Initialize to small values
        nn.init.normal_(self.linear.weight, std=0.1)

    def forward(self, z):
        quad = -torch.sum((z - self.target) ** 2, dim=-1)
        lin = self.linear(z).squeeze(-1)
        return quad + lin


class NonlinearRewardHead(nn.Module):
    """g: R^d → R, a nonlinear head (MLP).

    This creates a non-trivial H_g that varies with z,
    making the feature-space correction more interesting.
    """

    def __init__(self, feature_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


# ============================================================
# 2D Visualization
# ============================================================


def run_2d_demo(save_dir="figures"):
    """Run the 2D demo comparing all three approximation levels."""
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(42)

    D = 2  # input dimension
    d = 4  # feature dimension

    # Build decomposable reward r(x) = g(f(x))
    f_net = ToyFeatureExtractor(D, d, hidden_dim=32)
    g_net = NonlinearRewardHead(d, hidden_dim=16)
    reward_model = DecomposableReward(f_net, g_net, feature_dim=d)

    # Freeze parameters (pretend this is a pre-trained reward model)
    for p in reward_model.parameters():
        p.requires_grad_(False)

    # Flow matching parameters (at some intermediate time t)
    mu_t = torch.tensor([[0.5, 0.3]])  # (1, 2) - conditional mean
    sigma_t_sq = 0.8                    # conditional variance
    beta_t = 2.0                        # reward weight

    print("=" * 60)
    print("2D Demo: Feature-Space Second-Order Reward Guidance")
    print("=" * 60)
    print(f"  Input dim D = {D}, Feature dim d = {d}")
    print(f"  μ_t = {mu_t.squeeze().tolist()}")
    print(f"  σ_t² = {sigma_t_sq}, β_t = {beta_t}")

    # ── Compute all corrections ──

    # Step 1: Gradient (needed by all methods)
    grad_r = reward_model.compute_reward_gradient(mu_t)  # (1, 2)
    print(f"\n  ∇r(μ_t) = {grad_r.squeeze().tolist()}")

    # Step 2: Full Hessian (only possible because D=2)
    full_H = reward_model.compute_full_hessian(mu_t)  # (1, 2, 2)
    print(f"  Full Hessian:\n{full_H.squeeze().numpy()}")

    # Step 3: Feature-space components
    z_t = reward_model.f(mu_t.detach())  # (1, d)
    J_f = reward_model.compute_feature_jacobian(mu_t)  # (1, d, D)
    H_g = reward_model.compute_head_hessian(z_t)       # (1, d, d)
    GN_H = reward_model.compute_gauss_newton_hessian(mu_t)  # (1, 2, 2)
    print(f"  Gauss-Newton Hessian:\n{GN_H.squeeze().detach().numpy()}")
    print(f"  Hessian difference (Full - GN):\n"
          f"{(full_H - GN_H).squeeze().detach().numpy()}")

    # Step 4: Apply corrections
    mean_1st, cov_1st = first_order_correction(mu_t, sigma_t_sq, beta_t, grad_r)
    mean_full, cov_full = full_second_order_correction(
        mu_t, sigma_t_sq, beta_t, grad_r, full_H
    )
    mean_feat, cov_feat = feature_space_second_order_correction(
        mu_t, sigma_t_sq, beta_t, grad_r, J_f, H_g
    )

    print(f"\n  ── Corrected means ──")
    print(f"  Original μ_t:       {mu_t.squeeze().tolist()}")
    print(f"  First-order:        {mean_1st.squeeze().tolist()}")
    print(f"  Full 2nd-order:     {mean_full.squeeze().detach().tolist()}")
    print(f"  Feature-space 2nd:  {mean_feat.squeeze().detach().tolist()}")

    # ── Visualization ──

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Grid for density evaluation
    lim = 4.0
    n_grid = 200
    xx = torch.linspace(-lim, lim, n_grid)
    yy = torch.linspace(-lim, lim, n_grid)
    X, Y = torch.meshgrid(xx, yy, indexing="ij")
    grid = torch.stack([X.flatten(), Y.flatten()], dim=-1)  # (G, 2)

    # (a) Exact tilted density
    with torch.no_grad():
        log_exact = exact_tilted_log_density(
            grid, mu_t.squeeze(), sigma_t_sq, beta_t, reward_model
        )
    exact_density = torch.exp(log_exact - log_exact.max()).reshape(n_grid, n_grid)

    # (b) First-order Gaussian
    diff_1st = grid - mean_1st  # (G, 2)
    cov_1st_inv = torch.linalg.inv(cov_1st.squeeze())
    log_1st = -0.5 * torch.sum(diff_1st @ cov_1st_inv * diff_1st, dim=-1)
    density_1st = torch.exp(log_1st - log_1st.max()).reshape(n_grid, n_grid)

    # (c) Full second-order Gaussian
    diff_full = grid - mean_full.detach()
    cov_full_inv = torch.linalg.inv(cov_full.squeeze().detach())
    log_full = -0.5 * torch.sum(diff_full @ cov_full_inv * diff_full, dim=-1)
    density_full = torch.exp(log_full - log_full.max()).reshape(n_grid, n_grid)

    # (d) Feature-space second-order Gaussian
    diff_feat = grid - mean_feat.detach()
    cov_feat_inv = torch.linalg.inv(cov_feat.squeeze().detach())
    log_feat = -0.5 * torch.sum(diff_feat @ cov_feat_inv * diff_feat, dim=-1)
    density_feat = torch.exp(log_feat - log_feat.max()).reshape(n_grid, n_grid)

    # Plot
    titles = [
        "(a) Exact tilted\n$p \\propto \\mathcal{N} \\cdot e^{\\beta r}$",
        "(b) First-order\n(mean shift only)",
        "(c) Full 2nd-order\n(Laplace, $D{\\times}D$ Hessian)",
        "(d) Feature-space 2nd\n(Gauss-Newton, $d{\\times}d$)",
    ]
    densities = [exact_density, density_1st, density_full, density_feat]
    means = [mu_t, mean_1st, mean_full, mean_feat]

    for ax, title, density, mean in zip(axes, titles, densities, means):
        ax.contourf(
            X.numpy(), Y.numpy(), density.numpy(),
            levels=30, cmap="Blues",
        )
        ax.contour(
            X.numpy(), Y.numpy(), density.numpy(),
            levels=10, colors="steelblue", linewidths=0.5, alpha=0.7,
        )
        # Mark original mean
        ax.plot(*mu_t.squeeze().tolist(), "r+", markersize=12, markeredgewidth=2,
                label="$\\mu_t$ (original)")
        # Mark corrected mean
        m = mean.squeeze().detach().tolist()
        ax.plot(*m, "k*", markersize=10, label="$\\mu_{new}$")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Comparing Reward Guidance Corrections\n"
        f"D={D}, d={d}, σ²_t={sigma_t_sq}, β_t={beta_t}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    path = os.path.join(save_dir, "feature_space_2nd_order_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved figure: {path}")
    plt.close()

    # ── KL divergence comparison ──
    # Approximate KL(approx || exact) by numerical integration on the grid
    dx = (2 * lim / n_grid) ** 2  # grid cell area

    exact_norm = (exact_density * dx).sum()
    p_exact = exact_density / exact_norm + 1e-30

    kl_results = {}
    for name, density in [("First-order", density_1st),
                          ("Full 2nd-order", density_full),
                          ("Feature-space 2nd", density_feat)]:
        q_norm = (density * dx).sum()
        q = density / q_norm + 1e-30
        # KL(p || q) = Σ p log(p/q)
        kl = (p_exact * torch.log(p_exact / q) * dx).sum().item()
        kl_results[name] = kl

    print(f"\n  ── KL divergence from exact ──")
    for name, kl in kl_results.items():
        print(f"  {name:25s}: KL = {kl:.6f}")

    return kl_results


# ============================================================
# Scalability Benchmark
# ============================================================


def run_scalability_benchmark(save_dir="figures"):
    """Benchmark computation time as input dimension D grows."""
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(42)

    d = 16  # fixed feature dimension (like a small CLIP bottleneck)
    dims = [2, 10, 50, 100, 500, 1000]
    n_repeats = 3

    results = {
        "D": [],
        "first_order_ms": [],
        "full_second_order_ms": [],
        "feature_space_ms": [],
    }

    print("\n" + "=" * 60)
    print("Scalability Benchmark: Time vs. Input Dimension D")
    print(f"Feature dimension d = {d} (fixed)")
    print("=" * 60)

    for D in dims:
        print(f"\n  D = {D} ...")

        f_net = ToyFeatureExtractor(D, d, hidden_dim=64)
        g_net = NonlinearRewardHead(d, hidden_dim=16)
        reward_model = DecomposableReward(f_net, g_net, feature_dim=d)
        for p in reward_model.parameters():
            p.requires_grad_(False)

        mu_t = torch.randn(1, D)
        sigma_t_sq = 0.5
        beta_t = 1.0

        # ── First-order ──
        times_1st = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            grad_r = reward_model.compute_reward_gradient(mu_t)
            mean_1st, cov_1st = first_order_correction(
                mu_t, sigma_t_sq, beta_t, grad_r
            )
            times_1st.append((time.perf_counter() - t0) * 1000)

        # ── Feature-space second-order ──
        times_feat = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            grad_r = reward_model.compute_reward_gradient(mu_t)
            z_t = reward_model.f(mu_t.detach())
            J_f = reward_model.compute_feature_jacobian(mu_t)
            H_g = reward_model.compute_head_hessian(z_t)
            mean_feat, cov_feat = feature_space_second_order_correction(
                mu_t, sigma_t_sq, beta_t, grad_r, J_f, H_g,
            )
            times_feat.append((time.perf_counter() - t0) * 1000)

        # ── Full second-order (skip if D > 500, too slow) ──
        if D <= 500:
            times_full = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                grad_r = reward_model.compute_reward_gradient(mu_t)
                full_H = reward_model.compute_full_hessian(mu_t)
                mean_full, cov_full = full_second_order_correction(
                    mu_t, sigma_t_sq, beta_t, grad_r, full_H,
                )
                times_full.append((time.perf_counter() - t0) * 1000)
            avg_full = np.mean(times_full)
        else:
            avg_full = float("nan")

        avg_1st = np.mean(times_1st)
        avg_feat = np.mean(times_feat)

        results["D"].append(D)
        results["first_order_ms"].append(avg_1st)
        results["full_second_order_ms"].append(avg_full)
        results["feature_space_ms"].append(avg_feat)

        full_str = f"{avg_full:.1f}ms" if not np.isnan(avg_full) else "SKIPPED (too slow)"
        print(f"    First-order:       {avg_1st:.1f}ms")
        print(f"    Full 2nd-order:    {full_str}")
        print(f"    Feature-space 2nd: {avg_feat:.1f}ms")

    # ── Plot ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(results["D"], results["first_order_ms"], "o-",
            color="#2ecc71", linewidth=2, markersize=6, label="First-order (gradient only)")
    ax.plot(results["D"], results["feature_space_ms"], "s-",
            color="#3498db", linewidth=2, markersize=6,
            label=f"Feature-space 2nd (d={d})")

    # Full second-order (only where available)
    valid_full = [(d_val, t) for d_val, t in
                  zip(results["D"], results["full_second_order_ms"])
                  if not np.isnan(t)]
    if valid_full:
        d_vals, t_vals = zip(*valid_full)
        ax.plot(d_vals, t_vals, "^-",
                color="#e74c3c", linewidth=2, markersize=6,
                label="Full 2nd-order (D×D Hessian)")

    ax.set_xlabel("Input Dimension D", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Scalability: Feature-Space vs. Full Second-Order",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "scalability_benchmark.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved figure: {path}")
    plt.close()


# ============================================================
# Approximation Error Analysis
# ============================================================


def run_error_analysis(save_dir="figures"):
    """Analyze how the approximation error varies with β_t and σ_t²."""
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(42)

    D = 2
    d = 4
    f_net = ToyFeatureExtractor(D, d, hidden_dim=32)
    g_net = NonlinearRewardHead(d, hidden_dim=16)
    reward_model = DecomposableReward(f_net, g_net, feature_dim=d)
    for p in reward_model.parameters():
        p.requires_grad_(False)

    mu_t = torch.tensor([[0.5, 0.3]])

    print("\n" + "=" * 60)
    print("Error Analysis: KL divergence vs. β_t")
    print("=" * 60)

    betas = np.linspace(0.1, 5.0, 20)
    sigma_t_sq = 0.8

    kls_1st, kls_full, kls_feat = [], [], []

    # Grid for KL computation
    lim = 5.0
    n_grid = 200
    xx = torch.linspace(-lim, lim, n_grid)
    yy = torch.linspace(-lim, lim, n_grid)
    X, Y = torch.meshgrid(xx, yy, indexing="ij")
    grid = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    dx = (2 * lim / n_grid) ** 2

    for beta_t in betas:
        beta_t = float(beta_t)

        # Exact
        with torch.no_grad():
            log_exact = exact_tilted_log_density(
                grid, mu_t.squeeze(), sigma_t_sq, beta_t, reward_model
            )
        exact_density = torch.exp(log_exact - log_exact.max()).reshape(n_grid, n_grid)
        exact_norm = (exact_density * dx).sum()
        p_exact = exact_density / exact_norm + 1e-30

        # Compute corrections
        grad_r = reward_model.compute_reward_gradient(mu_t)
        full_H = reward_model.compute_full_hessian(mu_t)
        z_t = reward_model.f(mu_t.detach())
        J_f = reward_model.compute_feature_jacobian(mu_t)
        H_g = reward_model.compute_head_hessian(z_t)

        for name, correction_fn, kl_list in [
            ("1st", lambda: first_order_correction(mu_t, sigma_t_sq, beta_t, grad_r), kls_1st),
            ("full", lambda: full_second_order_correction(mu_t, sigma_t_sq, beta_t, grad_r, full_H), kls_full),
            ("feat", lambda: feature_space_second_order_correction(mu_t, sigma_t_sq, beta_t, grad_r, J_f, H_g), kls_feat),
        ]:
            mean_c, cov_c = correction_fn()
            diff_c = grid - mean_c.detach()
            try:
                cov_inv = torch.linalg.inv(cov_c.squeeze().detach())
                log_q = -0.5 * torch.sum(diff_c @ cov_inv * diff_c, dim=-1)
                q_density = torch.exp(log_q - log_q.max()).reshape(n_grid, n_grid)
                q_norm = (q_density * dx).sum()
                q = q_density / q_norm + 1e-30
                kl = (p_exact * torch.log(p_exact / q) * dx).sum().item()
            except Exception:
                kl = float("nan")
            kl_list.append(kl)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(betas, kls_1st, "o-", color="#2ecc71", linewidth=2, label="First-order")
    ax.plot(betas, kls_full, "^-", color="#e74c3c", linewidth=2, label="Full 2nd-order (Laplace)")
    ax.plot(betas, kls_feat, "s-", color="#3498db", linewidth=2, label="Feature-space 2nd-order")
    ax.set_xlabel("β_t (reward weight)", fontsize=12)
    ax.set_ylabel("KL(exact || approx)", fontsize=12)
    ax.set_title("Approximation Error vs. Reward Weight β_t",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "error_vs_beta.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved figure: {path}")
    plt.close()


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature-Space 2nd-Order Demo")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--scalability", action="store_true",
                        help="Run scalability benchmark only")
    parser.add_argument("--error-analysis", action="store_true",
                        help="Run error analysis only")
    parser.add_argument("--save-dir", type=str, default="./figures/feature_second_order",
                        help="Directory to save figures")
    args = parser.parse_args()

    if args.scalability:
        run_scalability_benchmark(args.save_dir)
    elif args.error_analysis:
        run_error_analysis(args.save_dir)
    elif args.all:
        run_2d_demo(args.save_dir)
        run_scalability_benchmark(args.save_dir)
        run_error_analysis(args.save_dir)
    else:
        run_2d_demo(args.save_dir)
