"""Demo v2: Scenarios where second-order correction CLEARLY outperforms first-order.

The key insight: second-order correction changes the SHAPE of the distribution.
To see this, we need rewards with STRONG, ANISOTROPIC curvature — i.e., the
reward landscape is steep in some directions but flat in others.

Three scenarios:
  1. Anisotropic quadratic reward (analytical, crystal clear)
  2. Ridge reward via neural network (steep valley + flat plateau)
  3. High-dimensional with reward on a low-dim subspace

Usage:
    python3 -m feature_second_order.demo_v2
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_second_order.core import (
    DecomposableReward,
    exact_tilted_log_density,
    feature_space_second_order_correction,
    first_order_correction,
    first_order_damped_correction,
    full_second_order_correction,
)


# ============================================================
# Scenario 1: Anisotropic Quadratic Reward
# ============================================================
# r(x) = -λ₁(x₁ - t₁)² - λ₂(x₂ - t₂)²
# with λ₁ >> λ₂ (e.g., λ₁=10, λ₂=0.1)
#
# Hessian = diag(-2λ₁, -2λ₂) → very different eigenvalues!
#
# Exact tilted distribution:
#   N(μ,σ²I) · exp(βr) ∝ N with:
#     Σ_exact = diag(1/(σ⁻² + 2βλ₁), 1/(σ⁻² + 2βλ₂))
#
# First-order: keeps Σ = σ²I (WRONG — ignores anisotropy)
# Second-order: Σ = diag(1/(σ⁻² + 2βλ₁), 1/(σ⁻² + 2βλ₂)) (EXACT!)
# ============================================================


class AnisotropicQuadFeatureExtractor(nn.Module):
    """f(x) = x (identity). Feature space = input space."""
    def forward(self, x):
        return x


class AnisotropicQuadHead(nn.Module):
    """g(z) = -λ₁(z₁-t₁)² - λ₂(z₂-t₂)²

    Hessian H_g = diag(-2λ₁, -2λ₂)
    """
    def __init__(self, lambdas, target):
        super().__init__()
        self.register_buffer("lambdas", torch.tensor(lambdas, dtype=torch.float32))
        self.register_buffer("target", torch.tensor(target, dtype=torch.float32))

    def forward(self, z):
        diff = z - self.target
        return -torch.sum(self.lambdas * diff ** 2, dim=-1)


# ============================================================
# Scenario 2: Ridge Reward (Neural Net)
# ============================================================
# A reward that creates a "ridge" — steep drop-off in one direction
# (perpendicular to the ridge), flat along the ridge.
# This is a natural scenario for image rewards (e.g., "blueness"
# depends strongly on color channels but weakly on texture).


class RidgeFeatureExtractor(nn.Module):
    """Projects D-dim input onto a learned 1D direction (+ nonlinearity).

    f(x) = [tanh(w₁·x), w₂·x]  where w₁ is the "ridge normal"
    Feature dim d=2: one "steep" feature, one "flat" feature.
    """
    def __init__(self, input_dim):
        super().__init__()
        # Ridge normal (steep direction)
        w1 = torch.randn(input_dim)
        w1 = w1 / w1.norm()
        self.register_buffer("w1", w1)
        # Ridge tangent (flat direction)
        w2 = torch.randn(input_dim)
        w2 = w2 - (w2 @ w1) * w1  # orthogonalize
        w2 = w2 / w2.norm()
        self.register_buffer("w2", w2)

    def forward(self, x):
        z1 = torch.tanh(5.0 * (x @ self.w1))  # steep, saturating
        z2 = x @ self.w2                        # linear, flat
        return torch.stack([z1, z2], dim=-1)


class RidgeRewardHead(nn.Module):
    """g(z) = -α·z₁² - ε·z₂²

    α >> ε creates strong anisotropy in feature space.
    """
    def __init__(self, alpha=8.0, epsilon=0.05):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, z):
        return -(self.alpha * z[:, 0] ** 2 + self.epsilon * z[:, 1] ** 2)


# ============================================================
# Visualization helpers
# ============================================================


def compute_gaussian_density(grid, mean, cov, n_grid):
    """Evaluate Gaussian density on a grid."""
    diff = grid - mean.detach()
    try:
        cov_inv = torch.linalg.inv(cov.squeeze().detach())
        log_q = -0.5 * torch.sum(diff @ cov_inv * diff, dim=-1)
        return torch.exp(log_q - log_q.max()).reshape(n_grid, n_grid)
    except Exception:
        return torch.zeros(n_grid, n_grid)


def draw_ellipse(ax, mean, cov, color, label, linestyle="-", n_std=2.0):
    """Draw a covariance ellipse on the plot."""
    mean = mean.squeeze().detach().numpy()
    cov = cov.squeeze().detach().numpy()

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Ensure positive eigenvalues
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ellipse = patches.Ellipse(
        mean, width, height, angle=angle,
        fill=False, edgecolor=color, linewidth=2.5, linestyle=linestyle,
        label=label,
    )
    ax.add_patch(ellipse)
    ax.plot(*mean, marker="o", color=color, markersize=6)


def run_scenario_1(save_dir):
    """Anisotropic quadratic reward — the clearest demonstration."""
    print("\n" + "=" * 60)
    print("Scenario 1: Anisotropic Quadratic Reward")
    print("  r(x) = -10·(x₁-1)² - 0.1·(x₂-1)²")
    print("  Hessian = diag(-20, -0.2) → 100× anisotropy!")
    print("=" * 60)

    # Strong anisotropy: λ₁=10 (steep in x₁), λ₂=0.1 (flat in x₂)
    lambdas = [10.0, 0.1]
    target = [1.0, 1.0]

    f_net = AnisotropicQuadFeatureExtractor()
    g_net = AnisotropicQuadHead(lambdas, target)
    reward_model = DecomposableReward(f_net, g_net, feature_dim=2)

    mu_t = torch.tensor([[0.0, 0.0]])
    sigma_t_sq = 1.0
    beta_t = 0.3  # moderate reward weight

    # Compute corrections
    grad_r = reward_model.compute_reward_gradient(mu_t)
    full_H = reward_model.compute_full_hessian(mu_t)
    z_t = reward_model.f(mu_t.detach())
    J_f = reward_model.compute_feature_jacobian(mu_t)
    H_g = reward_model.compute_head_hessian(z_t)

    mean_1st, cov_1st = first_order_correction(mu_t, sigma_t_sq, beta_t, grad_r)
    mean_damp, cov_damp = first_order_damped_correction(mu_t, sigma_t_sq, beta_t, grad_r)
    mean_full, cov_full = full_second_order_correction(
        mu_t, sigma_t_sq, beta_t, grad_r, full_H
    )
    mean_feat, cov_feat = feature_space_second_order_correction(
        mu_t, sigma_t_sq, beta_t, grad_r, J_f, H_g
    )

    # Analytical exact covariance for quadratic reward
    sigma_exact = torch.diag(torch.tensor([
        1.0 / (1.0 / sigma_t_sq + 2 * beta_t * lambdas[0]),
        1.0 / (1.0 / sigma_t_sq + 2 * beta_t * lambdas[1]),
    ]))

    print(f"\n  Exact Σ (analytical):\n{sigma_exact.numpy()}")
    print(f"  First-order Σ (isotropic):\n{cov_1st.squeeze().detach().numpy()}")
    print(f"  Full 2nd-order Σ:\n{cov_full.squeeze().detach().numpy()}")
    print(f"  Feature-space Σ:\n{cov_feat.squeeze().detach().numpy()}")

    cov_1st_ratio = cov_1st.squeeze()[0, 0].item() / cov_1st.squeeze()[1, 1].item()
    cov_exact_ratio = sigma_exact[0, 0].item() / sigma_exact[1, 1].item()
    cov_feat_ratio = cov_feat.squeeze()[0, 0].item() / cov_feat.squeeze()[1, 1].item()
    print(f"\n  Σ₁₁/Σ₂₂ ratio (anisotropy measure):")
    print(f"    Exact:          {cov_exact_ratio:.4f}")
    print(f"    First-order:    {cov_1st_ratio:.4f}  ← WRONG (isotropic)")
    print(f"    Feature-space:  {cov_feat_ratio:.4f}  ← CORRECT")

    # ── Visualization ──
    fig, axes = plt.subplots(1, 5, figsize=(27.5, 5.5))
    lim = 3.5
    n_grid = 300
    xx = torch.linspace(-lim, lim, n_grid)
    yy = torch.linspace(-lim, lim, n_grid)
    X, Y = torch.meshgrid(xx, yy, indexing="ij")
    grid = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    # (a) Exact tilted
    with torch.no_grad():
        log_exact = exact_tilted_log_density(
            grid, mu_t.squeeze(), sigma_t_sq, beta_t, reward_model
        )
    exact_density = torch.exp(log_exact - log_exact.max()).reshape(n_grid, n_grid)

    densities = [
        exact_density,
        compute_gaussian_density(grid, mean_1st, cov_1st, n_grid),
        compute_gaussian_density(grid, mean_damp, cov_damp, n_grid),
        compute_gaussian_density(grid, mean_full, cov_full, n_grid),
        compute_gaussian_density(grid, mean_feat, cov_feat, n_grid),
    ]
    titles = [
        "(a) Exact tilted\n$p \\propto \\mathcal{N} \\cdot e^{\\beta r}$",
        "(b) First-order\n(circular — WRONG)",
        "(c) First-order damped\n(circular — WRONG shape)",
        "(d) Full 2nd-order\n(elliptical — CORRECT)",
        "(e) Feature-space 2nd\n(elliptical — CORRECT)",
    ]
    colors = ["#2c3e50", "#e74c3c", "#f39c12", "#27ae60", "#2980b9"]
    means = [mu_t, mean_1st, mean_damp, mean_full, mean_feat]
    covs = [sigma_exact.unsqueeze(0), cov_1st, cov_damp, cov_full, cov_feat]

    for ax, title, density, mean, cov, col in zip(axes, titles, densities, means, covs, colors):
        ax.contourf(X.numpy(), Y.numpy(), density.numpy(), levels=30, cmap="RdYlBu_r")
        ax.contour(X.numpy(), Y.numpy(), density.numpy(), levels=10,
                   colors="gray", linewidths=0.3, alpha=0.5)
        draw_ellipse(ax, mean, cov, col, "2σ ellipse", n_std=2.0)
        ax.plot(*target, "w*", markersize=14, markeredgecolor="black",
                markeredgewidth=1, label="reward target")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Anisotropic Reward: $r(x) = -10(x_1{-}1)^2 - 0.1(x_2{-}1)^2$\n"
        "First-order stays circular, second-order correctly elongates!",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "scenario1_anisotropic.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {path}")
    plt.close()

    # ── KL comparison ──
    dx = (2 * lim / n_grid) ** 2
    exact_norm = (exact_density * dx).sum()
    p_exact = exact_density / exact_norm + 1e-30

    print(f"\n  ── KL divergence from exact ──")
    for name, density in [("First-order", densities[1]),
                          ("First-order damped", densities[2]),
                          ("Full 2nd-order", densities[3]),
                          ("Feature-space 2nd", densities[4])]:
        q_norm = (density * dx).sum()
        q = density / q_norm + 1e-30
        kl = (p_exact * torch.log(p_exact / q) * dx).sum().item()
        print(f"  {name:25s}: KL = {kl:.6f}")


def run_scenario_2(save_dir):
    """Ridge reward — steep in one direction, flat in another."""
    print("\n" + "=" * 60)
    print("Scenario 2: Ridge Reward (Neural Network)")
    print("  Steep drop-off perpendicular to ridge, flat along ridge")
    print("=" * 60)

    torch.manual_seed(123)
    D = 2
    d = 2

    f_net = RidgeFeatureExtractor(D)
    g_net = RidgeRewardHead(alpha=8.0, epsilon=0.05)
    reward_model = DecomposableReward(f_net, g_net, feature_dim=d)
    for p in reward_model.parameters():
        p.requires_grad_(False)

    mu_t = torch.tensor([[0.0, 0.0]])
    sigma_t_sq = 1.0
    beta_t = 1.5

    # Compute
    grad_r = reward_model.compute_reward_gradient(mu_t)
    full_H = reward_model.compute_full_hessian(mu_t)
    z_t = reward_model.f(mu_t.detach())
    J_f = reward_model.compute_feature_jacobian(mu_t)
    H_g = reward_model.compute_head_hessian(z_t)

    print(f"  Ridge normal (w₁): {f_net.w1.numpy()}")
    print(f"  Ridge tangent (w₂): {f_net.w2.numpy()}")
    print(f"  Full Hessian eigenvalues: {torch.linalg.eigvalsh(full_H.squeeze()).numpy()}")
    print(f"  GN Hessian eigenvalues: {torch.linalg.eigvalsh((J_f.transpose(1,2) @ H_g @ J_f).squeeze().detach()).numpy()}")

    mean_1st, cov_1st = first_order_correction(mu_t, sigma_t_sq, beta_t, grad_r)
    mean_damp, cov_damp = first_order_damped_correction(mu_t, sigma_t_sq, beta_t, grad_r)
    mean_full, cov_full = full_second_order_correction(
        mu_t, sigma_t_sq, beta_t, grad_r, full_H
    )
    mean_feat, cov_feat = feature_space_second_order_correction(
        mu_t, sigma_t_sq, beta_t, grad_r, J_f, H_g
    )

    # Visualization
    fig, axes = plt.subplots(1, 5, figsize=(27.5, 5.5))
    lim = 3.0
    n_grid = 300
    xx = torch.linspace(-lim, lim, n_grid)
    yy = torch.linspace(-lim, lim, n_grid)
    X, Y = torch.meshgrid(xx, yy, indexing="ij")
    grid = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    with torch.no_grad():
        log_exact = exact_tilted_log_density(
            grid, mu_t.squeeze(), sigma_t_sq, beta_t, reward_model
        )
    exact_density = torch.exp(log_exact - log_exact.max()).reshape(n_grid, n_grid)

    densities = [
        exact_density,
        compute_gaussian_density(grid, mean_1st, cov_1st, n_grid),
        compute_gaussian_density(grid, mean_damp, cov_damp, n_grid),
        compute_gaussian_density(grid, mean_full, cov_full, n_grid),
        compute_gaussian_density(grid, mean_feat, cov_feat, n_grid),
    ]
    titles = [
        "(a) Exact tilted\n(elongated along ridge)",
        "(b) First-order\n(circular — WRONG)",
        "(c) First-order damped\n(circular — WRONG shape)",
        "(d) Full 2nd-order\n(elliptical)",
        "(e) Feature-space 2nd\n(elliptical — matches!)",
    ]
    means = [mu_t, mean_1st, mean_damp, mean_full, mean_feat]
    covs = [
        cov_full,  # approximate exact cov
        cov_1st, cov_damp, cov_full, cov_feat,
    ]
    colors = ["#2c3e50", "#e74c3c", "#f39c12", "#27ae60", "#2980b9"]

    for ax, title, density, mean, cov, col in zip(axes, titles, densities, means, covs, colors):
        ax.contourf(X.numpy(), Y.numpy(), density.numpy(), levels=30, cmap="RdYlBu_r")
        ax.contour(X.numpy(), Y.numpy(), density.numpy(), levels=10,
                   colors="gray", linewidths=0.3, alpha=0.5)
        draw_ellipse(ax, mean, cov, col, "2σ ellipse", n_std=2.0)

        # Draw ridge direction
        w2 = f_net.w2.numpy()
        ax.annotate("", xy=w2 * 2, xytext=-w2 * 2,
                    arrowprops=dict(arrowstyle="<->", color="white", lw=1.5,
                                    linestyle="--"))

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Ridge Reward: steep ⊥ to ridge, flat ∥ to ridge\n"
        "Second-order captures the elongation, first-order cannot",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "scenario2_ridge.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()

    # KL
    dx = (2 * lim / n_grid) ** 2
    exact_norm = (exact_density * dx).sum()
    p_exact = exact_density / exact_norm + 1e-30
    print(f"\n  ── KL divergence from exact ──")
    for name, density in [("First-order", densities[1]),
                          ("First-order damped", densities[2]),
                          ("Full 2nd-order", densities[3]),
                          ("Feature-space 2nd", densities[4])]:
        q_norm = (density * dx).sum()
        q = density / q_norm + 1e-30
        kl = (p_exact * torch.log(p_exact / q) * dx).sum().item()
        print(f"  {name:25s}: KL = {kl:.6f}")


def run_scenario_3(save_dir):
    """Sweep: KL improvement ratio (2nd-order / 1st-order) vs. anisotropy ratio."""
    print("\n" + "=" * 60)
    print("Scenario 3: KL improvement vs. anisotropy ratio")
    print("  λ₁ fixed at 5.0, λ₂ varies from 5.0 (isotropic) to 0.01 (100:1)")
    print("=" * 60)

    lim = 4.0
    n_grid = 200
    xx = torch.linspace(-lim, lim, n_grid)
    yy = torch.linspace(-lim, lim, n_grid)
    X, Y = torch.meshgrid(xx, yy, indexing="ij")
    grid = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    dx = (2 * lim / n_grid) ** 2

    mu_t = torch.tensor([[0.0, 0.0]])
    sigma_t_sq = 1.0
    beta_t = 0.3
    target = [1.0, 1.0]
    lambda_1 = 5.0

    ratios = np.logspace(0, 3, 30)  # λ₁/λ₂ from 1 to 1000
    kl_1st_list, kl_damp_list, kl_feat_list = [], [], []

    for ratio in ratios:
        lambda_2 = lambda_1 / ratio

        f_net = AnisotropicQuadFeatureExtractor()
        g_net = AnisotropicQuadHead([lambda_1, lambda_2], target)
        reward_model = DecomposableReward(f_net, g_net, feature_dim=2)

        grad_r = reward_model.compute_reward_gradient(mu_t)
        z_t = reward_model.f(mu_t.detach())
        J_f = reward_model.compute_feature_jacobian(mu_t)
        H_g = reward_model.compute_head_hessian(z_t)

        mean_1st, cov_1st = first_order_correction(mu_t, sigma_t_sq, beta_t, grad_r)
        mean_damp, cov_damp = first_order_damped_correction(mu_t, sigma_t_sq, beta_t, grad_r)
        mean_feat, cov_feat = feature_space_second_order_correction(
            mu_t, sigma_t_sq, beta_t, grad_r, J_f, H_g
        )

        # Exact
        with torch.no_grad():
            log_exact = exact_tilted_log_density(
                grid, mu_t.squeeze(), sigma_t_sq, beta_t, reward_model
            )
        exact_density = torch.exp(log_exact - log_exact.max()).reshape(n_grid, n_grid)
        exact_norm = (exact_density * dx).sum()
        p_exact = exact_density / exact_norm + 1e-30

        for name, mean_c, cov_c, kl_list in [
            ("1st", mean_1st, cov_1st, kl_1st_list),
            ("damp", mean_damp, cov_damp, kl_damp_list),
            ("feat", mean_feat, cov_feat, kl_feat_list),
        ]:
            density = compute_gaussian_density(grid, mean_c, cov_c, n_grid)
            q_norm = (density * dx).sum()
            q = density / q_norm + 1e-30
            kl = (p_exact * torch.log(p_exact / q) * dx).sum().item()
            kl_list.append(max(kl, 1e-10))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) KL divergence vs anisotropy
    ax1.plot(ratios, kl_1st_list, "o-", color="#e74c3c", linewidth=2.5,
             markersize=4, label="First-order", alpha=0.9)
    ax1.plot(ratios, kl_damp_list, "x-", color="#f39c12", linewidth=2.5,
             markersize=4, label="First-order damped", alpha=0.9)
    ax1.plot(ratios, kl_feat_list, "s-", color="#2980b9", linewidth=2.5,
             markersize=4, label="Feature-space 2nd-order", alpha=0.9)
    ax1.set_xlabel("Anisotropy ratio λ₁/λ₂", fontsize=13)
    ax1.set_ylabel("KL(exact ‖ approx)", fontsize=13)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("(a) KL Divergence vs. Reward Anisotropy", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # (b) Improvement ratio
    improvement = [kl1 / kl2 for kl1, kl2 in zip(kl_1st_list, kl_feat_list)]
    ax2.plot(ratios, improvement, "D-", color="#8e44ad", linewidth=2.5,
             markersize=4, alpha=0.9)
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="No improvement")
    ax2.set_xlabel("Anisotropy ratio λ₁/λ₂", fontsize=13)
    ax2.set_ylabel("KL(1st) / KL(2nd)  (higher = better)", fontsize=13)
    ax2.set_xscale("log")
    ax2.set_title("(b) Improvement Factor of 2nd-order over 1st-order", fontsize=13,
                  fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "When does second-order correction help?\n"
        "Answer: When the reward has strong anisotropic curvature (λ₁/λ₂ ≫ 1)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "scenario3_anisotropy_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()

    print(f"\n  At anisotropy 1:1   → improvement = {improvement[0]:.2f}×")
    print(f"  At anisotropy 100:1 → improvement = {improvement[len(improvement)//2]:.2f}×")
    print(f"  At anisotropy 1000:1→ improvement = {improvement[-1]:.2f}×")


if __name__ == "__main__":
    save_dir = "./figures/feature_second_order"
    os.makedirs(save_dir, exist_ok=True)

    run_scenario_1(save_dir)
    run_scenario_2(save_dir)
    run_scenario_3(save_dir)

    print("\n" + "=" * 60)
    print("All scenarios complete!")
    print("=" * 60)
