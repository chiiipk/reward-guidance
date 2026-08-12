import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_comparison import generate_analytic_samples
from model import checkerboard_density

def _checker_background(ax):
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X, Y], axis=-1)
    density = checkerboard_density(points)
    ax.contourf(X, Y, density, levels=[0.5, 1.5], colors=["#cfd8dc"], alpha=0.85)

def _draw_panel(ax, samples, title, reward_center, panel_label=None):
    _checker_background(ax)
    if samples is not None and len(samples) > 0:
        ax.scatter(samples[:, 0], samples[:, 1], c="#f6c945", s=12, alpha=0.92, edgecolors="black", linewidths=0.25, zorder=4)
    ax.scatter(reward_center[0], reward_center[1], c="#e60000", s=340, marker="X", edgecolors="black", linewidths=0.9, zorder=6)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.set_aspect("equal")
    ax.tick_params(direction="in", length=4, top=False, right=False)
    ax.set_title(title, pad=15, fontsize=20, fontweight="bold")
    
    if panel_label:
        ax.text(0.04, 0.965, panel_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=22, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#26456e",
                          edgecolor="white", linewidth=0.9), zorder=10)

def plot_4_panels(reward_center, lam, suffix, out_name):
    target_s, _ = generate_analytic_samples(lam, reward_center, num_samples=2000)
    
    try:
        unguided = np.load(f'results/unguided.npz')['samples'][:2000]
    except FileNotFoundError:
        unguided = np.zeros((0, 2))
        
    plugin_s = np.load(f'results/guided_k1_lam{lam}{suffix}.npz')['samples'][:2000]
    damped_s = np.load(f'results/guided_damped_lam{lam}{suffix}.npz')['samples'][:2000]
    second_s = np.load(f'results/guided_second_order_lam{lam}{suffix}.npz')['samples'][:2000]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    
    _draw_panel(axes[0], target_s, "Target (Analytic)", reward_center, "A")
    _draw_panel(axes[1], unguided, "Unguided", reward_center, "B")
    _draw_panel(axes[2], damped_s, "First-Order + Damp", reward_center, "C")
    _draw_panel(axes[3], plugin_s, "First-Order", reward_center, "D")
    _draw_panel(axes[4], second_s, "Second-Order", reward_center, "E")
    
    fig.tight_layout()
    plt.savefig(f"results/{out_name}.png", dpi=200)
    print(f"Saved results/{out_name}.png")

if __name__ == "__main__":
    plot_4_panels([0.5, 1.5], 50.0, "_out", "checkerboard_out_lam50")
    plot_4_panels([0.5, 0.5], 50.0, "_in", "checkerboard_in_lam50")
