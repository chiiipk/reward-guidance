"""Render the four-panel checkerboard mode-selection figure.

Panels (left to right):
  A) Analytic reward-tilted distribution (rejection-sampled ground truth)
  B) Plug-in guidance with k = 1 particle
  C) Plug-in guidance with k = 8 particles
  D) Best-of-4 selection on top of damped guidance (sigma_damp = 0.2)

Inputs (produced by ``checkerboard/sample.py``):
  results/analytic_tilt_lam10.0.npz
  results/guided_k1_lam10.0.npz
  results/guided_k8_lam10.0.npz
  results/guided_k1_lam10.0_damp0.2.npz
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from model import checkerboard_density, DEFAULT_REWARD_CENTER
from sample import best_of_n

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "checkerboard" / "results"
OUT_DIR = REPO / "figures" / "checkerboard"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for ttf in (REPO / "assets" / "fonts").glob("Lato-*.ttf"):
    fm.fontManager.addfont(str(ttf))
plt.style.use(str(REPO / "assets" / "paper.mplstyle"))
mpl.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 19,
    "axes.titlesize": 23,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 20,
    "axes.spines.top":   True,
    "axes.spines.right": True,
})

SAMPLE_COLOR  = "#f6c945"
TARGET_COLOR  = "#e60000"
CHECKER_COLOR = "#cfd8dc"
PILL_COLOR    = "#26456e"

GRID_RANGE = (-3, 3)
GRID_RES = 300
DISPLAY_N = 5000
SEED = 0


def _make_grid():
    x = np.linspace(*GRID_RANGE, GRID_RES)
    y = np.linspace(*GRID_RANGE, GRID_RES)
    X, Y = np.meshgrid(x, y)
    return X, Y, np.stack([X, Y], axis=-1)


def _checker_background(ax):
    X, Y, points = _make_grid()
    density = checkerboard_density(points)
    ax.contourf(X, Y, density, levels=[0.5, 1.5], colors=[CHECKER_COLOR],
                alpha=0.85)


def _subsample(samples, seed=SEED):
    if len(samples) > DISPLAY_N:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(samples), DISPLAY_N, replace=False)
        return samples[idx]
    return samples


def _draw_panel(ax, samples, panel_label, panel_title, is_leftmost,
                 show_target=True):
    _checker_background(ax)
    ax.scatter(samples[:, 0], samples[:, 1],
               c=SAMPLE_COLOR, s=12, alpha=0.92, edgecolors="black",
               linewidths=0.25, zorder=4)
    if show_target:
        center = DEFAULT_REWARD_CENTER.numpy()
        ax.scatter(center[0], center[1],
                    c=TARGET_COLOR, s=340, marker="X",
                    edgecolors="black", linewidths=0.9, zorder=6)
    ax.set_xlim(*GRID_RANGE)
    ax.set_ylim(*GRID_RANGE)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.set_aspect("equal")
    ax.tick_params(direction="in", length=4, top=False, right=False)
    if not is_leftmost:
        ax.tick_params(left=False, labelleft=False)
    ax.text(0.04, 0.965, panel_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=22, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=PILL_COLOR,
                      edgecolor="white", linewidth=0.9))
    ax.set_title(panel_title, pad=10)


def main():
    analytic = np.load(RES / "analytic_tilt_lam10.0.npz")["samples"]

    g_k1 = np.load(RES / "guided_k1_lam10.0.npz")
    k1_samples = g_k1["samples"]

    g_k8 = np.load(RES / "guided_k8_lam10.0.npz")
    k8_samples = g_k8["samples"]

    g_damp = np.load(RES / "guided_k1_lam10.0_damp0.2.npz")
    sel, _ = best_of_n(torch.tensor(g_damp["samples"]),
                        torch.tensor(g_damp["rewards"]), 4)
    bon_damp_samples = sel.numpy()

    panels = [
        (analytic,         "A", r"Analytic tilt $\tilde{\rho}_1$"),
        (k1_samples,       "B", r"Plug-in ($k = 1$)"),
        (k8_samples,       "C", r"Plug-in ($k = 8$)"),
        (bon_damp_samples, "D", r"Best-of-$4$ ($\sigma_{\mathrm{damp}} = 0.2$)"),
    ]

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(20.5, 5.6),
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )
    for i, (ax, (samples, label, title)) in enumerate(zip(axes, panels)):
        _draw_panel(ax, _subsample(samples), label, title, is_leftmost=(i == 0))

    legend_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=SAMPLE_COLOR, markeredgecolor="black",
               markeredgewidth=0.25, markersize=10, label="Guided samples"),
        Line2D([0], [0], marker="X", color="none",
               markerfacecolor=TARGET_COLOR, markeredgecolor="black",
               markeredgewidth=0.9, markersize=14, label="Reward target"),
    ]
    leg = axes[-1].legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True, framealpha=0.85,
        facecolor="white", edgecolor="#888888", fancybox=True,
    )
    leg.set_zorder(20)
    leg.get_frame().set_alpha(0.85)

    fig.tight_layout()
    out_pdf = OUT_DIR / "main_figure.pdf"
    out_png = OUT_DIR / "main_figure.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"saved -> {out_pdf}")
    print(f"saved -> {out_png}")


if __name__ == "__main__":
    main()
