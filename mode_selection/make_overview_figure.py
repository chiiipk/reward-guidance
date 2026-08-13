"""Render the best-of-n overview figure with the step reward.

Four panels sharing the y-axis on the 1D symmetric Gaussian mixture:
  A) Analytic reward tilt
  B) Plug-in guidance (k = 1, best-of-1)
  C) Best-of-4
  D) Best-of-16

Input: results/step_lam5.0/samples.npz, produced by mode_selection/sample.py.
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
from matplotlib.patches import Patch
from scipy.stats import norm

from sample import best_of_n

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "mode_selection" / "results" / "step_lam5.0" / "samples.npz"
OUT_DIR = REPO / "figures" / "mode_selection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- styling ---
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

SAMPLE_COLOR  = "#f6c945"  # warm gold for the histogram bars
TILT_COLOR    = "#1f8f3d"  # green for analytic tilt curve
BASE_COLOR    = "#444444"  # neutral grey for unguided base density
PILL_COLOR    = "#26456e"  # deep blue pill background for A/B/C/D


def _load():
    d = np.load(DATA)
    unguided = torch.tensor(d["unguided"])
    guided = torch.tensor(d["guided"])
    rewards = torch.tensor(d["guided_rewards"])
    num_samples = int(d["num_samples"])
    lam = float(d["lam"])
    mu = float(d["mu"])
    sigma = float(d["sigma"])
    return unguided, guided, rewards, num_samples, lam, mu, sigma


def _analytic_tilt(x_grid, mu, sigma, lam, R=10.0):
    rho = 0.5 * norm.pdf(x_grid, mu, sigma) + 0.5 * norm.pdf(x_grid, -mu, sigma)
    log_r = np.where(x_grid >= 0, 0.0, -R)
    log_t = lam * log_r + np.log(rho + 1e-300)
    log_t -= log_t.max()
    t = np.exp(log_t)
    integral = np.trapezoid(t, x_grid) if hasattr(np, 'trapezoid') else np.trapz(t, x_grid)
    t /= integral
    return rho, t


def _draw_panel(ax, samples, x_grid, base_density, tilt_density,
                 panel_label, panel_title, is_leftmost,
                 show_tilt_curve=True, show_hist=True):
    bins = np.linspace(-8, 8, 60)
    if show_hist:
        ax.hist(samples, bins=bins, density=True, color=SAMPLE_COLOR,
                edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3)
    ax.plot(x_grid, base_density, color=BASE_COLOR, linewidth=1.2, alpha=0.6,
            zorder=4)
    if show_tilt_curve:
        ax.plot(x_grid, tilt_density, color=TILT_COLOR, linewidth=2.4,
                alpha=0.95, zorder=5)
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 0.62)
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.tick_params(direction="in", length=4, top=False, right=False)
    ax.grid(True, color="white", alpha=0.0, linewidth=0.5, zorder=1)
    if is_leftmost:
        ax.set_ylabel("Density", fontsize=22)
    else:
        ax.tick_params(left=False, labelleft=False)
    ax.text(0.04, 0.965, panel_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=22, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=PILL_COLOR,
                      edgecolor="white", linewidth=0.9))
    ax.set_title(panel_title, pad=10)


def main():
    unguided, guided, rewards, num_samples, lam, mu, sigma = _load()
    x_grid = np.linspace(-8, 8, 500)
    base, tilt = _analytic_tilt(x_grid, mu, sigma, lam)

    plugin_samples = guided[:num_samples].numpy()
    sel4, _, _ = best_of_n(guided, rewards, 4)
    sel16, _, _ = best_of_n(guided, rewards, 16)

    panels = [
        (None,                "A", r"Analytic tilt $\tilde{\rho}_1$"),
        (plugin_samples,      "B", r"Plug-in ($k=1$)"),
        (sel4.numpy(),        "C", r"Best-of-$4$"),
        (sel16.numpy(),       "D", r"Best-of-$16$"),
    ]

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(20.5, 5.4),
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )
    for i, (ax, (samples, label, title)) in enumerate(zip(axes, panels)):
        is_leftmost = (i == 0)
        _draw_panel(ax, samples if samples is not None else np.array([]),
                     x_grid, base, tilt, label, title,
                     is_leftmost=is_leftmost,
                     show_tilt_curve=True, show_hist=(samples is not None))

    legend_handles = [
        Line2D([0], [0], color=BASE_COLOR, lw=1.6, alpha=0.75,
               label=r"Unguided $\rho_1$"),
        Line2D([0], [0], color=TILT_COLOR, lw=2.4, alpha=1.0,
               label=r"Analytic tilt $\tilde{\rho}_1$"),
        Patch(facecolor=SAMPLE_COLOR, edgecolor="black", linewidth=0.4,
              alpha=0.85, label="Guided samples"),
    ]
    leg = axes[-1].legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True, framealpha=0.85,
        facecolor="white", edgecolor="#888888", fancybox=True,
    )
    leg.set_zorder(20)
    leg.get_frame().set_alpha(0.85)

    fig.tight_layout()
    out_pdf = OUT_DIR / "best_of_n_overview.pdf"
    out_png = OUT_DIR / "best_of_n_overview.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"saved -> {out_pdf}")
    print(f"saved -> {out_png}")


if __name__ == "__main__":
    main()
