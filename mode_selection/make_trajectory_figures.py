"""Render the mode-selection particle-trajectory figures.

Two figures, both on the 1D symmetric Gaussian mixture:
  step_trajectories.pdf      -- 4 panels: Unguided, Plug-in (k = 1),
                                Best-of-4, Best-of-16   (step reward)
  gaussian_trajectories.pdf  -- 3 panels: Plug-in (k = 1), Best-of-4,
                                Best-of-16              (Gaussian reward)

Trajectories are colored by terminal sign (positive mode = green, negative
mode = coral). Inputs come from mode_selection/sample.py.
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

from sample import best_of_n

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "mode_selection" / "results"
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

POS_COLOR  = "#1f8f3d"  # green for positive (correct) mode
NEG_COLOR  = "#e07a5f"  # coral for negative mode
GUIDE_COLOR = "#888888"  # neutral grey for mu reference lines
PILL_COLOR  = "#26456e"
MAX_PARTICLES = 200
SEED = 0


def _select_indices(samples, rewards, n, num_samples):
    rng = np.random.default_rng(SEED)
    if n == 1:
        n_show = min(MAX_PARTICLES, num_samples)
        return rng.choice(num_samples, n_show, replace=False)
    _, _, best_idx = best_of_n(samples, rewards, n)
    num_groups = samples.shape[0] // n
    global_idx = (torch.arange(num_groups) * n + best_idx).numpy()
    n_show = min(MAX_PARTICLES, len(global_idx))
    sel = rng.choice(len(global_idx), n_show, replace=False)
    return global_idx[sel]


def _draw_panel(ax, traj_sub, ts, mu, panel_label, panel_title, is_leftmost,
                  is_rightmost, ylim):
    final = traj_sub[-1]
    for i in range(traj_sub.shape[1]):
        color = POS_COLOR if final[i] >= 0 else NEG_COLOR
        ax.plot(ts, traj_sub[:, i], color=color, alpha=0.18, linewidth=0.7,
                zorder=2)
    ax.axhline(mu,  color=POS_COLOR, linestyle="--", alpha=0.55,
               linewidth=1.0, zorder=3)
    ax.axhline(-mu, color=NEG_COLOR, linestyle="--", alpha=0.55,
               linewidth=1.0, zorder=3)
    ax.axhline(0,   color=GUIDE_COLOR, linestyle=":", alpha=0.55,
               linewidth=1.0, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    # Skip the 0 and 1 ticks entirely so adjacent shared panels don't have
    # their left/right tick labels colliding.
    ax.set_xticks([0.25, 0.5, 0.75])
    ax.tick_params(direction="in", length=4, top=False, right=False)
    if is_leftmost:
        ax.set_ylabel(r"Position $x$", fontsize=22)
    else:
        ax.tick_params(left=False, labelleft=False)
    ax.set_xlabel(r"Time $t$")
    ax.text(0.04, 0.965, panel_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=22, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=PILL_COLOR,
                      edgecolor="white", linewidth=0.9))
    ax.set_title(panel_title, pad=10)


def _legend_handles(mu_label_pos, mu_label_neg):
    return [
        Line2D([0], [0], color=POS_COLOR, lw=2.2, alpha=0.9,
               label="Positive trajectories"),
        Line2D([0], [0], color=NEG_COLOR, lw=2.2, alpha=0.9,
               label="Negative trajectories"),
        Line2D([0], [0], color=POS_COLOR, lw=1.4, ls="--", alpha=0.7,
               label=mu_label_pos),
        Line2D([0], [0], color=NEG_COLOR, lw=1.4, ls="--", alpha=0.7,
               label=mu_label_neg),
    ]


def _build(reward_name, panels, ylim, out_name):
    samples = np.load(RES / f"{reward_name}_lam5.0" / "samples.npz")
    traj = np.load(RES / f"{reward_name}_lam5.0" / "trajectories.npz")["guided"]
    guided_samples = torch.tensor(samples["guided"])
    guided_rewards = torch.tensor(samples["guided_rewards"])
    num_samples = int(samples["num_samples"])
    mu = float(samples["mu"])

    T = traj.shape[0]
    ts = np.linspace(0, 1, T)

    n_panels = len(panels)
    width = 5.0 * n_panels + 0.5
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(width, 5.4),
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )
    if n_panels == 1:
        axes = [axes]
    for i, (ax, (n, label, title)) in enumerate(zip(axes, panels)):
        idx = _select_indices(guided_samples, guided_rewards, n, num_samples)
        traj_sub = traj[:, idx]
        _draw_panel(ax, traj_sub, ts, mu, label, title,
                     is_leftmost=(i == 0),
                     is_rightmost=(i == n_panels - 1),
                     ylim=ylim)

    leg = fig.legend(
        handles=_legend_handles(r"$+\mu$", r"$-\mu$"),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.14),
        frameon=True, framealpha=0.9,
        facecolor="white", edgecolor="#888888", fancybox=True,
    )
    leg.set_zorder(20)
    leg.get_frame().set_alpha(0.9)

    fig.tight_layout(rect=(0, 0.20, 1, 1))
    out_pdf = OUT_DIR / f"{out_name}.pdf"
    out_png = OUT_DIR / f"{out_name}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"saved -> {out_pdf}")
    print(f"saved -> {out_png}")


def main():
    _build_step()
    _build("gaussian", panels=[
        (1,  "A", r"Plug-in ($k = 1$)"),
        (4,  "B", r"Best-of-$4$"),
        (16, "C", r"Best-of-$16$"),
    ], ylim=(-9, 9), out_name="gaussian_trajectories")


def _build_step():
    samples = np.load(RES / "step_lam5.0" / "samples.npz")
    traj = np.load(RES / "step_lam5.0" / "trajectories.npz")
    guided_samples = torch.tensor(samples["guided"])
    guided_rewards = torch.tensor(samples["guided_rewards"])
    num_samples = int(samples["num_samples"])
    mu = float(samples["mu"])

    unguided_traj = traj["unguided"]
    guided_traj = traj["guided"]
    T = guided_traj.shape[0]
    ts = np.linspace(0, 1, T)

    rng = np.random.default_rng(SEED)
    unguided_idx = rng.choice(unguided_traj.shape[1],
                                min(MAX_PARTICLES, unguided_traj.shape[1]),
                                replace=False)
    unguided_sub = unguided_traj[:, unguided_idx]

    panels_data = [
        ("A", r"Unguided", unguided_sub),
    ]
    for n, label, title in [
        (1,  "B", r"Plug-in ($k = 1$)"),
        (4,  "C", r"Best-of-$4$"),
        (16, "D", r"Best-of-$16$"),
    ]:
        idx = _select_indices(guided_samples, guided_rewards, n, num_samples)
        panels_data.append((label, title, guided_traj[:, idx]))

    n_panels = len(panels_data)
    width = 5.0 * n_panels + 0.5
    ylim = (-9, 9)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(width, 5.4),
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )
    for i, (ax, (label, title, traj_sub)) in enumerate(zip(axes, panels_data)):
        _draw_panel(ax, traj_sub, ts, mu, label, title,
                     is_leftmost=(i == 0),
                     is_rightmost=(i == n_panels - 1),
                     ylim=ylim)

    leg = fig.legend(
        handles=_legend_handles(r"$+\mu$", r"$-\mu$"),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.14),
        frameon=True, framealpha=0.9,
        facecolor="white", edgecolor="#888888", fancybox=True,
    )
    leg.set_zorder(20)
    leg.get_frame().set_alpha(0.9)

    fig.tight_layout(rect=(0, 0.20, 1, 1))
    out_pdf = OUT_DIR / "step_trajectories.pdf"
    out_png = OUT_DIR / "step_trajectories.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"saved -> {out_pdf}")
    print(f"saved -> {out_png}")


if __name__ == "__main__":
    main()
