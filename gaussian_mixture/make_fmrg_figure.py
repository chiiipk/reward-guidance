"""Render the FMRG vs. plug-in regime-crossover comparison.

Produces a single 1x4 figure on an isotropic Gaussian target with quadratic
reward, with two regimes:
  A) Plug-in (k = 1) -- narrow (sigma = 0.5, lambda = 1)
  B) FMRG            -- narrow (sigma = 0.5, lambda = 1)
  C) Plug-in (k = 1) -- wide (sigma = 16, lambda = 0.1)
  D) FMRG            -- wide (sigma = 16, lambda = 0.1)

The narrow and wide panels use different x/y limits, so the y-axis is shared
within each pair but not across the regime change.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from model import GaussianMixture, LinearInterpolant, reward_quadratic
from sample import GuidedSampler

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "figures" / "gaussian_mixture"
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

SAMPLE_COLOR  = "#f6c945"
TRAJ_COLOR    = "#222222"
TARGET_COLOR  = "#e60000"
CONTOUR_COLOR = "#1f8f3d"
PILL_COLOR    = "#26456e"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
N_SAMPLES = 200
NUM_STEPS = 50
FMRG_INNER_STEPS = 30
SEED = 42

# Target placed in the positive-y half of the plane.
TARGET = torch.tensor([0.0, 2.5])

# Narrow ($\sigma^2 = 0.5$, std $\approx 0.71$) and wide ($\sigma^2 = 16$,
# std $= 4$) isotropic Gaussian variants.
NARROW = dict(
    cov=[[0.5, 0.0], [0.0, 0.5]],
    lam=1.0,
    xlim=(-3, 3), ylim=(-2, 4),
    xticks=(-2, 0, 2), yticks=(-2, 0, 2, 4),
    grid_lim=4,
    sigma_label=r"$\sigma = 0.5$",
)
WIDE = dict(
    cov=[[16.0, 0.0], [0.0, 16.0]],
    lam=0.1,
    xlim=(-8, 8), ylim=(-8, 8),
    xticks=(-6, -3, 0, 3, 6), yticks=(-6, -3, 0, 3, 6),
    grid_lim=8,
    sigma_label=r"$\sigma = 16$",
)


def _sample(method, cfg):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    gmm = GaussianMixture([1.0], [[0.0, 0.0]], [cfg["cov"]]).to(DEVICE)
    interp = LinearInterpolant(gmm).to(DEVICE)
    target_dev = TARGET.to(DEVICE)
    sampler = GuidedSampler(
        interp,
        lambda x: reward_quadratic(x, target_dev),
        lam=cfg["lam"],
        sigma_hyper=float(np.sqrt(cfg["cov"][0][0])),
        apply_damping_scale=False,
        target=target_dev,
    ).to(DEVICE)
    x = torch.randn(N_SAMPLES, 2, device=DEVICE)
    ts = np.linspace(0.001, 0.999, NUM_STEPS)
    dt = ts[1] - ts[0]
    traj = [x.cpu().numpy()]
    with torch.no_grad():
        for i, t in enumerate(ts[:-1]):
            v1 = sampler.vector_field(x, t, method=method, k_particles=1,
                                       fmrg_inner_steps=FMRG_INNER_STEPS)
            x_pred = x + dt * v1
            v2 = sampler.vector_field(x_pred, ts[i + 1], method=method,
                                       k_particles=1,
                                       fmrg_inner_steps=FMRG_INNER_STEPS)
            x = x + (dt / 2.0) * (v1 + v2)
            if i % 10 == 0:
                traj.append(x.cpu().numpy())
    traj.append(x.cpu().numpy())
    return x.cpu().numpy(), np.array(traj)


def _density(cfg):
    gmm = GaussianMixture([1.0], [[0.0, 0.0]], [cfg["cov"]]).to(DEVICE)
    target_dev = TARGET.to(DEVICE)
    gx = torch.linspace(cfg["xlim"][0], cfg["xlim"][1], 240, device=DEVICE)
    gy = torch.linspace(cfg["ylim"][0], cfg["ylim"][1], 240, device=DEVICE)
    X, Y = torch.meshgrid(gx, gy, indexing="ij")
    XY = torch.stack([X.flatten(), Y.flatten()], dim=1)
    with torch.no_grad():
        mvn = torch.distributions.MultivariateNormal(
            loc=gmm.means[0], covariance_matrix=gmm.covs[0]
        )
        log_p = mvn.log_prob(XY)
        log_r = cfg["lam"] * reward_quadratic(XY, target_dev)
        log_dens = log_p + log_r
        log_dens = log_dens - log_dens.max()
        density = torch.exp(log_dens).cpu().numpy().reshape(240, 240)
    return X.cpu().numpy(), Y.cpu().numpy(), density


def _draw_panel(ax, cfg, samples, traj, X, Y, density, panel_label, panel_title,
                  show_yticks=True):
    contour_levels = np.geomspace(0.02, 0.95, 6)
    ax.contour(X, Y, density, levels=contour_levels, colors=CONTOUR_COLOR,
               alpha=0.65, linewidths=1.2)
    ax.plot(traj[:, :, 0], traj[:, :, 1],
            color=TRAJ_COLOR, alpha=0.2, linewidth=0.3, zorder=2)
    ax.scatter(samples[:, 0], samples[:, 1],
               c=SAMPLE_COLOR, s=22, alpha=0.95, edgecolors="black",
               linewidths=0.4, zorder=4)
    ax.scatter(TARGET[0].item(), TARGET[1].item(),
               c=TARGET_COLOR, s=340, marker="X",
               edgecolors="black", linewidths=0.9, zorder=6)
    ax.set_xlim(*cfg["xlim"])
    ax.set_ylim(*cfg["ylim"])
    ax.set_xticks(list(cfg["xticks"]))
    ax.set_yticks(list(cfg["yticks"]))
    ax.tick_params(direction="in", length=4, top=False, right=False)
    if not show_yticks:
        ax.tick_params(left=False, labelleft=False)
    ax.text(0.04, 0.965, panel_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=22, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=PILL_COLOR,
                      edgecolor="white", linewidth=0.9))
    ax.set_title(panel_title, pad=10)


def main():
    Xn, Yn, dens_n = _density(NARROW)
    Xw, Yw, dens_w = _density(WIDE)

    plug_n_s, plug_n_t = _sample("plugin", NARROW)
    fmrg_n_s, fmrg_n_t = _sample("fmrg",   NARROW)
    plug_w_s, plug_w_t = _sample("plugin", WIDE)
    fmrg_w_s, fmrg_w_t = _sample("fmrg",   WIDE)

    # 1x4 layout with a wider gap between B (narrow) and C (wide).
    fig = plt.figure(figsize=(20.5, 5.4))
    gs = GridSpec(
        1, 5,
        width_ratios=[1.0, 1.0, 0.18, 1.0, 1.0],
        wspace=0.06,
        figure=fig,
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1], sharey=axA)
    axC = fig.add_subplot(gs[0, 3])
    axD = fig.add_subplot(gs[0, 4], sharey=axC)

    _draw_panel(axA, NARROW, plug_n_s, plug_n_t, Xn, Yn, dens_n,
                 "A", r"Plug-in ($k = 1$), " + NARROW["sigma_label"],
                 show_yticks=True)
    _draw_panel(axB, NARROW, fmrg_n_s, fmrg_n_t, Xn, Yn, dens_n,
                 "B", r"FMRG, " + NARROW["sigma_label"],
                 show_yticks=False)
    _draw_panel(axC, WIDE, plug_w_s, plug_w_t, Xw, Yw, dens_w,
                 "C", r"Plug-in ($k = 1$), " + WIDE["sigma_label"],
                 show_yticks=True)
    _draw_panel(axD, WIDE, fmrg_w_s, fmrg_w_t, Xw, Yw, dens_w,
                 "D", r"FMRG, " + WIDE["sigma_label"],
                 show_yticks=False)

    legend_handles = [
        Line2D([0], [0], color=CONTOUR_COLOR, lw=2.4, alpha=1.0,
               label="Analytic tilt"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SAMPLE_COLOR,
               markeredgecolor="black", markeredgewidth=0.5, markersize=10,
               label="Guided samples"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=TARGET_COLOR,
               markeredgecolor="black", markeredgewidth=0.9, markersize=14,
               label="Reward target"),
    ]
    leg = axD.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True, framealpha=0.85,
        facecolor="white", edgecolor="#888888", fancybox=True,
    )
    leg.set_zorder(20)
    leg.get_frame().set_alpha(0.85)

    fig.tight_layout()
    out_pdf = OUT_DIR / "fmrg_crossover.pdf"
    out_png = OUT_DIR / "fmrg_crossover.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"saved -> {out_pdf}")
    print(f"saved -> {out_png}")


if __name__ == "__main__":
    main()
