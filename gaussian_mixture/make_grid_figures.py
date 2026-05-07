"""Render the Gaussian-mixture comparison grids.

For each variant (gaussian, quadratic, double_well, noniso, unequal,
uncentered), produces one combined 4-panel figure with shared y-axis:
exact guidance, plug-in (k = 1), plug-in (k = 8), and damped plug-in.
The double-well variant has no closed-form analytic guidance; its leftmost
panel uses an exact h-transform sampler instead.
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

from model import (
    DOUBLE_WELL_COVS, DOUBLE_WELL_MEANS, DOUBLE_WELL_SIGMA_HYPER, DOUBLE_WELL_WEIGHTS,
    GAUSSIAN_COVS, GAUSSIAN_MEANS, GAUSSIAN_SIGMA_HYPER, GAUSSIAN_TARGET, GAUSSIAN_WEIGHTS,
    NONISO_COVS, NONISO_MEANS, NONISO_SIGMA_HYPER, NONISO_TARGET, NONISO_WEIGHTS,
    QUADRATIC_COVS, QUADRATIC_MEANS, QUADRATIC_SIGMA_HYPER, QUADRATIC_TARGET, QUADRATIC_WEIGHTS,
    UNCENTERED_COVS, UNCENTERED_MEANS, UNCENTERED_SIGMA_HYPER, UNCENTERED_TARGET, UNCENTERED_WEIGHTS,
    UNEQUAL_COVS, UNEQUAL_MEANS, UNEQUAL_SIGMA_HYPER, UNEQUAL_TARGET, UNEQUAL_WEIGHTS,
    GaussianMixture, LinearInterpolant, reward_double_well, reward_quadratic,
)
from sample import GuidedSampler

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "figures" / "gaussian_mixture"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- styling -----------------------------------------------------------------
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

MAKO_LIKE = mpl.colors.LinearSegmentedColormap.from_list(
    "mako_like",
    ["#dfe6e9", "#83c5be", "#1f9e89", "#26456e", "#2a1755"],
)
SAMPLE_COLOR = "#f6c945"
TRAJ_COLOR   = "#222222"
TARGET_COLOR = "#e60000"
CONTOUR_COLOR = "#1f8f3d"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
N_SAMPLES = 200
NUM_STEPS = 50
FMRG_INNER_STEPS = 30
SEED = 42
LAM = 3.0


# Per-variant config: weights, means, covs, target_or_None, sigma_hyper,
# reward_kind, exact_method, xlim, ylim, peaks_for_marker, xticks
VARIANTS = {
    "gaussian": dict(
        weights=GAUSSIAN_WEIGHTS, means=GAUSSIAN_MEANS, covs=GAUSSIAN_COVS,
        target=GAUSSIAN_TARGET, sigma_hyper=GAUSSIAN_SIGMA_HYPER,
        reward_kind="quadratic", exact_method="analytic",
        xlim=(-3, 3), ylim=(-2, 4), xticks=(-2, 0, 2), yticks=(-2, 0, 2, 4),
    ),
    "quadratic": dict(
        weights=QUADRATIC_WEIGHTS, means=QUADRATIC_MEANS, covs=QUADRATIC_COVS,
        target=QUADRATIC_TARGET, sigma_hyper=QUADRATIC_SIGMA_HYPER,
        reward_kind="quadratic", exact_method="analytic",
        xlim=(-4, 4), ylim=(-2, 4), xticks=(-3, 0, 3), yticks=(-2, 0, 2, 4),
    ),
    "double_well": dict(
        weights=DOUBLE_WELL_WEIGHTS, means=DOUBLE_WELL_MEANS, covs=DOUBLE_WELL_COVS,
        target=None, sigma_hyper=DOUBLE_WELL_SIGMA_HYPER,
        reward_kind="quartic", exact_method="exact",
        xlim=(-4, 4), ylim=(-3, 3), xticks=(-3, 0, 3), yticks=(-2, 0, 2),
    ),
    "noniso": dict(
        weights=NONISO_WEIGHTS, means=NONISO_MEANS, covs=NONISO_COVS,
        target=NONISO_TARGET, sigma_hyper=NONISO_SIGMA_HYPER,
        reward_kind="quadratic", exact_method="analytic",
        xlim=(-4, 4), ylim=(-2, 4), xticks=(-3, 0, 3), yticks=(-2, 0, 2, 4),
    ),
    "unequal": dict(
        weights=UNEQUAL_WEIGHTS, means=UNEQUAL_MEANS, covs=UNEQUAL_COVS,
        target=UNEQUAL_TARGET, sigma_hyper=UNEQUAL_SIGMA_HYPER,
        reward_kind="quadratic", exact_method="analytic",
        xlim=(-4, 4), ylim=(-2, 4), xticks=(-3, 0, 3), yticks=(-2, 0, 2, 4),
    ),
    "uncentered": dict(
        weights=UNCENTERED_WEIGHTS, means=UNCENTERED_MEANS, covs=UNCENTERED_COVS,
        target=UNCENTERED_TARGET, sigma_hyper=UNCENTERED_SIGMA_HYPER,
        reward_kind="quadratic", exact_method="analytic",
        xlim=(-6, 4), ylim=(-2, 4), xticks=(-5, -2, 1, 4), yticks=(-2, 0, 2, 4),
    ),
}

DOUBLE_WELL_PEAKS = [(-2.5, -1.0), (2.5, 1.0)]


def _build_sampler(cfg, sigma_damp=None):
    gmm = GaussianMixture(cfg["weights"], cfg["means"], cfg["covs"]).to(DEVICE)
    interp = LinearInterpolant(gmm).to(DEVICE)
    if cfg["reward_kind"] == "quadratic":
        target_dev = cfg["target"].to(DEVICE)
        reward_fn = lambda x: reward_quadratic(x, target_dev)
    else:
        target_dev = None
        reward_fn = reward_double_well
    sampler = GuidedSampler(
        interp,
        reward_fn,
        lam=LAM,
        sigma_hyper=cfg["sigma_hyper"] if sigma_damp is None else sigma_damp,
        apply_damping_scale=(sigma_damp is not None),
        target=target_dev,
    ).to(DEVICE)
    return gmm, sampler, reward_fn


def _sample(cfg, method, k_particles=1, sigma_damp=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    _, sampler, _ = _build_sampler(cfg, sigma_damp=sigma_damp)
    x = torch.randn(N_SAMPLES, 2, device=DEVICE)
    ts = np.linspace(0.001, 0.999, NUM_STEPS)
    dt = ts[1] - ts[0]
    traj = [x.cpu().numpy()]
    with torch.no_grad():
        for i, t in enumerate(ts[:-1]):
            v1 = sampler.vector_field(
                x, t, method=method, k_particles=k_particles,
                fmrg_inner_steps=FMRG_INNER_STEPS,
            )
            x_pred = x + dt * v1
            v2 = sampler.vector_field(
                x_pred, ts[i + 1], method=method, k_particles=k_particles,
                fmrg_inner_steps=FMRG_INNER_STEPS,
            )
            x = x + (dt / 2.0) * (v1 + v2)
            if i % 10 == 0:
                traj.append(x.cpu().numpy())
    traj.append(x.cpu().numpy())
    return x.cpu().numpy(), np.array(traj)


def _density(cfg, grid_lim_x, grid_lim_y):
    gmm, _, reward_fn = _build_sampler(cfg)
    gx = torch.linspace(grid_lim_x[0], grid_lim_x[1], 240, device=DEVICE)
    gy = torch.linspace(grid_lim_y[0], grid_lim_y[1], 240, device=DEVICE)
    X, Y = torch.meshgrid(gx, gy, indexing="ij")
    XY = torch.stack([X.flatten(), Y.flatten()], dim=1)
    with torch.no_grad():
        log_p = []
        for k in range(gmm.K):
            mvn = torch.distributions.MultivariateNormal(
                loc=gmm.means[k], covariance_matrix=gmm.covs[k]
            )
            log_p.append(mvn.log_prob(XY) + torch.log(gmm.weights[k]))
        log_p = torch.logsumexp(torch.stack(log_p), dim=0)
        log_r = LAM * reward_fn(XY)
        log_dens = log_p + log_r
        log_dens = log_dens - log_dens.max()
        density = torch.exp(log_dens).cpu().numpy().reshape(240, 240)
    return X.cpu().numpy(), Y.cpu().numpy(), density


def _peak_xys(cfg):
    if cfg["reward_kind"] == "quadratic":
        return [(cfg["target"][0].item(), cfg["target"][1].item())]
    return DOUBLE_WELL_PEAKS


def _draw_panel(ax, cfg, samples, traj, X, Y, density,
                 panel_label, panel_title, is_leftmost):
    ax.contourf(X, Y, density, levels=15, cmap=MAKO_LIKE, alpha=0.85)
    contour_levels = np.geomspace(0.02, 0.95, 6)
    ax.contour(X, Y, density, levels=contour_levels, colors=CONTOUR_COLOR,
               alpha=0.75, linewidths=1.1)
    ax.plot(traj[:, :, 0], traj[:, :, 1],
            color=TRAJ_COLOR, alpha=0.4, linewidth=0.35, zorder=2)
    ax.scatter(samples[:, 0], samples[:, 1],
               c=SAMPLE_COLOR, s=22, alpha=0.95, edgecolors="black",
               linewidths=0.4, zorder=4)
    for px, py in _peak_xys(cfg):
        ax.scatter(px, py, c=TARGET_COLOR, s=340, marker="X",
                   edgecolors="black", linewidths=0.9, zorder=6)
    ax.set_xlim(*cfg["xlim"])
    ax.set_ylim(*cfg["ylim"])
    ax.set_xticks(list(cfg["xticks"]))
    ax.set_yticks(list(cfg["yticks"]))
    ax.tick_params(direction="in", length=4, top=False, right=False)
    ax.grid(True, color="white", alpha=0.35, linewidth=0.5, zorder=1)
    if not is_leftmost:
        ax.tick_params(left=False, labelleft=False)
    ax.text(0.04, 0.965, panel_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=22, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#26456e",
                      edgecolor="white", linewidth=0.9))
    ax.set_title(panel_title, pad=10)


def _build_figure(variant, cfg):
    X, Y, density = _density(cfg, cfg["xlim"], cfg["ylim"])
    panels = [
        (cfg["exact_method"], 1, None,                  "A", "Exact guidance"),
        ("plugin",            1, None,                  "B", r"Plug-in ($k = 1$)"),
        ("plugin",            8, None,                  "C", r"Plug-in ($k = 8$)"),
        ("plugin",            1, cfg["sigma_hyper"],    "D", "Plug-in (damped)"),
    ]
    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(20.5, 5.4),
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )
    for ax, (method, k, sd, label, title) in zip(axes, panels):
        samples, traj = _sample(cfg, method, k_particles=k, sigma_damp=sd)
        _draw_panel(ax, cfg, samples, traj, X, Y, density,
                    label, title, is_leftmost=(ax is axes[0]))

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
    leg = axes[-1].legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True, framealpha=0.85,
        facecolor="white", edgecolor="#888888", fancybox=True,
    )
    leg.set_zorder(20)
    leg.get_frame().set_alpha(0.85)

    fig.tight_layout()
    out_pdf = OUT_DIR / f"{variant}.pdf"
    out_png = OUT_DIR / f"{variant}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"saved -> {out_pdf}")
    print(f"saved -> {out_png}")


def main():
    for variant, cfg in VARIANTS.items():
        print(f"=== {variant} ===")
        _build_figure(variant, cfg)


if __name__ == "__main__":
    main()
