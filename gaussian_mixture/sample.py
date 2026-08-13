"""Guided sampling via GLASS flow + plug-in estimator, with caching."""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from model import (
    GaussianMixture,
    LinearInterpolant,
    DOUBLE_WELL_COVS,
    DOUBLE_WELL_MEANS,
    DOUBLE_WELL_SIGMA_HYPER,
    DOUBLE_WELL_WEIGHTS,
    GAUSSIAN_COVS,
    GAUSSIAN_MEANS,
    GAUSSIAN_SIGMA_HYPER,
    GAUSSIAN_TARGET,
    GAUSSIAN_WEIGHTS,
    NONISO_COVS,
    NONISO_MEANS,
    NONISO_SIGMA_HYPER,
    NONISO_TARGET,
    NONISO_WEIGHTS,
    QUADRATIC_COVS,
    QUADRATIC_MEANS,
    QUADRATIC_SIGMA_HYPER,
    QUADRATIC_TARGET,
    QUADRATIC_WEIGHTS,
    UNCENTERED_COVS,
    UNCENTERED_MEANS,
    UNCENTERED_SIGMA_HYPER,
    UNCENTERED_TARGET,
    UNCENTERED_WEIGHTS,
    UNEQUAL_COVS,
    UNEQUAL_MEANS,
    UNEQUAL_SIGMA_HYPER,
    UNEQUAL_TARGET,
    UNEQUAL_WEIGHTS,
    reward_double_well,
    reward_quadratic,
)


# ---------------------------------------------------------------------------
# Guided sampler
# ---------------------------------------------------------------------------


class GuidedSampler(nn.Module):
    """Doob h-transform sampler with GLASS flow plug-in estimator.

    Supports three methods for computing the guidance:
      - "analytic": closed-form E[exp(λ r(X₁)) | X_t] for quadratic rewards
      - "exact": Monte Carlo over the analytic posterior (any reward)
      - "plugin": GLASS flow plug-in estimator (any reward)
    """

    def __init__(
        self,
        interpolant,
        reward_fn,
        lam=1.0,
        sigma_hyper=1.0,
        apply_damping_scale=True,
        target=None,
    ):
        super().__init__()
        self.interpolant = interpolant
        self.reward_fn = reward_fn
        self.lam = lam
        self.D = interpolant.dim
        self.sigma_hyper = sigma_hyper
        self.apply_damping_scale = apply_damping_scale
        self.target = target  # required for method="analytic"

    def get_effective_lambda(self, t):
        """Compute the time-dependent reward scale lambda_t for reward damping."""
        lam_eff = self.lam
        if self.apply_damping_scale:
            t_val = t.item() if torch.is_tensor(t) else t
            var_t = (1 - t_val) ** 2
            sig_1_t_sq = (self.sigma_hyper**2 * var_t) / (
                var_t + t_val**2 * self.sigma_hyper**2
            )
            lam_eff = self.lam / (1 + 2 * self.lam * sig_1_t_sq)
        return lam_eff

    def get_analytic_value(self, x_t, t):
        """Closed-form log E[exp(λ r(X₁)) | X_t = x_t] for quadratic reward."""
        means, covs, _, weights = self.interpolant.get_posterior_components(x_t, t)
        log_expectations = []

        for k in range(self.interpolant.gmm.K):
            m = means[:, k, :]
            S = covs[k]

            M = torch.eye(self.D, device=x_t.device) + 2 * self.lam * S
            L_M = torch.linalg.cholesky(M)
            log_det_M = 2 * torch.sum(torch.log(torch.diagonal(L_M)))
            prefactor = -0.5 * log_det_M

            diff = m - self.target
            y = torch.linalg.solve_triangular(L_M, diff.T, upper=False)
            quad = torch.sum(y**2, dim=0)
            log_expectations.append(prefactor + (-self.lam * quad))

        log_expectations = torch.stack(log_expectations, dim=1)
        log_w = torch.log(weights + 1e-10)
        return -torch.logsumexp(log_w + log_expectations, dim=1)

    def get_numeric_exact_value(self, x_t, t, n_mc_samples=1000):
        """MC estimate of log E[exp(λ r(X₁)) | X_t = x_t] over the analytic posterior."""
        B, D = x_t.shape
        means, _, covs_chol, weights = self.interpolant.get_posterior_components(x_t, t)

        Z = torch.randn(n_mc_samples, D, device=x_t.device)
        lam_eff = self.get_effective_lambda(t)

        log_expectations = []
        for k in range(self.interpolant.gmm.K):
            mu_k = means[:, k, :]
            L_k = covs_chol[k]

            X_1_k = mu_k.unsqueeze(1) + (Z @ L_k.T).unsqueeze(0)
            rewards_k = self.reward_fn(X_1_k.view(-1, D)).view(B, n_mc_samples)

            log_exp_k = torch.logsumexp(lam_eff * rewards_k, dim=1) - np.log(n_mc_samples)
            log_expectations.append(log_exp_k)

        log_expectations = torch.stack(log_expectations, dim=1)
        log_w = torch.log(weights + 1e-10)
        return -torch.logsumexp(log_w + log_expectations, dim=1)

    def sample_glass_transition(self, x_t, t, k_particles=1, n_inner_steps=20):
        """GLASS inner probability-flow ODE from X_t to X_1.

        Strictly memoryless to preserve denoiser identities.
        """
        B, D = x_t.shape
        t_val = t.item() if torch.is_tensor(t) else t
        t_prime = 0.99

        bar_alpha = t_prime
        bar_sigma = 1.0 - t_prime

        x_t_exp = x_t.repeat_interleave(k_particles, dim=0)
        bar_x = torch.randn_like(x_t_exp)

        ds = 1.0 / n_inner_steps

        for step in range(n_inner_steps):
            s = max(step * ds, 0.01)

            bar_alpha_s = s * bar_alpha
            bar_sigma_s = 1.0 + s * (bar_sigma - 1.0)

            dot_bar_sigma_s = bar_sigma - 1.0
            dot_bar_alpha_s = bar_alpha

            w1 = dot_bar_sigma_s / bar_sigma_s
            w2 = dot_bar_alpha_s - bar_alpha_s * w1

            sigma_t_sq = max((1.0 - t_val) ** 2, 1e-8)
            bar_sigma_s_sq = max(bar_sigma_s**2, 1e-8)

            v1 = t_val / sigma_t_sq
            v2 = bar_alpha_s / bar_sigma_s_sq

            S_numerator = v1 * x_t_exp + v2 * bar_x
            precision_sum = (t_val**2) / sigma_t_sq + (bar_alpha_s**2) / bar_sigma_s_sq

            S = S_numerator / precision_sum

            g_star = 1.0 / precision_sum
            sqrt_g = np.sqrt(max(g_star, 1e-10))
            t_star = max(0.01, min(1.0 / (1.0 + sqrt_g), 0.99))

            z_hat = self.interpolant.denoise(t_star * S, t_star)

            bar_x = bar_x + ds * (w1 * bar_x + w2 * z_hat)

        return bar_x

    def get_plugin_value(self, x_t, t, k_particles=1):
        """Log E[exp(λ r(X₁)) | X_t = x_t] via GLASS flow plug-in."""
        B, D = x_t.shape
        t_val = t.item() if torch.is_tensor(t) else t
        lam_eff = self.get_effective_lambda(t_val)

        if t_val >= 0.99:
            return -(lam_eff * self.reward_fn(x_t))

        bar_x_1 = self.sample_glass_transition(x_t, t_val, k_particles=k_particles)
        x1_est = self.interpolant.denoise(bar_x_1, 0.99)

        r = self.reward_fn(x1_est)
        r_reshaped = r.view(B, k_particles)
        log_exp = torch.logsumexp(lam_eff * r_reshaped, dim=1) - np.log(k_particles)

        return -log_exp

    def get_fmrg_value(self, x_t, t, num_inner_steps=50, t_end=0.999):
        """Negative log h with the FMRG single-endpoint surrogate.

        Returns -lam_eff * r(X_{t, t_end}(x_t)), where X_{t, t_end} is the deterministic
        unguided probability flow integrated forward from t to t_end. We integrate in
        the log-time coordinate u = -log(1 - t), in which the flow becomes
        dx/du = D_{t(u)}(x) - x and is bounded for all u >= 0, avoiding the 1/(1-t)
        singularity in the original velocity field.
        """
        t_val = t.item() if torch.is_tensor(t) else t
        lam_eff = self.get_effective_lambda(t_val)

        if t_val >= t_end:
            return -(lam_eff * self.reward_fn(x_t))

        num_steps = max(1, num_inner_steps)
        u_start = -np.log(1.0 - t_val)
        u_end = -np.log(1.0 - t_end)
        du = (u_end - u_start) / num_steps
        x_curr = x_t
        for k in range(num_steps):
            u = u_start + k * du
            s = 1.0 - np.exp(-u)
            s_next = 1.0 - np.exp(-(u + du))
            v0 = self.interpolant.denoise(x_curr, s) - x_curr
            x_pred = x_curr + du * v0
            v1 = self.interpolant.denoise(x_pred, s_next) - x_pred
            x_curr = x_curr + 0.5 * du * (v0 + v1)

        return -(lam_eff * self.reward_fn(x_curr))

    def vector_field(self, x, t, method="plugin", k_particles=1, fmrg_inner_steps=50):
        r"""Compute the guided vector field b_t + (scale * grad log h_t or u_t^FMRG).

        Args:
            method: "analytic" (closed-form, quadratic reward only),
                    "exact" (MC over analytic posterior),
                    "plugin" (GLASS flow plug-in),
                    "fmrg" (flow map trajectory guidance, deterministic ODE endpoint)

        For "analytic"/"exact"/"plugin", the guidance is the Doob h-transform
        score scaled by 0.5 * sigma_t^2 = (1 - t)/t (memoryless schedule).
        For "fmrg", the guidance is added directly to the probability flow
        velocity with no noise-schedule scaling, matching the deterministic
        ODE d\tilde{X}_t = (b_t(\tilde{X}_t) + u_t^FMRG(\tilde{X}_t)) dt of
        \citet{huang2025guide}.
        """
        if not torch.is_tensor(t):
            t_tensor = torch.tensor(t, device=x.device, dtype=torch.float32)
        else:
            t_tensor = t

        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)

            if method == "analytic":
                V = self.get_analytic_value(x_in, t_tensor)
            elif method == "exact":
                V = self.get_numeric_exact_value(x_in, t_tensor)
            elif method == "fmrg":
                V = self.get_fmrg_value(x_in, t_tensor, num_inner_steps=fmrg_inner_steps)
            else:
                V = self.get_plugin_value(x_in, t_tensor, k_particles=k_particles)

            grad_V = torch.autograd.grad(V.sum(), x_in)[0]

            gnorm = torch.norm(grad_V, dim=1, keepdim=True)
            grad_V = torch.where(gnorm > 100.0, grad_V * (100.0 / (gnorm + 1e-6)), grad_V)

            b_t = self.interpolant.velocity(x_in, t_tensor)
            if method == "fmrg":
                total_vel = b_t - grad_V
            else:
                scale = (1.0 - t_tensor) / (t_tensor + 1e-6)
                total_vel = b_t - scale * grad_V

        return total_vel.detach()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_guided(
    sampler, num_samples, num_steps=50, method="plugin", k_particles=1,
    fmrg_inner_steps=50, device="cpu",
):
    """Generate guided samples by integrating the Doob h-transform ODE.

    Uses a Heun (2nd-order) integrator over t in [0.001, 0.999].

    Args:
        sampler: GuidedSampler instance
        num_samples: number of trajectories
        num_steps: number of integration steps
        method: "analytic", "exact", "plugin", or "fmrg"
        k_particles: number of plug-in particles (only used for method="plugin")
        fmrg_inner_steps: number of Heun steps for the inner ODE (only used for method="fmrg")
        device: torch device
    Returns:
        samples: (num_samples, D) terminal samples
        rewards: (num_samples,) rewards at terminal samples
    """
    x = torch.randn(num_samples, sampler.D, device=device)
    ts = np.linspace(0.001, 0.999, num_steps)
    dt = ts[1] - ts[0]

    with torch.no_grad():
        for i in trange(len(ts) - 1, desc=f"Guided ODE ({method}, k={k_particles})"):
            t = ts[i]
            t_next = ts[i + 1]
            kp = k_particles if method == "plugin" else 1

            v1 = sampler.vector_field(
                x, t, method=method, k_particles=kp, fmrg_inner_steps=fmrg_inner_steps,
            )
            x_pred = x + dt * v1
            v2 = sampler.vector_field(
                x_pred, t_next, method=method, k_particles=kp, fmrg_inner_steps=fmrg_inner_steps,
            )
            x = x + (dt / 2.0) * (v1 + v2)

    with torch.no_grad():
        rewards = sampler.reward_fn(x)

    return x.detach().cpu(), rewards.detach().cpu()


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def cache_path(output_dir, reward, method, k, lam, sigma_damp=None):
    if method == "plugin":
        name = f"plugin_{reward}_k{k}_lam{lam}"
    elif method == "fmrg":
        name = f"fmrg_{reward}_lam{lam}"
    else:
        name = f"{method}_{reward}_lam{lam}"
    if sigma_damp is not None:
        name += f"_damp{sigma_damp}"
    return os.path.join(output_dir, name + ".npz")


def load_cache(output_dir, reward, method, k, lam, sigma_damp=None):
    path = cache_path(output_dir, reward, method, k, lam, sigma_damp)
    if os.path.exists(path):
        data = np.load(path)
        return torch.tensor(data["samples"]), torch.tensor(data["rewards"])
    return None, None


def save_cache(output_dir, reward, method, k, lam, samples, rewards, sigma_damp=None):
    os.makedirs(output_dir, exist_ok=True)
    path = cache_path(output_dir, reward, method, k, lam, sigma_damp)
    np.savez(path, samples=samples.numpy(), rewards=rewards.numpy())
    print(f"Cached {samples.shape[0]} samples to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device

    configs = {
        "gaussian":    (GAUSSIAN_WEIGHTS,     GAUSSIAN_MEANS,     GAUSSIAN_COVS,     GAUSSIAN_TARGET,    GAUSSIAN_SIGMA_HYPER,     "analytic"),
        "quadratic":   (QUADRATIC_WEIGHTS,   QUADRATIC_MEANS,   QUADRATIC_COVS,   QUADRATIC_TARGET,   QUADRATIC_SIGMA_HYPER,   "analytic"),
        "double_well": (DOUBLE_WELL_WEIGHTS,  DOUBLE_WELL_MEANS,  DOUBLE_WELL_COVS,  None,               DOUBLE_WELL_SIGMA_HYPER,  "exact"),
        "noniso":      (NONISO_WEIGHTS,       NONISO_MEANS,       NONISO_COVS,       NONISO_TARGET,      NONISO_SIGMA_HYPER,       "analytic"),
        "unequal":     (UNEQUAL_WEIGHTS,      UNEQUAL_MEANS,      UNEQUAL_COVS,      UNEQUAL_TARGET,     UNEQUAL_SIGMA_HYPER,      "analytic"),
        "uncentered":  (UNCENTERED_WEIGHTS,   UNCENTERED_MEANS,   UNCENTERED_COVS,   UNCENTERED_TARGET,  UNCENTERED_SIGMA_HYPER,   "analytic"),
    }
    weights, means, covs, target_cfg, sigma_hyper, default_method = configs[args.reward]

    gmm = GaussianMixture(weights, means, covs).to(device)
    interpolant = LinearInterpolant(gmm).to(device)

    if target_cfg is not None:
        target = target_cfg.to(device)
        reward_fn = lambda x: reward_quadratic(x, target)
    else:
        target = None
        reward_fn = reward_double_well

    method = args.method if args.method is not None else default_method

    sampler = GuidedSampler(
        interpolant,
        reward_fn,
        lam=args.lam,
        sigma_hyper=sigma_hyper,
        apply_damping_scale=args.sigma_damp is not None,
        target=target,
    ).to(device)

    if args.sigma_damp is not None:
        sampler.sigma_hyper = args.sigma_damp

    cached_samples, cached_rewards = load_cache(
        args.output_dir, args.reward, method, args.k, args.lam, args.sigma_damp
    )
    if cached_samples is not None and cached_samples.shape[0] >= args.num_samples:
        print(f"Loaded {args.num_samples} samples from cache.")
        return

    samples, rewards = sample_guided(
        sampler,
        args.num_samples,
        num_steps=args.num_steps,
        method=method,
        k_particles=args.k,
        fmrg_inner_steps=args.fmrg_inner_steps,
        device=device,
    )
    save_cache(
        args.output_dir, args.reward, method, args.k, args.lam, samples, rewards, args.sigma_damp
    )
    print(f"Average reward: {rewards.mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guided sampling on Gaussian mixture")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument(
        "--reward",
        type=str,
        default="quadratic",
        choices=["gaussian", "quadratic", "double_well", "noniso", "unequal", "uncentered"],
        help="Reward/GMM variant",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["analytic", "exact", "plugin", "fmrg"],
        help="Guidance method (default: analytic for quadratic, exact for quartic)",
    )
    parser.add_argument("--k", type=int, default=1, help="Number of plug-in particles")
    parser.add_argument(
        "--fmrg-inner-steps",
        type=int,
        default=50,
        help="Number of Heun steps for the inner ODE in FMRG mode",
    )
    parser.add_argument("--lam", type=float, default=3.0, help="Reward scale lambda")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--num-steps", type=int, default=50, help="Heun integration steps")
    parser.add_argument(
        "--sigma-damp",
        type=float,
        default=None,
        help="Reward damping sigma (None = use default sigma_hyper for reward)",
    )
    args = parser.parse_args()
    main(args)
