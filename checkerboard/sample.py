"""Guided sampling via GLASS flow + plug-in estimator, with best-of-n and caching.

The Heun (predictor-corrector) integrator and adaptive-Gaussian initialization
follow Nicholas Boffi's jax-interpolants reference implementation
(https://github.com/nmboffi/jax-interpolants).
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
from tqdm import trange

from model import VelocityMLP, denoiser, reward_fn, DEFAULT_REWARD_CENTER


# ---------------------------------------------------------------------------
# GLASS flow for transition sampling
# ---------------------------------------------------------------------------


def glass_flow(velocity_net, t, x, num_steps=50, device="cpu"):
    """Sample from (X_1 | X_t = x) using the GLASS flow ODE.

    Simulates d X_bar_s = (1/(1-s)) (D_{tau*}(X*) - X_bar_s) ds
    from s=0 to s=1, starting from X_bar_0 ~ N(0, I).

    Args:
        velocity_net: learned velocity field
        t: scalar, current time in [0, 1)
        x: (B, 2) current positions
        num_steps: number of Euler steps for the GLASS ODE
        device: torch device
    Returns:
        X_bar_1: (B, 2) samples from (X_1 | X_t = x)
    """
    B = x.shape[0]
    x_bar = torch.randn(B, 2, device=device)

    snr_t = t / (1.0 - t)  # signal-to-noise ratio for the conditioning observation

    ds = 1.0 / num_steps
    for i in range(num_steps):
        s = i * ds
        if s >= 1.0 - 1e-6:
            break

        snr_s = s / (1.0 - s) if s > 0 else 0.0
        eta_star = np.sqrt(snr_t**2 + snr_s**2)
        tau_star = eta_star / (1.0 + eta_star)

        # Effective observation: combine conditioning x and current x_bar
        coeff = eta_star * (1.0 + eta_star)
        x_star = (snr_t / (1.0 - t) * x + snr_s / (1.0 - s) * x_bar) / coeff

        # Evaluate denoiser at effective time and observation
        tau_tensor = torch.full((B,), tau_star, device=device)
        D = denoiser(velocity_net, tau_tensor, x_star)

        # Euler step
        dx = (D - x_bar) / (1.0 - s)
        x_bar = x_bar + dx * ds

    return x_bar


# ---------------------------------------------------------------------------
# Plug-in guidance
# ---------------------------------------------------------------------------


def compute_plugin_guidance(
    velocity_net, t, x, lam, reward_center, sigma_r, k=1, glass_steps=50, device="cpu", return_log_h=False
):
    """Compute the plug-in guidance term nabla_x log hat{h}_t^{(k)}(x).

    Uses GLASS flow for transition sampling and autograd for the gradient.

    Args:
        velocity_net: learned velocity field
        t: scalar, current time
        x: (B, 2) current positions (will be detached and re-attached for grad)
        lam: reward scale lambda
        reward_center: (2,) center of Gaussian bump reward
        sigma_r: width of Gaussian bump
        k: number of particles for plug-in estimator
        glass_steps: number of GLASS ODE steps
        device: torch device
        return_log_h: whether to return the raw log_h scalar tensor
    Returns:
        guidance: (B, 2) gradient of log hat{h}
        (optional) log_h: (B,) log hat{h}
    """
    x_input = x.detach().requires_grad_(True)

    # Draw k samples from (X_1 | X_t = x) via GLASS and average
    log_rewards = []
    for _ in range(k):
        x1_sample = glass_flow(
            velocity_net, t, x_input, num_steps=glass_steps, device=device
        )
        r = reward_fn(x1_sample, reward_center, sigma_r)
        log_rewards.append(lam * r)

    # log hat{h} = log(1/k sum_i exp(lam * r_i))
    # = log_sum_exp(lam * r_i) - log(k)
    log_rewards = torch.stack(log_rewards, dim=0)  # (k, B)
    log_h = torch.logsumexp(log_rewards, dim=0) - np.log(k)  # (B,)

    # Gradient w.r.t. x
    grad_log_h = torch.autograd.grad(log_h.sum(), x_input)[0]
    
    if return_log_h:
        return grad_log_h.detach(), log_h.detach()
    return grad_log_h.detach()


# ---------------------------------------------------------------------------
# Flow map trajectory guidance (FMRG)
# ---------------------------------------------------------------------------


def compute_fmrg_guidance(
    velocity_net, t, x, lam, reward_center, sigma_r, num_inner_steps=50, device="cpu"
):
    """Compute the flow map trajectory guidance term lam * grad_x r(X_{t,1}(x)).

    Integrates the unguided probability flow ODE forward from t to 1 with autograd
    enabled, then takes the gradient of the reward at the deterministic endpoint.

    Args:
        velocity_net: learned velocity field
        t: scalar, current time in [0, 1)
        x: (B, 2) current positions
        lam: reward scale lambda
        reward_center: (2,) center of Gaussian bump reward
        sigma_r: width of Gaussian bump
        num_inner_steps: number of Heun steps for the inner ODE from t to 1
        device: torch device
    Returns:
        guidance: (B, 2) gradient of lam * r(X_{t,1}(x)) w.r.t. x
    """
    x_input = x.detach().requires_grad_(True)
    B = x_input.shape[0]

    dt = (1.0 - t) / num_inner_steps
    x_curr = x_input
    for k in range(num_inner_steps):
        s = t + k * dt
        s_tensor = torch.full((B,), s, device=device)
        s_next_tensor = torch.full((B,), s + dt, device=device)
        v0 = velocity_net(s_tensor, x_curr)
        x_pred = x_curr + dt * v0
        v1 = velocity_net(s_next_tensor, x_pred)
        x_curr = x_curr + 0.5 * dt * (v0 + v1)

    log_h = lam * reward_fn(x_curr, reward_center, sigma_r)
    grad_log_h = torch.autograd.grad(log_h.sum(), x_input)[0]
    return grad_log_h.detach()


# ---------------------------------------------------------------------------
# Second-Order guidance (Analytical Integral)
# ---------------------------------------------------------------------------

def compute_second_order_guidance(
    velocity_net, t, x, lam, reward_center, sigma_r, sigma_t_sq, glass_steps=50, device="cpu", return_log_h=False
):
    """Compute the second-order Doob h-transform guidance term.

    Uses a second-order Taylor expansion of the reward r(X_1) around the expected
    X_1 given X_t=x (obtained via the GLASS flow). Evaluates the Gaussian integral
    exactly to form V_{2nd}(mu) and takes its gradient.

    Args:
        velocity_net: learned velocity field
        t: scalar, current time
        x: (B, 2) current positions
        lam: reward scale lambda
        reward_center: (2,) center of Gaussian bump reward
        sigma_r: width of Gaussian bump
        sigma_t_sq: conditional variance sigma_{1|t}^2
        glass_steps: number of GLASS ODE steps
        device: torch device
        return_log_h: whether to return the raw log_h scalar tensor
    Returns:
        guidance: (B, 2) gradient of log hat{h}_t
        (optional) log_h: (B,) log hat{h}
    """
    x_input = x.detach().requires_grad_(True)
    B = x_input.shape[0]

    # 1. Predict mu = E[X_1 | X_t=x_input]
    mu = glass_flow(velocity_net, t, x_input, num_steps=glass_steps, device=device)

    with torch.no_grad():
        # 2. Compute analytical grad and Hessian of r(mu) = exp(-||mu - c||^2 / (2s^2))
        r = reward_fn(mu, reward_center, sigma_r)  # (B,)
        diff = mu - reward_center
        grad_r_ana = -diff / (sigma_r ** 2) * r.unsqueeze(-1)  # (B, 2)
        
        I = torch.eye(2, device=device).unsqueeze(0).expand(B, -1, -1)
        H_r_ana = -1.0 / (sigma_r ** 2) * (
            I * r.unsqueeze(-1).unsqueeze(-1) +
            torch.bmm(diff.unsqueeze(-1), grad_r_ana.unsqueeze(1))
        )  # (B, 2, 2)

        # Clamp eigenvalues to be <= 0 to ensure precision is positive definite
        L, Q = torch.linalg.eigh(H_r_ana)
        L = torch.clamp(L, max=0.0)
        H_r_ana = torch.bmm(Q, torch.bmm(torch.diag_embed(L), Q.transpose(1, 2)))

        # 3. Compute the preconditioned gradient: (I - lam * sigma^2 * H)^-1 * (lam * grad_r)
        # We use torch.linalg.solve instead of Neumann series for exact inversion.
        I_minus_lam_sigma_H = I - lam * sigma_t_sq * H_r_ana
        lam_grad_r = lam * grad_r_ana
        precond_grad = torch.linalg.solve(I_minus_lam_sigma_H, lam_grad_r.unsqueeze(-1)).squeeze(-1)

        # Compute actual log Z for SMC if requested
        if return_log_h:
            # log det(I - lam * sigma_t_sq * H) = sum(log(1 - lam * sigma_t_sq * L_i))
            # using log1p for numerical stability as recommended
            log_det = torch.log1p(-lam * sigma_t_sq * L).sum(-1)
            # quadratic term = 1/2 * lam * sigma_t_sq * grad_r^T * precond_grad
            # since precond_grad is already (I - lam*sig^2*H)^-1 * (lam * grad_r)
            # the term is 1/2 * sigma_t_sq * grad_r^T precond_grad
            quad_term = 0.5 * sigma_t_sq * lam * (grad_r_ana * precond_grad).sum(-1)
            log_h_val = lam * r - 0.5 * log_det + quad_term

    # 4. We want to apply the chain rule J_mu^T * precond_grad.
    # We do this efficiently via a surrogate scalar for autograd.
    log_h_surrogate = (mu * precond_grad).sum()
    grad_V = torch.autograd.grad(log_h_surrogate, x_input)[0]
    
    if return_log_h:
        return grad_V.detach(), log_h_val.detach()
    return grad_V.detach()


# ---------------------------------------------------------------------------
# Guided ODE integration
# ---------------------------------------------------------------------------


def _posterior_var(t, sigma_data):
    """Var[X_1 | X_t]  — dùng cho CẢ damping schedule LẪN Laplace preconditioner."""
    return sigma_data**2 * (1-t)**2 / ((1-t)**2 + t**2 * sigma_data**2)


def _damped_lam(lam, t, sigma_data):
    """Compute the time-dependent reward scale lambda_t for reward damping.

    lambda_t = lambda / (1 + 2 * lambda * sigma_{1|t}^2)
    """
    return lam / (1 + 2 * lam * _posterior_var(t, sigma_data))


def sample_guided(
    velocity_net,
    num_samples,
    lam,
    reward_center,
    sigma_r,
    k=1,
    num_ode_steps=200,
    glass_steps=50,
    sigma_schedule=1.0,
    sigma_damp=None,
    rescale=1.0,
    device="cpu",
    method="plugin",
    fmrg_inner_steps=10,
    return_traj=False,
):
    """Generate guided samples using the Doob h-transform with plug-in estimator.

    Integrates: d X_t = (b_t(X_t) + 0.5 * sigma_t^2 * grad log h_t(X_t)) dt
    using the memoryless noise schedule sigma_t^2 = 2(1-t)/t, clamped at the
    value it takes at the second time step t=dt to avoid the singularity at t=0.

    Args:
        velocity_net: learned velocity field
        num_samples: number of trajectories to generate
        lam: reward scale lambda
        reward_center: (2,) reward center
        sigma_r: reward width
        k: number of plug-in particles
        num_ode_steps: Euler steps for outer ODE
        method: "plugin" (k-particle GLASS plug-in) or "fmrg" (flow map trajectory guidance)
        sigma_schedule: if float, constant schedule; if "memoryless", use clamped memoryless
        rescale: std of the data; initial noise is rescale * N(0, I). Loaded from checkpoint.
        sigma_damp: if set, use reward damping with this sigma (None = no damping)
        fmrg_inner_steps: number of Heun steps for the inner ODE in FMRG mode
        glass_steps: Euler steps for inner GLASS ODE
        device: torch device
        k: number of plug-in particles
        return_traj: whether to return the full trajectory list
    Returns:
        samples: (num_samples, 2) terminal samples
        rewards: (num_samples,) rewards at terminal samples
        (optional) traj: list of (num_samples, 2) at each t
    """
    velocity_net.eval()
    reward_center = reward_center.to(device)

    x = rescale * torch.randn(num_samples, 2, device=device)
    traj = [x.detach().cpu()]
    dt = 1.0 / num_ode_steps
    # Cap sigma_t^2 at the value it takes at the second step t=dt
    sigma_max_sq = 2.0 * (1.0 - dt) / dt

    with torch.no_grad():
        for step in trange(num_ode_steps, desc=f"Guided ODE ({method})"):
            t = step * dt
            t_tensor = torch.full((num_samples,), t, device=device)

            # Base velocity at true t
            b = velocity_net(t_tensor, x)

            # Noise schedule: clamped memoryless (cap at sigma_max_sq to handle t=0)
            if isinstance(sigma_schedule, str) and sigma_schedule == "memoryless":
                sigma_t_sq = min(2.0 * (1.0 - t) / t, sigma_max_sq) if t > 0 else sigma_max_sq
            else:
                sigma_t_sq = float(sigma_schedule) ** 2

            # Posterior variance of X_1 given X_t
            sigma_1t_sq = _posterior_var(t, rescale)

            # Reward damping: replace constant lam with time-dependent lam_t
            lam_t = _damped_lam(lam, t, sigma_damp) if sigma_damp is not None else lam

            # Guidance (needs grad, so temporarily enable)
            with torch.enable_grad():
                if method == "fmrg":
                    guidance = compute_fmrg_guidance(
                        velocity_net,
                        t if t > 0 else dt,
                        x,
                        lam_t,
                        reward_center,
                        sigma_r,
                        num_inner_steps=fmrg_inner_steps,
                        device=device,
                    )
                elif method == "second_order":
                    guidance = compute_second_order_guidance(
                        velocity_net,
                        t if t > 0 else dt,
                        x,
                        lam_t,
                        reward_center,
                        sigma_r,
                        sigma_t_sq=sigma_1t_sq,
                        glass_steps=glass_steps,
                        device=device,
                    )
                else:
                    guidance = compute_plugin_guidance(
                        velocity_net,
                        t if t > 0 else dt,
                        x,
                        lam_t,
                        reward_center,
                        sigma_r,
                        k=k,
                        glass_steps=glass_steps,
                        device=device,
                    )

            # Euler step. FMRG adds the guidance directly with no noise-schedule
            # scaling (the Doob h-transform plug-in flow uses 0.5 * sigma_t^2 here
            # because it inherits the score-times-diffusion-coefficient structure).
            if method == "fmrg":
                x = x + (b + guidance) * dt
            else:
                x = x + (b + 0.5 * sigma_t_sq * guidance) * dt
            
            if return_traj:
                traj.append(x.detach().cpu())

    # Evaluate terminal rewards
    rewards = reward_fn(x, reward_center, sigma_r).cpu()
    if return_traj:
        return x.cpu().numpy(), rewards.numpy(), [t.numpy() for t in traj]
    return x.cpu().numpy(), rewards.numpy()


def sample_unguided(velocity_net, num_samples, num_ode_steps=200, device="cpu",
                    rescale=1.73):
    """Generate unguided samples from the base flow using RK4."""
    velocity_net.eval()

    x = rescale * torch.randn(num_samples, 2, device=device)
    dt = 1.0 / num_ode_steps

    def vel(t_val, x_val):
        t_tensor = torch.full((num_samples,), t_val, device=device)
        return velocity_net(t_tensor, x_val)

    with torch.no_grad():
        for step in trange(num_ode_steps, desc="Unguided ODE"):
            t_i = step * dt
            # Heun (2nd-order predictor-corrector), matching nmboffi/jax-interpolants
            v0 = vel(t_i, x)
            x_pred = x + dt * v0
            v1 = vel(t_i + dt, x_pred)
            x = x + 0.5 * dt * (v0 + v1)

    return x.detach().cpu()


# ---------------------------------------------------------------------------
# Best-of-n selection
# ---------------------------------------------------------------------------


def best_of_n(samples, rewards, n):
    """Select the best sample from each group of n.

    Args:
        samples: (M, 2) all guided samples
        rewards: (M,) corresponding rewards
        n: group size
    Returns:
        selected: (M // n, 2) best samples
        selected_rewards: (M // n,) their rewards
    """
    M = samples.shape[0]
    num_groups = M // n
    # Truncate to exact multiple of n
    samples = samples[: num_groups * n].reshape(num_groups, n, 2)
    rewards = rewards[: num_groups * n].reshape(num_groups, n)
    # Replace NaN rewards with -inf so they are never selected
    rewards_for_argmax = rewards.clone()
    rewards_for_argmax[torch.isnan(rewards_for_argmax)] = float("-inf")
    best_idx = rewards_for_argmax.argmax(dim=1)  # (num_groups,)
    selected = samples[torch.arange(num_groups), best_idx]
    selected_rewards = rewards[torch.arange(num_groups), best_idx]
    return selected, selected_rewards


def softmax_of_n(samples, rewards, n, lam):
    """Select a sample from each group of n using weights proportional to exp(lam * r).

    This constructs a discrete distribution over the n candidates with Boltzmann
    weights exp(lam * r(X_1^(i))) and samples from it. By the Laplace approximation,
    this converges to best-of-n as lam -> infinity.

    Args:
        samples: (M, 2) all guided samples
        rewards: (M,) corresponding rewards
        n: group size
        lam: reward scale (used for Boltzmann weights)
    Returns:
        selected: (M // n, 2) selected samples
        selected_rewards: (M // n,) their rewards
    """
    M = samples.shape[0]
    num_groups = M // n
    samples = samples[: num_groups * n].reshape(num_groups, n, 2)
    rewards_g = rewards[: num_groups * n].reshape(num_groups, n)

    # For n=1, selection is trivial
    if n == 1:
        return samples.squeeze(1), rewards_g.squeeze(1)

    # Numerically stable softmax: subtract max within each group
    # Replace NaN rewards with -inf so they get zero weight
    rewards_g = rewards_g.clone()
    nan_mask = torch.isnan(rewards_g)
    rewards_g[nan_mask] = float("-inf")
    log_w = lam * rewards_g  # (num_groups, n)
    max_log_w = log_w.max(dim=1, keepdim=True)[0]
    # If all entries are -inf (all NaN), fall back to uniform
    all_nan = (max_log_w == float("-inf"))
    max_log_w = max_log_w.clamp(min=-1e38)  # avoid -inf - (-inf) = nan
    log_w = log_w - max_log_w
    log_w[nan_mask] = float("-inf")
    w = torch.softmax(log_w, dim=1)  # (num_groups, n)
    # For all-NaN groups, use uniform weights over non-NaN entries (shouldn't happen in practice)
    if all_nan.any():
        uniform = torch.ones(n, device=samples.device) / n
        w[all_nan.squeeze(1)] = uniform

    indices = torch.multinomial(w, num_samples=1).squeeze(1)  # (num_groups,)
    selected = samples[torch.arange(num_groups), indices]
    selected_rewards = rewards_g[torch.arange(num_groups), indices]
    return selected, selected_rewards


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def load_cache(output_dir, k, lam, sigma_damp=None, method="plugin"):
    """Load cached guided samples if they exist."""
    suffix = f"_damp{sigma_damp}" if sigma_damp is not None else ""
    method_str = method if method != "plugin" else f"k{k}"
    path = os.path.join(output_dir, f"guided_{method_str}_lam{lam}{suffix}.npz")
    if os.path.exists(path):
        data = np.load(path)
        return torch.tensor(data["samples"]), torch.tensor(data["rewards"])
    return None, None


def save_cache(
    output_dir, k, lam, samples, rewards, sigma_damp, method="plugin"
):
    """Save guided samples to cache."""
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_damp{sigma_damp}" if sigma_damp is not None else ""
    method_str = method if method != "plugin" else f"k{k}"
    path = os.path.join(output_dir, f"guided_{method_str}_lam{lam}{suffix}.npz")
    np.savez(path, samples=samples.numpy(), rewards=rewards.numpy())
    print(f"Cached {samples.shape[0]} samples to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    # Load trained model
    checkpoint = torch.load(
        os.path.join(args.model_dir, "velocity_net.pt"),
        map_location=device,
        weights_only=False,
    )
    model_args = checkpoint["args"]
    rescale = model_args["rescale"]
    model = VelocityMLP(
        hidden_dim=model_args["hidden_dim"],
        num_layers=model_args["num_layers"],
        rescale=rescale,
    ).to(device)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded velocity network.")
    velocity_net = model

    reward_center = torch.tensor(args.reward_center, device=device)

    # Generate or load unguided samples
    unguided_path = os.path.join(args.output_dir, "unguided.npz")
    if os.path.exists(unguided_path):
        data = np.load(unguided_path)
        unguided_samples = torch.tensor(data["samples"])
        print(f"Loaded {unguided_samples.shape[0]} unguided samples from cache.")
    else:
        print("Generating unguided samples...")
        unguided_samples = sample_unguided(model, args.num_samples, device=device,
                                           rescale=rescale)
        os.makedirs(args.output_dir, exist_ok=True)
        np.savez(unguided_path, samples=unguided_samples.numpy())
        print(f"Saved {unguided_samples.shape[0]} unguided samples.")

    # Generate or load guided samples for this method, k, lam, and sigma_damp
    sigma_damp = args.sigma_damp
    cached_samples, cached_rewards = load_cache(
        args.output_dir, args.k, args.lam, sigma_damp, method=args.method
    )
    if cached_samples is not None and cached_samples.shape[0] >= args.num_samples:
        guided_samples = cached_samples[: args.num_samples]
        guided_rewards = cached_rewards[: args.num_samples]
        method_str = args.method if args.method != "plugin" else f"k={args.k}"
        print(
            f"Loaded {guided_samples.shape[0]} guided samples ({method_str}) from cache."
        )
    else:
        damp_str = f", σ_damp={sigma_damp}" if sigma_damp is not None else ""
        method_str = args.method if args.method != "plugin" else f"k={args.k}"
        print(f"Generating {args.num_samples} guided samples with {method_str}{damp_str}...")
        guided_samples, guided_rewards = sample_guided(
            model,
            args.num_samples,
            args.lam,
            reward_center,
            args.sigma_r,
            k=args.k,
            num_ode_steps=args.num_ode_steps,
            glass_steps=args.glass_steps,
            sigma_schedule=args.sigma_schedule,
            sigma_damp=sigma_damp,
            rescale=rescale,
            device=device,
            method=args.method,
            fmrg_inner_steps=args.fmrg_inner_steps,
        )
        save_cache(
            args.output_dir, args.k, args.lam, guided_samples, guided_rewards,
            sigma_damp, method=args.method,
        )

    # Best-of-n selection
    if args.n > 1:
        selected, selected_rewards = best_of_n(guided_samples, guided_rewards, args.n)
        print(f"Best-of-{args.n}: {selected.shape[0]} samples selected.")
        bon_path = os.path.join(args.output_dir, f"bon_k{args.k}_n{args.n}_lam{args.lam}.npz")
        np.savez(bon_path, samples=selected.numpy(), rewards=selected_rewards.numpy())
    else:
        print("n=1, no best-of-n selection.")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guided sampling on checkerboard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--model-dir", type=str, default="./results")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--k", type=int, default=1, help="Number of plug-in particles")
    parser.add_argument(
        "--n", type=int, default=1, help="Best-of-n selection (1 = no selection)"
    )
    parser.add_argument("--lam", type=float, default=5.0, help="Reward scale lambda")
    parser.add_argument("--sigma-r", type=float, default=1.5, help="Reward bump width")
    parser.add_argument(
        "--num-ode-steps", type=int, default=200, help="Euler steps for guided ODE"
    )
    parser.add_argument(
        "--glass-steps", type=int, default=50, help="Euler steps for GLASS ODE"
    )
    parser.add_argument(
        "--sigma-schedule",
        type=str,
        default="memoryless",
        help="Noise schedule: 'memoryless' or a float for constant",
    )
    parser.add_argument(
        "--sigma-damp",
        type=float,
        default=None,
        help="Reward damping sigma (None = no damping, i.e. constant lambda)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="plugin",
        choices=["plugin", "fmrg", "second_order"],
        help="Guidance method: 'plugin' (k-particle GLASS plug-in), 'fmrg' "
             "(flow map trajectory guidance), or 'second_order' (analytical Gaussian integral)",
    )
    parser.add_argument(
        "--fmrg-inner-steps",
        type=int,
        default=50,
        help="Number of Heun steps for the inner ODE in FMRG mode",
    )
    parser.add_argument(
        "--reward-center", type=float, nargs=2, default=[0.5, 0.5], help="Reward center (x, y)"
    )
    args = parser.parse_args()

    # Parse sigma_schedule
    if args.sigma_schedule != "memoryless":
        args.sigma_schedule = float(args.sigma_schedule)

    main(args)
def sample_smc(
    velocity_net,
    num_samples,
    num_particles,
    lam,
    reward_center,
    sigma_r,
    num_ode_steps=200,
    method="second_order",
    rescale=1.0,
    sigma_damp=None,
    device="cpu",
    t_stop=0.9,
):
    """Twisted Diffusion Sampler (SMC) using first/second order guidance as twist.
    
    Args:
        num_samples: B (number of independent groups)
        num_particles: n (particles per group)
        method: "plugin", "second_order", or "naive" (no guidance, only resampling)
        t_stop: time after which to stop resampling (since noise vanishes as t -> 1)
        
    Returns:
        samples: (B, 2) drawn proportionally to final weights
        rewards: (B,)
        ess_history: list of average ESS over time
        resample_count: total number of resampling steps triggered
    """
    velocity_net.eval()
    reward_center = reward_center.to(device)
    
    B, n = num_samples, num_particles
    
    # Initialize B * n particles
    # Shape: (B, n, 2)
    x = rescale * torch.randn(B, n, 2, device=device)
    
    dt = 1.0 / num_ode_steps
    sigma_max_sq = 2.0 * (1.0 - dt) / dt
    
    # log weights for each group
    logw = torch.zeros(B, n, device=device)
    
    # Initialize guidance and lh at t=0
    x_flat = x.view(B * n, 2)
    with torch.enable_grad():
        lam_t_0 = _damped_lam(lam, dt, sigma_damp) if sigma_damp is not None else lam
        if method in ["plugin", "naive"]:
            guidance_prev, lh_prev = compute_plugin_guidance(velocity_net, dt, x_flat, lam_t_0, reward_center, sigma_r, device=device, return_log_h=True) # use dt instead of 0 to avoid division by zero
        elif method == "second_order":
            sigma_1t_sq = _posterior_var(0.0, rescale)
            guidance_prev, lh_prev = compute_second_order_guidance(velocity_net, dt, x_flat, lam_t_0, reward_center, sigma_r, sigma_1t_sq, device=device, return_log_h=True)

    ess_history = []
    resample_count = 0
    
    # Track log_alpha percentiles
    log_alpha_history = []
    
    with torch.no_grad():
        for step in range(num_ode_steps):
            t = step * dt
            t_next = t + dt
            
            t_tensor = torch.full((B * n,), t, device=device)
            x_flat = x.view(B * n, 2)
            
            b = velocity_net(t_tensor, x_flat)
            
            # SDE noise schedule (memoryless)
            sigma_t_sq = min(2.0 * (1.0 - t) / t, sigma_max_sq) if t > 0 else sigma_max_sq
            sigma_t = math.sqrt(sigma_t_sq)
            
            # Compute guidance d
            d = torch.zeros_like(x_flat)
            if method != "naive":
                d = 0.5 * sigma_t_sq * guidance_prev * dt
            
            # Draw noise u
            eps = torch.randn_like(x_flat)
            u = sigma_t * math.sqrt(dt) * eps
            
            # Euler step
            x_new_flat = x_flat + b * dt + d + u
            
            # Compute log_h at new position (t_next) and cache guidance for next step
            if step < num_ode_steps - 1:
                with torch.enable_grad():
                    lam_t_next = _damped_lam(lam, t_next, sigma_damp) if sigma_damp is not None else lam
                    if method in ["plugin", "naive"]:
                        guidance_new, lh_new = compute_plugin_guidance(velocity_net, t_next, x_new_flat, lam_t_next, reward_center, sigma_r, device=device, return_log_h=True)
                    elif method == "second_order":
                        sigma_1t_sq_next = _posterior_var(t_next, rescale)
                        guidance_new, lh_new = compute_second_order_guidance(velocity_net, t_next, x_new_flat, lam_t_next, reward_center, sigma_r, sigma_1t_sq_next, device=device, return_log_h=True)
            else:
                # Terminal step: h_1 = exp(lam * r(x_1))
                lh_new = lam * reward_fn(x_new_flat, reward_center, sigma_r)
                guidance_new = torch.zeros_like(x_flat)
                
            log_h_new = lh_new.view(B, n)
            log_h_prev = lh_prev.view(B, n)
            d = d.view(B, n, 2)
            u = u.view(B, n, 2)
            
            # Compute log weight increment
            u_dot_d = (u * d).sum(-1)
            d_norm_sq = (d * d).sum(-1)
            
            log_alpha = -(2 * u_dot_d + d_norm_sq) / (2 * sigma_t_sq * dt) + log_h_new - log_h_prev
            logw = logw + log_alpha
            
            # Save percentiles for debugging
            log_alpha_history.append(torch.quantile(log_alpha.view(-1), torch.tensor([0.1, 0.5, 0.9], device=device)).cpu().numpy())
            
            # Compute ESS
            max_logw = logw.max(dim=1, keepdim=True)[0]
            if torch.isnan(max_logw).any() or torch.isinf(max_logw).any():
                # Fix NaNs by resetting those groups
                bad_mask = torch.isnan(max_logw) | torch.isinf(max_logw)
                bad_mask = bad_mask.squeeze(1)
                logw[bad_mask] = 0.0
                max_logw = logw.max(dim=1, keepdim=True)[0]
                
            w = torch.exp(logw - max_logw)
            W = w / w.sum(dim=1, keepdim=True)  # (B, n)
            
            ess = 1.0 / (W ** 2).sum(dim=1)  # (B,)
            ess_history.append(ess.mean().item())
            
            x = x_new_flat.view(B, n, 2)
            
            # Conditional Resampling
            if t < t_stop:
                mask = ess < (n / 2.0)
                if mask.any():
                    resample_count += mask.sum().item()
                    # For groups that need resampling
                    W_resample = W[mask]  # (K, n)
                    idx = torch.multinomial(W_resample, n, replacement=True)  # (K, n)
                    
                    # Gather the new particles for those groups
                    # x[mask] has shape (K, n, 2)
                    x_resampled = torch.gather(x[mask], 1, idx.unsqueeze(-1).expand(-1, -1, 2))
                    lh_new_resampled = torch.gather(lh_new.view(B, n)[mask], 1, idx).view(-1)
                    guidance_new_resampled = torch.gather(guidance_new.view(B, n, 2)[mask], 1, idx.unsqueeze(-1).expand(-1, -1, 2)).view(-1, 2)
                    
                    x[mask] = x_resampled
                    # Need to properly inject resampled lh and guidance back into flattened tensors
                    lh_new_view = lh_new.view(B, n)
                    lh_new_view[mask] = lh_new_resampled.view(-1, n)
                    lh_new = lh_new_view.view(-1)
                    
                    guidance_new_view = guidance_new.view(B, n, 2)
                    guidance_new_view[mask] = guidance_new_resampled.view(-1, n, 2)
                    guidance_new = guidance_new_view.view(-1, 2)
                    
                    logw[mask] = 0.0  # Reset log weights for resampled groups
                    
            lh_prev = lh_new
            guidance_prev = guidance_new
            
    # Print log_alpha stats
    la_arr = np.array(log_alpha_history)
    print(f"[{method}] log_alpha p10: min={la_arr[:, 0].min():.2f}, max={la_arr[:, 0].max():.2f}")
    print(f"[{method}] log_alpha p90: min={la_arr[:, 2].min():.2f}, max={la_arr[:, 2].max():.2f}")
    
    # Final step: sample 1 particle from each group based on final weights
    max_logw = logw.max(dim=1, keepdim=True)[0]
    # Check for NaNs
    if torch.isnan(max_logw).any() or torch.isinf(max_logw).any():
        print(f"WARNING: max_logw has NaNs/Infs in {method}!")
        print(f"logw min: {logw.min().item()}, max: {logw.max().item()}, nan count: {torch.isnan(logw).sum().item()}")
    
    w = torch.exp(logw - max_logw)
    W = w / w.sum(dim=1, keepdim=True)
    
    if torch.isnan(W).any():
        print("WARNING: W has NaNs. Falling back to uniform.")
        W = torch.ones_like(W) / n
    
    final_idx = torch.multinomial(W, 1)  # (B, 1)
    final_samples = torch.gather(x, 1, final_idx.unsqueeze(-1).expand(-1, -1, 2)).squeeze(1)  # (B, 2)
    
    rewards = reward_fn(final_samples, reward_center, sigma_r).cpu()
    return final_samples.cpu().numpy(), rewards.numpy(), ess_history, resample_count
