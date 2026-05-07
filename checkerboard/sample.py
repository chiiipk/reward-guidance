"""Guided sampling via GLASS flow + plug-in estimator, with best-of-n and caching.

The Heun (predictor-corrector) integrator and adaptive-Gaussian initialization
follow Nicholas Boffi's jax-interpolants reference implementation
(https://github.com/nmboffi/jax-interpolants).
"""

import argparse
import os
import torch
import numpy as np
import random

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
    velocity_net, t, x, lam, reward_center, sigma_r, k=1, glass_steps=50, device="cpu"
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
    Returns:
        guidance: (B, 2) gradient of log hat{h}
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
# Guided ODE integration
# ---------------------------------------------------------------------------


def _damped_lam(lam, t, sigma_damp):
    """Compute the time-dependent reward scale lambda_t for reward damping.

    lambda_t = lambda / (1 + 2 * lambda * sigma_{1|t}^2)
    where sigma_{1|t}^2 = sigma^2 (1-t)^2 / ((1-t)^2 + t^2 sigma^2).
    """
    sigma_1t_sq = sigma_damp ** 2 * (1 - t) ** 2 / ((1 - t) ** 2 + t ** 2 * sigma_damp ** 2)
    return lam / (1 + 2 * lam * sigma_1t_sq)


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
    rescale=1.73,
    device="cpu",
    method="plugin",
    fmrg_inner_steps=50,
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
        glass_steps: Euler steps for inner GLASS ODE
        sigma_schedule: if float, constant schedule; if "memoryless", use clamped memoryless
        sigma_damp: if set, use reward damping with this sigma (None = no damping)
        rescale: std of the data; initial noise is rescale * N(0, I). Loaded from checkpoint.
        device: torch device
        method: "plugin" (k-particle GLASS plug-in) or "fmrg" (flow map trajectory guidance)
        fmrg_inner_steps: number of Heun steps for the inner ODE in FMRG mode
    Returns:
        samples: (num_samples, 2) terminal samples
        rewards: (num_samples,) rewards at terminal samples
    """
    velocity_net.eval()
    reward_center = reward_center.to(device)

    x = rescale * torch.randn(num_samples, 2, device=device)
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

    # Evaluate terminal rewards
    with torch.no_grad():
        rewards = reward_fn(x, reward_center, sigma_r)

    return x.detach().cpu(), rewards.detach().cpu()


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


def cache_path(output_dir, k, lam, sigma_damp=None, method="plugin"):
    if method == "fmrg":
        name = f"guided_fmrg_lam{lam}"
    else:
        name = f"guided_k{k}_lam{lam}"
    if sigma_damp is not None:
        name += f"_damp{sigma_damp}"
    return os.path.join(output_dir, name + ".npz")


def load_cache(output_dir, k, lam, sigma_damp=None, method="plugin"):
    path = cache_path(output_dir, k, lam, sigma_damp, method=method)
    if os.path.exists(path):
        data = np.load(path)
        return torch.tensor(data["samples"]), torch.tensor(data["rewards"])
    return None, None


def save_cache(output_dir, k, lam, samples, rewards, sigma_damp=None, method="plugin"):
    os.makedirs(output_dir, exist_ok=True)
    path = cache_path(output_dir, k, lam, sigma_damp, method=method)
    np.savez(path, samples=samples.numpy(), rewards=rewards.numpy())
    print(f"Cached {samples.shape[0]} samples to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device
    reward_center = DEFAULT_REWARD_CENTER.to(device)

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
        method_str = "fmrg" if args.method == "fmrg" else f"k={args.k}"
        print(
            f"Loaded {guided_samples.shape[0]} guided samples ({method_str}) from cache."
        )
    else:
        damp_str = f", σ_damp={sigma_damp}" if sigma_damp is not None else ""
        method_str = "fmrg" if args.method == "fmrg" else f"k={args.k}"
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
        choices=["plugin", "fmrg"],
        help="Guidance method: 'plugin' (k-particle GLASS plug-in) or 'fmrg' "
             "(flow map trajectory guidance, deterministic ODE endpoint)",
    )
    parser.add_argument(
        "--fmrg-inner-steps",
        type=int,
        default=50,
        help="Number of Heun steps for the inner ODE in FMRG mode",
    )
    args = parser.parse_args()

    # Parse sigma_schedule
    if args.sigma_schedule != "memoryless":
        args.sigma_schedule = float(args.sigma_schedule)

    main(args)
