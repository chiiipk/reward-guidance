"""Guided sampling with analytic GLASS flows, plug-in guidance, and best-of-n selection."""

import argparse
import os
import torch
import numpy as np
import random
from tqdm import trange

from model import analytic_denoiser, analytic_velocity, REWARD_FUNCTIONS


def glass_flow(t, x, mu, sigma, num_steps=50, device="cpu"):
    """Sample from (X_1 | I_t = x) via the GLASS flow ODE with the analytic denoiser."""
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    B = x.shape[0]
    x_bar = torch.randn(B, 1, device=device)

    snr_t = t / (1.0 - t)
    ds = 1.0 / num_steps

    for i in range(num_steps):
        s = i * ds
        if s >= 1.0 - 1e-6:
            break

        snr_s = s / (1.0 - s) if s > 0 else 0.0
        eta_star = np.sqrt(snr_t ** 2 + snr_s ** 2)
        tau_star = eta_star / (1.0 + eta_star)

        coeff = eta_star * (1.0 + eta_star)
        x_star = (snr_t / (1.0 - t) * x + snr_s / (1.0 - s) * x_bar) / coeff

        D = analytic_denoiser(tau_star, x_star, mu, sigma)

        dx = (D - x_bar) / (1.0 - s)
        x_bar = x_bar + dx * ds

    return x_bar


def compute_plugin_guidance(t, x, mu, sigma, lam, reward_name, reward_kwargs,
                            k=1, glass_steps=50, device="cpu"):
    """Compute grad_x log hat{h}_t^{(k)}(x) via GLASS + autograd."""
    x_input = x.detach().unsqueeze(-1).requires_grad_(True)
    reward_fn = REWARD_FUNCTIONS[reward_name]

    log_rewards = []
    for _ in range(k):
        x1_sample = glass_flow(t, x_input, mu, sigma, num_steps=glass_steps, device=device)
        r = reward_fn(x1_sample, **reward_kwargs)
        log_rewards.append(lam * r)

    log_rewards = torch.stack(log_rewards, dim=0)
    log_h = torch.logsumexp(log_rewards, dim=0) - np.log(k)

    # Step-function rewards have zero gradient a.e., so no grad_fn exists
    if log_h.grad_fn is None:
        return torch.zeros_like(x_input)
    grad = torch.autograd.grad(log_h.sum(), x_input)[0]
    return grad.detach()


def sample_guided(num_samples, mu, sigma, lam, reward_name, reward_kwargs,
                  k=1, num_ode_steps=200, glass_steps=50, device="cpu",
                  record_trajectories=False):
    """Generate guided samples with optional trajectory recording."""
    reward_fn = REWARD_FUNCTIONS[reward_name]

    x = torch.randn(num_samples, device=device)
    dt = 1.0 / num_ode_steps
    # Cap sigma_t^2 at the value at the second step to avoid the t=0 singularity
    sigma_max_sq = 2.0 * (1.0 - dt) / dt

    trajectories = []
    if record_trajectories:
        trajectories.append(x.detach().cpu().numpy().copy())

    with torch.no_grad():
        for step in trange(num_ode_steps, desc="Guided ODE"):
            t = step * dt
            b = analytic_velocity(t, x, mu, sigma)

            sigma_t_sq = min(2.0 * (1.0 - t) / t, sigma_max_sq) if t > 0 else sigma_max_sq

            with torch.enable_grad():
                guidance = compute_plugin_guidance(
                    t if t > 0 else dt, x, mu, sigma, lam,
                    reward_name, reward_kwargs,
                    k=k, glass_steps=glass_steps, device=device,
                ).squeeze(-1)

            x = x + (b + 0.5 * sigma_t_sq * guidance) * dt

            if record_trajectories:
                trajectories.append(x.detach().cpu().numpy().copy())

    with torch.no_grad():
        rewards = reward_fn(x, **reward_kwargs)

    traj_array = np.stack(trajectories, axis=0) if record_trajectories else None
    return x.detach().cpu(), rewards.detach().cpu(), traj_array


def sample_unguided(num_samples, mu, sigma, num_ode_steps=200, device="cpu",
                    record_trajectories=False):
    """Generate unguided samples with Heun integrator."""
    x = torch.randn(num_samples, device=device)
    dt = 1.0 / num_ode_steps

    trajectories = []
    if record_trajectories:
        trajectories.append(x.detach().cpu().numpy().copy())

    def vel(t_val, x_val):
        return analytic_velocity(t_val, x_val, mu, sigma)

    with torch.no_grad():
        for step in trange(num_ode_steps, desc="Unguided ODE"):
            t_i = step * dt
            v0 = vel(t_i, x)
            x_pred = x + dt * v0
            v1 = vel(t_i + dt, x_pred)
            x = x + 0.5 * dt * (v0 + v1)

            if record_trajectories:
                trajectories.append(x.detach().cpu().numpy().copy())

    traj_array = np.stack(trajectories, axis=0) if record_trajectories else None
    return x.detach().cpu(), traj_array


def best_of_n(samples, rewards, n):
    """Select the best sample from each group of $n$."""
    M = samples.shape[0]
    num_groups = M // n
    samples = samples[: num_groups * n].reshape(num_groups, n)
    rewards = rewards[: num_groups * n].reshape(num_groups, n)
    best_idx = rewards.argmax(dim=1)
    selected = samples[torch.arange(num_groups), best_idx]
    selected_rewards = rewards[torch.arange(num_groups), best_idx]
    return selected, selected_rewards, best_idx


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device
    mu, sigma = args.mu, args.sigma

    reward_kwargs = {}
    if args.reward == "step":
        reward_kwargs = {"R": args.R}
    elif args.reward == "gaussian":
        reward_kwargs = {"center": args.reward_center, "sigma_r": args.sigma_r}

    print("Generating unguided samples...")
    unguided, unguided_traj = sample_unguided(
        args.num_samples, mu, sigma, num_ode_steps=args.num_ode_steps,
        device=device, record_trajectories=args.record_trajectories,
    )

    total = args.num_samples * args.max_n
    print(f"Generating {total} guided samples (for best-of-{args.max_n})...")
    guided, guided_rewards, guided_traj = sample_guided(
        total, mu, sigma, args.lam, args.reward, reward_kwargs,
        k=args.k, num_ode_steps=args.num_ode_steps, glass_steps=args.glass_steps,
        device=device, record_trajectories=args.record_trajectories,
    )

    out_dir = os.path.join(args.output_dir, f"{args.reward}_lam{args.lam}")
    os.makedirs(out_dir, exist_ok=True)

    np.savez(
        os.path.join(out_dir, "samples.npz"),
        unguided=unguided.numpy(),
        guided=guided.numpy(),
        guided_rewards=guided_rewards.numpy(),
        mu=mu, sigma=sigma,
        lam=args.lam, reward_name=args.reward,
        max_n=args.max_n, num_samples=args.num_samples,
    )

    if args.record_trajectories:
        np.savez(
            os.path.join(out_dir, "trajectories.npz"),
            unguided=unguided_traj, guided=guided_traj,
        )

    print(f"\n{'='*60}")
    print(f"Mode selection statistics (reward={args.reward}, lam={args.lam})")
    print(f"{'='*60}")
    print(f"Unguided: P(X >= 0) = {(unguided >= 0).float().mean():.4f}")
    print(f"Guided (k={args.k}, n=1): P(X >= 0) = {(guided >= 0).float().mean():.4f}")

    for n in [2, 4, 8, 16, 32]:
        if n > args.max_n:
            break
        sel, _, _ = best_of_n(guided, guided_rewards, n)
        p = (sel >= 0).float().mean()
        theory = 1 - 0.5 ** n if args.reward == "step" else None
        theory_str = f" (theory: {theory:.4f})" if theory is not None else ""
        print(f"Best-of-{n:2d}: P(X >= 0) = {p:.4f}{theory_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--max-n", type=int, default=16)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--lam", type=float, default=5.0)
    parser.add_argument("--reward", type=str, default="step", choices=["step", "gaussian", "linear"])
    parser.add_argument("--R", type=float, default=10.0)
    parser.add_argument("--reward-center", type=float, default=3.0)
    parser.add_argument("--sigma-r", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=3.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--num-ode-steps", type=int, default=200)
    parser.add_argument("--glass-steps", type=int, default=50)
    parser.add_argument("--record-trajectories", action="store_true")
    args = parser.parse_args()
    main(args)
