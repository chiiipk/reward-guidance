"""Train the flow matching velocity network on the checkerboard distribution.

The training loop (Heun integrator, EMA decay, adaptive Gaussian rescaling of
the noise std to match the data std) follows Nicholas Boffi's jax-interpolants
reference implementation (https://github.com/nmboffi/jax-interpolants).
"""

import argparse
import copy
import os
from pathlib import Path

import torch
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import trange

from model import VelocityMLP, sample_checkerboard, checkerboard_density, checkerboard_rescale

REPO = Path(__file__).resolve().parents[1]
plt.style.use(str(REPO / "assets" / "default.mplstyle"))


def generate_samples(model, num_samples, num_steps=200, device="cpu",
                     snapshot_times=None, rescale=1.0):
    """Generate samples by integrating the learned flow ODE with RK4.

    Args:
        snapshot_times: optional list of times in [0, 1] at which to save snapshots.
            If provided, returns (final_samples, dict of {t: samples_at_t}).
        rescale: std of the data; initial noise is rescale * N(0, I).
    """
    model.eval()
    x = rescale * torch.randn(num_samples, 2, device=device)
    dt = 1.0 / num_steps

    snapshots = {}
    snapshot_steps = set()
    if snapshot_times is not None:
        for st in snapshot_times:
            step_idx = int(round(st / dt))
            snapshot_steps.add(step_idx)

    def vel(t_val, x_val):
        t_tensor = torch.full((num_samples,), t_val, device=device)
        return model(t_tensor, x_val)

    with torch.no_grad():
        for i in range(num_steps):
            if i in snapshot_steps:
                snapshots[i * dt] = x.detach().cpu()
            t_i = i * dt
            # Heun (2nd-order predictor-corrector), matching nmboffi/jax-interpolants
            v0 = vel(t_i, x)
            x_pred = x + dt * v0
            v1 = vel(t_i + dt, x_pred)
            x = x + 0.5 * dt * (v0 + v1)

    model.train()
    final = x.detach().cpu()
    if snapshot_times is not None:
        snapshots[1.0] = final
        return final, snapshots
    return final


def plot_loss_curve(losses, image_dir):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    fig.savefig(os.path.join(image_dir, "loss_curve.pdf"))
    plt.close(fig)


def plot_samples(samples, image_dir, filename="train_samples.pdf",
                 xlim=(-3, 3), ylim=(-3, 3)):
    fig, ax = plt.subplots(figsize=(5, 5))
    s = samples.numpy()
    # Checkerboard background
    res = 200
    gx = np.linspace(-3, 3, res)
    gy = np.linspace(-3, 3, res)
    X, Y = np.meshgrid(gx, gy)
    pts = np.stack([X, Y], axis=-1)
    density = checkerboard_density(pts)
    ax.contourf(X, Y, density, levels=[0.5, 1.5], colors=["gray"], alpha=0.15)
    # Scatter
    ax.scatter(s[:, 0], s[:, 1], s=2, alpha=0.5, c="C0", edgecolors="none")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    fig.savefig(os.path.join(image_dir, filename))
    plt.close(fig)


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device

    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(os.path.dirname(args.output_dir), "images", "training")
    os.makedirs(image_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.output_dir, "velocity_net.pt")
    losses = []
    start_step = 1

    # Pre-generate a large pool of checkerboard samples and compute rescale
    pool_size = max(args.batch_size * 100, 500_000)
    print(f"Pre-generating {pool_size} checkerboard samples...", flush=True)
    data_pool = sample_checkerboard(pool_size, device=device)

    # Adaptive Gaussian rescaling: match noise std to data std (nmboffi/jax-interpolants)
    rescale = float(data_pool.std().item())
    print(f"Data std (rescale): {rescale:.4f}", flush=True)

    model = VelocityMLP(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rescale=rescale,
    ).to(device)
    ema_model = copy.deepcopy(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps, eta_min=0
    )

    # Resume from checkpoint if it exists and --resume is set
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1
        losses = checkpoint.get("losses", [])
        rescale = checkpoint["args"].get("rescale", rescale)
        # Fast-forward scheduler to correct LR
        for _ in range(checkpoint["step"]):
            scheduler.step()
        print(f"Resumed from step {start_step - 1}", flush=True)

    pbar = trange(start_step, args.num_steps + 1, desc="Training")
    for step in pbar:
        # Draw a random batch from the pool
        idx = torch.randint(0, pool_size, (args.batch_size,), device=device)
        x1 = data_pool[idx]
        # Adaptive Gaussian base: N(0, rescale^2 * I)
        x0 = rescale * torch.randn_like(x1)
        t = torch.rand(args.batch_size, device=device)

        # Linear interpolant: I_t = (1 - t) x0 + t x1
        xt = (1.0 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1

        # Target velocity: dI_t/dt = x1 - x0
        target = x1 - x0

        # Flow matching loss
        pred = model(t, xt)
        loss = torch.mean((pred - target) ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        # EMA update (decay 0.9999, following nmboffi/jax-interpolants)
        update_ema(ema_model, model, decay=0.9999)

        losses.append(loss.item())
        if step % 10 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Checkpoint every save_every steps
        if step % args.save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "losses": losses,
                "args": {**vars(args), "rescale": rescale},
            }, checkpoint_path)

    # Final save
    torch.save({
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": args.num_steps,
        "losses": losses,
        "args": {**vars(args), "rescale": rescale},
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}", flush=True)

    # Save loss curve
    np.save(os.path.join(args.output_dir, "losses.npy"), np.array(losses))
    plot_loss_curve(losses, image_dir)
    print(f"Loss curve saved to {image_dir}/loss_curve.pdf", flush=True)

    # Generate and plot samples from EMA model
    print("Generating samples with snapshots...", flush=True)
    snapshot_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    samples, snapshots = generate_samples(model, 5000, device=device,
                                          snapshot_times=snapshot_times,
                                          rescale=rescale)
    plot_samples(samples, image_dir)
    print(f"Sample plot saved to {image_dir}/train_samples.pdf", flush=True)

    # Save snapshot plots
    snap_dir = os.path.join(image_dir, "flow_snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    for t_snap, x_snap in sorted(snapshots.items()):
        plot_samples(x_snap, snap_dir, filename=f"t_{t_snap:.2f}.pdf")
    print(f"Snapshot plots saved to {snap_dir}/", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flow matching on checkerboard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=2000, help="Checkpoint every N steps")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    train(args)
