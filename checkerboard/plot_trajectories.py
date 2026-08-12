import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import VelocityMLP, DEFAULT_REWARD_CENTER
from sample import sample_guided, sample_unguided
from plot_checkerboards import _checker_background
from plot_comparison import compute_metrics

def plot_trajectories():
    device = "cpu"
    num_samples = 500  # 500 for stable W2, we only plot first 20
    lam = 50.0
    reward_center = torch.tensor([0.5, 1.5], dtype=torch.float32)
    sigma_r = 0.5
    
    # Load model
    checkpoint = torch.load("results/velocity_net.pt", map_location=device, weights_only=False)
    model_args = checkpoint["args"]
    rescale = model_args["rescale"]
    velocity_net = VelocityMLP(
        hidden_dim=model_args["hidden_dim"],
        num_layers=model_args["num_layers"],
        rescale=rescale,
    ).to(device)
    velocity_net.load_state_dict(checkpoint["model"])
    velocity_net.eval()
    
    methods = ["plugin", "second_order"]
    trajs = {}
    
    for method in methods:
        print(f"Sampling trajectory for {method} with 200 steps...")
        torch.manual_seed(42)
        samples, rewards, traj = sample_guided(
            velocity_net, num_samples, lam, reward_center, sigma_r,
            num_ode_steps=200, method=method, sigma_schedule="memoryless",
            rescale=rescale, device=device, return_traj=True
        )
        trajs[method] = traj
        
        _, _, _, _, w2 = compute_metrics(samples, rewards, lam, reward_center.numpy(), is_unguided=False)
        print(f"W2 for {method} at 200 steps: {w2:.4f}")
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    for i, method in enumerate(methods):
        ax = axes[i]
        _checker_background(ax)
        
        traj = trajs[method]
        traj_np = np.stack(traj, axis=1) # (num_samples, 201, 2)
        
        for k in range(20):
            x_vals = traj_np[k, :, 0]
            y_vals = traj_np[k, :, 1]
            ax.plot(x_vals, y_vals, color='blue', alpha=0.5, linewidth=1)
            ax.scatter(x_vals[-1], y_vals[-1], color='red', s=20, zorder=5)
            ax.scatter(x_vals[0], y_vals[0], color='black', s=10, zorder=5)
            
        ax.scatter(reward_center[0], reward_center[1], c="red", s=200, marker="X", edgecolors="black", linewidths=1, zorder=10)
        ax.set_title(f"Method: {method} (200 steps)")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("results/trajectories_lam50_steps200.png", dpi=200)
    print("Saved results/trajectories_lam50_steps200.png")

if __name__ == "__main__":
    plot_trajectories()
