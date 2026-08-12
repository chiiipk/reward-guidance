import os
import time
import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import VelocityMLP, DEFAULT_REWARD_CENTER
from sample import sample_guided, sample_smc, best_of_n
from plot_checkerboards import _checker_background
from plot_comparison import compute_metrics

def main():
    device = "cpu"
    B = 100
    n = 8
    lam = 50.0
    reward_center = torch.tensor([0.5, 1.5], dtype=torch.float32)
    sigma_r = 0.5
    num_ode_steps = 200
    
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
    
    results = {}
    
    # 1. Best-of-n
    print(f"--- Running Best-of-N (B={B}, n={n}) ---")
    torch.manual_seed(42)
    t0 = time.time()
    bon_samples_all, bon_rewards_all = sample_guided(
        velocity_net, B * n, lam, reward_center, sigma_r,
        num_ode_steps=num_ode_steps, method="plugin", sigma_schedule="memoryless",
        rescale=rescale, device=device
    )
    bon_samples, bon_rewards = best_of_n(torch.tensor(bon_samples_all), torch.tensor(bon_rewards_all), n)
    bon_samples, bon_rewards = bon_samples.numpy(), bon_rewards.numpy()
    results["bon"] = {"samples": bon_samples, "rewards": bon_rewards, "time": time.time() - t0}
    
    # SMC runs
    methods = ["naive", "plugin", "second_order"]
    for m in methods:
        print(f"--- Running SMC ({m}) (B={B}, n={n}) ---")
        torch.manual_seed(42)
        t0 = time.time()
        s, r, ess, resamples = sample_smc(
            velocity_net, B, n, lam, reward_center, sigma_r,
            num_ode_steps=num_ode_steps, method=m, rescale=rescale, device=device, t_stop=0.9
        )
        results[m] = {"samples": s, "rewards": r, "ess": ess, "resamples": resamples, "time": time.time() - t0}
        print(f"Resample count: {resamples}")
        
    # Print metrics
    print("-" * 80)
    print(f"{'Method':<20} | {'W2':<8} | {'Resamples':<10} | {'Time (s)':<8}")
    print("-" * 80)
    
    labels = {
        "bon": "Best-of-N",
        "naive": "Naive SMC",
        "plugin": "First-order SMC",
        "second_order": "Second-order SMC"
    }
    
    for k in ["bon", "naive", "plugin", "second_order"]:
        _, _, _, _, w2 = compute_metrics(results[k]["samples"], results[k]["rewards"], lam, reward_center.numpy(), is_unguided=False)
        res = results[k].get("resamples", "N/A")
        t = results[k]["time"]
        print(f"{labels[k]:<20} | {w2:<8.4f} | {str(res):<10} | {t:<8.1f}")
        results[k]["w2"] = w2
        
    print("-" * 80)
    
    # Plot ESS
    fig, ax = plt.subplots(figsize=(8, 6))
    t_vals = np.linspace(0, 1.0, num_ode_steps)
    ax.plot(t_vals, results["naive"]["ess"], label="Naive SMC", color='gray')
    ax.plot(t_vals, results["plugin"]["ess"], label="First-order SMC", color='orange')
    ax.plot(t_vals, results["second_order"]["ess"], label="Second-order SMC", color='blue')
    
    ax.axhline(y=n/2, color='red', linestyle='--', label='Resampling threshold (n/2)')
    ax.axvline(x=0.9, color='black', linestyle=':', label='Resampling stops (t=0.9)')
    
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Average Effective Sample Size (ESS)")
    ax.set_title(f"ESS over time for $\lambda={lam}$ (B={B}, n={n})")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/smc_ess.png", dpi=200)
    print("Saved results/smc_ess.png")
    
    # 3. Run without resampling (t_stop=0) to measure twist quality
    print("--- Running SMC without resampling (t_stop = 0) ---")
    ess_no_resample = {}
    for m in methods:
        torch.manual_seed(42)
        _, _, ess, _ = sample_smc(
            velocity_net, B, n, lam, reward_center, sigma_r,
            num_ode_steps=num_ode_steps, method=m, rescale=rescale, device=device, t_stop=0.0
        )
        ess_no_resample[m] = ess
        
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(t_vals, ess_no_resample["naive"], label="Naive SMC", color='gray')
    ax2.plot(t_vals, ess_no_resample["plugin"], label="First-order SMC", color='orange')
    ax2.plot(t_vals, ess_no_resample["second_order"], label="Second-order SMC", color='blue')
    
    ax2.set_xlabel("Time $t$")
    ax2.set_ylabel("Average Effective Sample Size (ESS)")
    ax2.set_title(f"Pure Twist Quality (No Resampling) for $\lambda={lam}$ (B={B}, n={n})")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/smc_ess_no_resample.png", dpi=200)
    print("Saved results/smc_ess_no_resample.png")
    
    # Plot scatter
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    methods_list = ["bon", "naive", "plugin", "second_order"]
    
    for i, m in enumerate(methods_list):
        ax = axes[i]
        _checker_background(ax)
        samples = results[m]["samples"]
        ax.scatter(samples[:, 0], samples[:, 1], c="#f6c945", s=12, alpha=0.92, edgecolors="black", linewidths=0.25, zorder=4)
        ax.scatter(reward_center[0].item(), reward_center[1].item(), c="#e60000", s=340, marker="X", edgecolors="black", linewidths=0.9, zorder=6)
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.set_aspect("equal")
        ax.set_title(f"{labels[m]}\nW2: {results[m]['w2']:.2f}", pad=10)
        
    plt.tight_layout()
    plt.savefig("results/smc_scatter.png", dpi=200)
    print("Saved results/smc_scatter.png")

if __name__ == "__main__":
    main()
