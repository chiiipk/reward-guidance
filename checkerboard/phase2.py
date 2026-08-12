import torch
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import VelocityMLP, sample_checkerboard
import ot
import math

torch.set_num_threads(1)
device = "cpu"
rescale = 1.73

reward_center = torch.tensor([0.5, 0.5], dtype=torch.float32)
s = 1.5

def bump_reward(x, c, s):
    diff = x - c
    return torch.exp(-0.5 * (diff**2).sum(-1) / s**2)

def compute_plugin_guidance(x, t, velocity_net, rescale, lam, c, s):
    mu = x + velocity_net(t, x) * (1 - t.unsqueeze(-1))
    diff = mu - c
    val = bump_reward(mu, c, s)
    grad = -(diff / s**2) * val.unsqueeze(-1)
    return lam * grad

@torch.no_grad()
def sample_bump(velocity_net, num_samples, method, lam, num_ode_steps=200):
    x = torch.randn(num_samples, 2, device=device)
    dt = 1.0 / num_ode_steps
    for step in range(num_ode_steps):
        t = torch.ones((num_samples,)) * (step * dt)
        v = velocity_net(t, x)
        if step < num_ode_steps - 1:
            if method == "plugin":
                g = compute_plugin_guidance(x, t, velocity_net, rescale, lam, reward_center, s)
            elif method == "plugin_damping":
                g = compute_plugin_guidance(x, t, velocity_net, rescale, lam, reward_center, s)
                g = g * (1 - t.unsqueeze(-1))
            else:
                g = 0
            
            beta_t = 2 * (1 - (step * dt)) * rescale**2
            x = x + (v + 0.5 * beta_t * g) * dt + math.sqrt(beta_t * dt) * torch.randn_like(x)
        else:
            x = x + v * dt
    return x

def generate_target_samples(velocity_net, lam, num_samples=2000):
    x_base = sample_checkerboard(num_samples * 50)
    rewards = bump_reward(x_base, reward_center, s)
    lam_r = lam * rewards
    weights = torch.exp(lam_r)
    weights = weights / weights.sum()
    indices = torch.multinomial(weights, num_samples, replacement=True)
    return x_base[indices]

def analyze_samples(samples, target_samples):
    if torch.isnan(samples).any() or torch.isinf(samples).any() or samples.abs().max() > 1e4:
        return float('nan'), float('nan'), float('nan'), 0.0
    
    cov = torch.cov(samples.T)
    L, _ = torch.linalg.eigh(cov)
    ratio = L[1].item() / L[0].item() if L[0].item() > 0 else 0
    
    r_vals = bump_reward(samples, reward_center, s)
    mean_reward = r_vals.mean().item()
    
    M = ot.dist(samples.detach().numpy(), target_samples.detach().numpy())
    a, b = np.ones((len(samples),)) / len(samples), np.ones((len(target_samples),)) / len(target_samples)
    w2 = ot.emd2(a, b, M, numItermax=500000)
    
    in_bounds = ((samples[:, 0] > -4) & (samples[:, 0] < 4) & (samples[:, 1] > -4) & (samples[:, 1] < 4))
    in_dist = in_bounds.float().mean().item()
    
    return ratio, mean_reward, w2, in_dist

def run_phase2():
    checkpoint = torch.load("results/velocity_net.pt", map_location=device, weights_only=False)
    velocity_net = VelocityMLP(hidden_dim=256, num_layers=4, rescale=rescale).to(device)
    velocity_net.load_state_dict(checkpoint["model_state_dict"])
    velocity_net.eval()
    
    lams = [5, 20, 50, 100, 300]
    methods = ["plugin", "plugin_damping"]
    results = []
    
    for lam in lams:
        print(f"\n--- Running lambda = {lam} ---")
        target_samples = generate_target_samples(velocity_net, lam)
        
        for method in methods:
            t0 = time.time()
            samples = sample_bump(velocity_net, 2000, method, lam)
            t1 = time.time()
            
            ratio, rew, w2, ind = analyze_samples(samples, target_samples)
            results.append({
                "Lambda": lam,
                "Method": method,
                "W2": w2,
                "Reward": rew,
                "In-Dist": ind,
                "Time": t1 - t0
            })
            print(f"{method}: W2={w2:.3f}, Reward={rew:.3f}")
            
    df = pd.DataFrame(results)
    print("\nPhase 2 Results:")
    print(df)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for method in methods:
        method_data = df[df["Method"] == method]
        axes[0].plot(method_data["Lambda"], method_data["W2"], marker='o', label=method)
        axes[1].plot(method_data["Lambda"], method_data["Reward"], marker='o', label=method)
        
    axes[0].set_title("W2 Distance vs Lambda")
    axes[0].set_xlabel("Lambda")
    axes[0].set_ylabel("W2")
    axes[0].legend()
    
    axes[1].set_title("Mean Reward vs Lambda")
    axes[1].set_xlabel("Lambda")
    axes[1].set_ylabel("Reward")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("phase2_sweep.png")
    print("Saved phase2_sweep.png")

if __name__ == "__main__":
    run_phase2()
