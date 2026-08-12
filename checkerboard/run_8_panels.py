import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import time

from model import VelocityMLP, checkerboard_density, DEFAULT_REWARD_CENTER
from sample import sample_guided, sample_smc, best_of_n
from plot_comparison import compute_metrics, generate_analytic_samples
from plot_checkerboards import _draw_panel

def run_experiment():
    device = "mps"
    num_samples = 1500 # Slightly reduced for faster SMC
    lam = 10.0
    reward_center = DEFAULT_REWARD_CENTER
    sigma_r = 1.5
    rescale = 1.73
    
    ck = torch.load('results/velocity_net.pt', map_location=device, weights_only=False)
    net = VelocityMLP(hidden_dim=256, num_layers=4, rescale=rescale).to(device)
    net.load_state_dict(ck['model'])
    net.eval()
    
    results = {}
    
    print("A) Target...")
    s, r = generate_analytic_samples(lam, reward_center.numpy(), num_samples=num_samples)
    results["Target"] = (s, r, "Target (Analytic)", "A")
    
    print("B) Plugin k=8...")
    s, r = sample_guided(net, num_samples, lam, reward_center, sigma_r, k=8, rescale=rescale, device=device, method="plugin", sigma_schedule="memoryless")
    results["Plugin_k8"] = (s, r, "Plug-in (k=8)", "B")
    
    print("C) Second-Order k=1...")
    s, r = sample_guided(net, num_samples, lam, reward_center, sigma_r, k=1, rescale=rescale, device=device, method="second_order", sigma_schedule="memoryless")
    results["Second_k1"] = (s, r, "Second-Order (k=1)", "C")
    
    print("D) Second-Order + Bo4...")
    s_all, r_all = sample_guided(net, num_samples * 4, lam, reward_center, sigma_r, k=1, rescale=rescale, device=device, method="second_order", sigma_schedule="memoryless")
    s, r = best_of_n(torch.tensor(s_all), torch.tensor(r_all), 4)
    results["Second_Bo4"] = (s.numpy(), r.numpy(), "Second-Order + Bo4", "D")
    
    print("E) Plugin + Damp + Bo4 (Seed 1)...")
    s_all, r_all = sample_guided(net, num_samples * 4, lam, reward_center, sigma_r, k=1, rescale=rescale, device=device, method="plugin", sigma_damp=0.2, sigma_schedule="memoryless")
    s, r = best_of_n(torch.tensor(s_all), torch.tensor(r_all), 4)
    results["Plugin_Damp_Bo4_1"] = (s.numpy(), r.numpy(), "Plug-in + Damp + Bo4 (S1)", "E")

    print("F) Plugin + Damp + Bo4 (Seed 2)...")
    s_all, r_all = sample_guided(net, num_samples * 4, lam, reward_center, sigma_r, k=1, rescale=rescale, device=device, method="plugin", sigma_damp=0.2, sigma_schedule="memoryless")
    s, r = best_of_n(torch.tensor(s_all), torch.tensor(r_all), 4)
    results["Plugin_Damp_Bo4_2"] = (s.numpy(), r.numpy(), "Plug-in + Damp + Bo4 (S2)", "F")
    
    print("G) Second-Order + Bo4 (Seed 2)...")
    s_all, r_all = sample_guided(net, num_samples * 4, lam, reward_center, sigma_r, k=1, rescale=rescale, device=device, method="second_order", sigma_schedule="memoryless")
    s, r = best_of_n(torch.tensor(s_all), torch.tensor(r_all), 4)
    results["Second_Bo4_2"] = (s.numpy(), r.numpy(), "Second-Order + Bo4 (S2)", "G")
    
    print("H) Second-Order + Damp + Bo4...")
    s_all, r_all = sample_guided(net, num_samples * 4, lam, reward_center, sigma_r, k=1, rescale=rescale, device=device, method="second_order", sigma_damp=0.2, sigma_schedule="memoryless")
    s, r = best_of_n(torch.tensor(s_all), torch.tensor(r_all), 4)
    results["Second_Damp_Bo4"] = (s.numpy(), r.numpy(), "Second-Order + Damp + Bo4", "H")
    
    # Plotting
    keys = ["Target", "Plugin_k8", "Second_k1", "Second_Bo4", "Second_Bo4_2", "Plugin_Damp_Bo4_1", "Plugin_Damp_Bo4_2", "Second_Damp_Bo4"]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True, sharex=True)
    c = reward_center.cpu().numpy()
    
    # Save results to disk
    np.save("results_8_panels_data.npy", results)
    
    print("-" * 80)
    print(f"{'Method':<30} | {'W2':<8} | {'In-Dist':<8} | {'Reward':<8}")
    print("-" * 80)
    
    for i, key in enumerate(keys):
        ax = axes[i // 4, i % 4]
        s, r, title, label = results[key]
        _draw_panel(ax, s, title, c, None)
        
        # Compute metrics
        mean_r, in_dist, cov, ent, w2 = compute_metrics(s, r, lam, c, is_unguided=False)
        print(f"{title:<30} | {w2:<8.3f} | {in_dist*100:5.1f}% | {mean_r:<8.3f}")
        
    plt.tight_layout()
    plt.savefig("results_8_panels.png", dpi=200)
    print("Saved results_8_panels.png")

if __name__ == "__main__":
    run_experiment()
