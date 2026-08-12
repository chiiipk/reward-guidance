import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from phase1 import VelocityMLP, device, rescale, sample_ridge, generate_target_samples, A, reward_center
from plot_checkerboards import _draw_panel

def run_and_plot():
    # Load model
    ck = torch.load('results/velocity_net.pt', map_location=device, weights_only=False)
    net = VelocityMLP(hidden_dim=256, num_layers=4, rescale=rescale).to(device)
    net.load_state_dict(ck['model'])
    net.eval()
    
    lam = 5.0
    print("Generating Target...")
    target_s = generate_target_samples(lam, 2000, seed=0).numpy()
    print("Generating Unguided...")
    unguided_s = sample_ridge(net, 2000, "unguided", 200, lam, seed=0).numpy()
    print("Generating Plugin...")
    plugin_s = sample_ridge(net, 2000, "plugin", 200, lam, seed=0).numpy()
    print("Generating Damping...")
    damping_s = sample_ridge(net, 2000, "plugin_damping", 200, lam, seed=0).numpy()
    print("Generating Second-Order...")
    second_s = sample_ridge(net, 2000, "second_order", 200, lam, seed=0).numpy()
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    c = reward_center.cpu().numpy()
    
    _draw_panel(axes[0], target_s, "Target (Analytic)", c, "A")
    _draw_panel(axes[1], unguided_s, "Unguided", c, "B")
    _draw_panel(axes[2], damping_s, "First-Order + Damp", c, "C")
    _draw_panel(axes[3], plugin_s, "First-Order", c, "D")
    _draw_panel(axes[4], second_s, "Second-Order", c, "E")
    
    fig.tight_layout()
    plt.savefig("quad_sweep_lam5.png", dpi=200)
    print("Saved quad_sweep_lam5.png")

if __name__ == "__main__":
    run_and_plot()
