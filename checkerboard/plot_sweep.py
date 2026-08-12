import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_comparison import compute_metrics

def sweep():
    lams = [10.0, 20.0, 30.0, 40.0, 50.0]
    reward_center_out = [0.5, 1.5]
    reward_center_in = [0.5, 0.5]
    
    # We will plot Reward, In-Dist, and W2 for Damped and Second-Order
    # First, collect data for out-of-support
    out_damp_w2 = []
    out_second_w2 = []
    out_plugin_w2 = []
    
    in_damp_w2 = []
    in_second_w2 = []
    in_plugin_w2 = []
    
    for lam in lams:
        # Out-of-support
        damped_out = np.load(f'results/guided_damped_lam{lam}_out.npz')
        second_out = np.load(f'results/guided_second_order_lam{lam}_out.npz')
        plugin_out = np.load(f'results/guided_k1_lam{lam}_out.npz')
        
        _, _, _, _, w2_damp_out = compute_metrics(damped_out['samples'], damped_out['rewards'], lam, reward_center_out, is_unguided=False)
        _, _, _, _, w2_second_out = compute_metrics(second_out['samples'], second_out['rewards'], lam, reward_center_out, is_unguided=False)
        _, _, _, _, w2_plugin_out = compute_metrics(plugin_out['samples'], plugin_out['rewards'], lam, reward_center_out, is_unguided=False)
        
        out_damp_w2.append(w2_damp_out)
        out_second_w2.append(w2_second_out)
        out_plugin_w2.append(w2_plugin_out)
        
        # In-support
        damped_in = np.load(f'results/guided_damped_lam{lam}_in.npz')
        second_in = np.load(f'results/guided_second_order_lam{lam}_in.npz')
        plugin_in = np.load(f'results/guided_k1_lam{lam}_in.npz')
        
        _, _, _, _, w2_damp_in = compute_metrics(damped_in['samples'], damped_in['rewards'], lam, reward_center_in, is_unguided=False)
        _, _, _, _, w2_second_in = compute_metrics(second_in['samples'], second_in['rewards'], lam, reward_center_in, is_unguided=False)
        _, _, _, _, w2_plugin_in = compute_metrics(plugin_in['samples'], plugin_in['rewards'], lam, reward_center_in, is_unguided=False)
        
        in_damp_w2.append(w2_damp_in)
        in_second_w2.append(w2_second_in)
        in_plugin_w2.append(w2_plugin_in)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(lams, out_damp_w2, 'x--', label="First-Order + Damp", color="orange")
    axes[0].plot(lams, out_plugin_w2, 's-', label="First-Order", color="red")
    axes[0].plot(lams, out_second_w2, 'o-', label="Second-Order", color="blue")
    axes[0].set_title(f"Mode Collapse (Out of Support)\nReward Center: {reward_center_out}")
    axes[0].set_xlabel("Guidance Scale $\\lambda$")
    axes[0].set_ylabel("W2 Distance to Target $\\downarrow$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(lams, in_damp_w2, 'x--', label="First-Order + Damp", color="orange")
    axes[1].plot(lams, in_plugin_w2, 's-', label="First-Order", color="red")
    axes[1].plot(lams, in_second_w2, 'o-', label="Second-Order", color="blue")
    axes[1].set_title(f"Mode Seeking (In Support)\nReward Center: {reward_center_in}")
    axes[1].set_xlabel("Guidance Scale $\\lambda$")
    axes[1].set_ylabel("W2 Distance to Target $\\downarrow$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/lambda_sweep_w2.png")
    print("Saved sweep figure to results/lambda_sweep_w2.png")

if __name__ == "__main__":
    sweep()
