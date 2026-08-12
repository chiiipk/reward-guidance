import torch
import pandas as pd
from phase1 import generate_target_samples, sample_ridge, analyze_samples, VelocityMLP, rescale, reward_center, A

device = "cpu"
checkpoint = torch.load("results/velocity_net.pt", map_location=device, weights_only=False)
velocity_net = VelocityMLP(hidden_dim=256, num_layers=4, rescale=rescale).to(device)
velocity_net.load_state_dict(checkpoint["model_state_dict"])
velocity_net.eval()

for lam in [1.5, 5.0, 15.0]:
    print(f"\n--- lam = {lam} ---")
    target_samples = generate_target_samples(velocity_net, lam=lam, num_samples=2000)
    target_ratio, target_r, _, target_indist = analyze_samples(target_samples, target_samples, lam=lam)
    print(f"Target: Ratio={target_ratio:.2f}, Reward={target_r:.2f}, In-dist={target_indist:.2f}")
    
    for method in ["plugin", "second_order"]:
        samples = sample_ridge(velocity_net, 2000, method, num_ode_steps=200, lam=lam)
        ratio, r, w2, indist = analyze_samples(samples, target_samples, lam=lam)
        print(f"{method}: W2={w2:.2f}, Ratio={ratio:.2f}, Reward={r:.2f}")
