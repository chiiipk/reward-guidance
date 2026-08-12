import torch
from phase1 import compute_plugin_guidance_ridge, compute_second_order_guidance_ridge, VelocityMLP, rescale, reward_center, A

velocity_net = VelocityMLP(hidden_dim=256, num_layers=4, rescale=rescale)
x = torch.tensor([[1.0, 1.0]])
t = torch.tensor([0.5])

for lam in [0.5, 1.5, 5.0, 15.0]:
    g1 = compute_plugin_guidance_ridge(x, t, velocity_net, rescale, lam, reward_center, A)
    g2 = compute_second_order_guidance_ridge(x, t, velocity_net, rescale, lam, reward_center, A)
    print(f"lam={lam}: ||g1||={g1.norm().item():.4f}, ||g2||={g2.norm().item():.4f}")
