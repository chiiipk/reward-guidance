import torch
import time
from model import VelocityMLP
from sample import sample_smc

device = "cpu"
torch.set_num_threads(1)
rescale = 1.73

def test_timing():
    print("Loading model...")
    checkpoint = torch.load("results/velocity_net.pt", map_location=device, weights_only=False)
    velocity_net = VelocityMLP(hidden_dim=256, num_layers=4, rescale=rescale).to(device)
    velocity_net.load_state_dict(checkpoint["model"])
    velocity_net.eval()
    
    B = 500
    n = 8
    lam = 50.0
    reward_center = torch.tensor([0.5, 1.5], dtype=torch.float32)
    sigma_r = 0.5
    num_ode_steps = 200
    
    print(f"Running second_order + SMC with B={B}, n={n}, steps={num_ode_steps}")
    t0 = time.time()
    sample_smc(
        velocity_net, B, n, lam, reward_center, sigma_r,
        num_ode_steps=num_ode_steps, method="second_order", rescale=rescale, device=device, t_stop=0.9
    )
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.2f} seconds")

if __name__ == "__main__":
    test_timing()
