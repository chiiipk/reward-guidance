import torch
x = torch.tensor([1.0])
dt = 1/200
for step in range(200):
    t = step * dt
    if step < 199:
        # exact drift to 0 is -x / (1-t)
        drift = -x / (1-t)
        x = x + drift * dt
print("Final x without noise:", x.item())

x = torch.tensor([1.0])
for step in range(200):
    t = step * dt
    if step < 199:
        drift = -x / (1-t)
        beta_t = 2 * (1 - t) * 3.0
        x = x + drift * dt + torch.sqrt(torch.tensor(beta_t * dt)) * torch.randn(1)
print("Final x with noise:", x.item())
