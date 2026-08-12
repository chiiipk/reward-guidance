import torch
import math
import matplotlib.pyplot as plt
import numpy as np

def compute_errors(sigma_t_sq, beta_t, n_newton):
    D = 2
    mu_t = torch.tensor([[0.5, 0.3]])
    c = torch.tensor([[0.5, 1.5]])
    s = 1.5
    
    def r(x):
        diff = x - c
        return torch.exp(-(diff**2).sum(-1) / (2 * s**2))
        
    def grad_r(x):
        diff = x - c
        return -diff / (s**2) * r(x).unsqueeze(-1)
        
    def H_r(x):
        val = r(x)
        diff = x - c
        I = torch.eye(2).unsqueeze(0).expand(x.shape[0], -1, -1)
        return -1.0/(s**2) * (I * val.unsqueeze(-1).unsqueeze(-1) - torch.bmm(diff.unsqueeze(-1), grad_r(x).unsqueeze(1)))
        
    z = mu_t.clone()
    for _ in range(n_newton):
        g = grad_r(z)
        H = H_r(z)[0]
        res = (z - mu_t)/sigma_t_sq - beta_t*g
        Jac = torch.eye(D)/sigma_t_sq - beta_t*H
        delta = torch.linalg.solve(Jac, res.T).T
        z = z - delta
        
    z = z.detach()
    
    precision_2nd = torch.eye(D)/sigma_t_sq - beta_t * H_r(z)[0]
    Sigma_2nd = torch.linalg.inv(precision_2nd)
    
    if n_newton == 0:
        mu_new = mu_t + beta_t * (Sigma_2nd @ grad_r(mu_t).T).T
    else:
        mu_new = z
    
    grid_size = 400
    lim = 5.0
    x_vals = torch.linspace(-lim, lim, grid_size)
    y_vals = torch.linspace(-lim, lim, grid_size)
    grid_x, grid_y = torch.meshgrid(x_vals, y_vals, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
    cell_area = (2*lim / (grid_size - 1))**2
    
    diff = grid - mu_t
    log_prior = -0.5 * (diff**2).sum(-1) / sigma_t_sq - math.log(2*math.pi*sigma_t_sq)
    log_reward = beta_t * r(grid)
    log_exact = log_prior + log_reward
    
    exact_prob_unnorm = torch.exp(log_exact)
    Z = (exact_prob_unnorm * cell_area).sum()
    exact_prob = exact_prob_unnorm / Z
    
    mean_grid = (exact_prob.unsqueeze(-1) * grid).sum(0) * cell_area
    grid_centered = grid - mean_grid
    outer = torch.bmm(grid_centered.unsqueeze(-1), grid_centered.unsqueeze(1))
    cov_grid = (exact_prob.unsqueeze(-1).unsqueeze(-1) * outer).sum(0) * cell_area
    
    err_mean = torch.abs(mean_grid - mu_new[0]).max().item()
    err_cov = torch.abs(cov_grid - Sigma_2nd).max().item()
    
    return err_mean, err_cov

sigma_vals = np.logspace(np.log10(0.02), np.log10(3.0), 20)
beta_t = 2.0

err_cov_0 = []
err_cov_5 = []

print("Sweeping sigma_t_sq...")
for s_sq in sigma_vals:
    _, ec0 = compute_errors(s_sq, beta_t, 0)
    _, ec5 = compute_errors(s_sq, beta_t, 5)
    err_cov_0.append(ec0)
    err_cov_5.append(ec5)

plt.figure(figsize=(8, 6))
plt.plot(sigma_vals, err_cov_0, marker='o', label='Standard (n_newton=0)')
plt.plot(sigma_vals, err_cov_5, marker='s', label='Newton Mode (n_newton=5)')
plt.xscale('log')
plt.xlabel("sigma_t^2")
plt.ylabel("Max Error in Covariance")
plt.title("Covariance Truncation Error vs sigma_t^2 (beta_t=2.0)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("/Users/skye/.gemini/antigravity-ide/brain/e389e5c1-4a55-4bf4-be21-d05d6b6e7615/error_vs_sigma.png", dpi=150)
print("Saved error_vs_sigma.png")
