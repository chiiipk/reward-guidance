import torch
import math

def test_p2(reward_type="gaussian_bump", use_newton=False):
    D = 2
    mu_t = torch.tensor([[0.5, 0.3]])
    sigma_t_sq = 0.8
    beta_t = 2.0
    
    A = torch.tensor([[10.0, 0.0], [0.0, 0.1]])
    c = torch.tensor([[0.5, 1.5]])
    s = 1.5
    
    def r(x):
        if reward_type == "quadratic":
            return -0.5 * (x @ A * x).sum(-1)
        else: # gaussian bump
            diff = x - c
            return torch.exp(-(diff**2).sum(-1) / (2 * s**2))
            
    def grad_r(x):
        if reward_type == "quadratic":
            return -x @ A
        else:
            diff = x - c
            return -diff / (s**2) * r(x).unsqueeze(-1)
            
    def H_r(x):
        if reward_type == "quadratic":
            return -A.unsqueeze(0).expand(x.shape[0], -1, -1)
        else:
            val = r(x)
            diff = x - c
            I = torch.eye(2).unsqueeze(0).expand(x.shape[0], -1, -1)
            return -1.0/(s**2) * (I * val.unsqueeze(-1).unsqueeze(-1) - torch.bmm(diff.unsqueeze(-1), grad_r(x).unsqueeze(1)))
            
    # Find expansion point
    x_star = mu_t.clone().requires_grad_(True)
    if use_newton:
        optimizer = torch.optim.LBFGS([x_star], max_iter=100)
        def closure():
            optimizer.zero_grad()
            # Maximize L(x) => Minimize -L(x)
            loss = 0.5 * ((x_star - mu_t)**2).sum(-1) / sigma_t_sq - beta_t * r(x_star)
            loss.backward()
            return loss
        optimizer.step(closure)
        
    x_star = x_star.detach()
    
    if use_newton:
        precision_2nd = torch.eye(D)/sigma_t_sq - beta_t * H_r(x_star)[0]
        Sigma_2nd = torch.linalg.inv(precision_2nd)
        grad_L_x_star = -(x_star - mu_t) / sigma_t_sq + beta_t * grad_r(x_star)
        mu_2nd = x_star + (Sigma_2nd @ grad_L_x_star.T).T
    else:
        # Standard Taylor at mu_t
        grad_at_mu = grad_r(mu_t)
        H_at_mu = H_r(mu_t)[0]
        
        precision_2nd = torch.eye(D)/sigma_t_sq - beta_t * H_at_mu
        Sigma_2nd = torch.linalg.inv(precision_2nd)
        mu_2nd = mu_t + beta_t * (Sigma_2nd @ grad_at_mu.T).T
    
    # Grid integration
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
    
    # Normalize to get exact probability
    exact_prob_unnorm = torch.exp(log_exact)
    Z = (exact_prob_unnorm * cell_area).sum()
    exact_prob = exact_prob_unnorm / Z
    
    # Compute mean and cov of exact grid distribution
    mean_grid = (exact_prob.unsqueeze(-1) * grid).sum(0) * cell_area
    
    grid_centered = grid - mean_grid
    outer = torch.bmm(grid_centered.unsqueeze(-1), grid_centered.unsqueeze(1))
    cov_grid = (exact_prob.unsqueeze(-1).unsqueeze(-1) * outer).sum(0) * cell_area
    
    print(f"--- Reward: {reward_type}, LBFGS: {use_newton} ---")
    print(f"Error in mean: {torch.abs(mean_grid - mu_2nd[0]).max().item():.2e}")
    print(f"Error in cov:  {torch.abs(cov_grid - Sigma_2nd).max().item():.2e}")

test_p2("gaussian_bump", False)
test_p2("gaussian_bump", True)
