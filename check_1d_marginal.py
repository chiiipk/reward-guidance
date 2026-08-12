import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def run_1d_marginal_test():
    # We want to verify the marginals p_t(x) = \int p_t(x|x_0) p_data(x_0) dx_0
    # where p_t(x|x_0) = N(x | mu_t x_0, sigma_t^2) * exp(beta_t r(x)) / Z_t(x_0)
    
    # Setup
    x = np.linspace(-5, 5, 500)
    x_0_vals = np.linspace(-5, 5, 500)
    dx = x[1] - x[0]
    
    # 1. p_data(x_0) is N(-2, 0.5^2)
    p_data = stats.norm.pdf(x_0_vals, loc=-2, scale=0.5)
    
    # 2. Linear reward r(x) = x, beta_t = t
    # OT-CFM schedule: mu_t = 1-t, sigma_t = t
    # Or variance preserving: mu_t = cos(pi t / 2), sigma_t = sin(pi t / 2)
    
    t_vals = [0.0, 0.5, 1.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, t in enumerate(t_vals):
        mu_t = 1 - t
        sigma_t_sq = t**2
        beta_t = 2.0 * t  # max beta is 2.0
        
        # If t=0, it's a delta function. We handle it carefully.
        if t == 0:
            marginal_t = p_data
        else:
            # p_t(x|x_0)
            # = N(x | mu_t x_0 + beta_t sigma_t^2, sigma_t^2)
            # because N(x | m, s^2) exp(b x) \propto N(x | m + b s^2, s^2)
            marginal_t = np.zeros_like(x)
            for j, x_0 in enumerate(x_0_vals):
                mean_shift = mu_t * x_0 + beta_t * sigma_t_sq
                cond_pdf = stats.norm.pdf(x, loc=mean_shift, scale=np.sqrt(sigma_t_sq))
                marginal_t += cond_pdf * p_data[j] * dx
                
        # Plot
        axes[i].plot(x, marginal_t, label=f'Marginal p_{t}(x)', color='blue')
        axes[i].plot(x, p_data, '--', label='p_data(x)', color='black', alpha=0.5)
        
        # Target prior at t=1 is N(0, 1) * exp(2.0 * x) \propto N(2.0, 1)
        if t == 1.0:
            target_prior = stats.norm.pdf(x, loc=2.0, scale=1.0)
            axes[i].plot(x, target_prior, 'r:', label='Target Prior', lw=2)
            
        axes[i].set_title(f"t = {t}")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig('results/marginal_1d.png')
    print("Saved 1D marginals to results/marginal_1d.png")

if __name__ == "__main__":
    run_1d_marginal_test()
