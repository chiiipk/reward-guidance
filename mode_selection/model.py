"""Analytic model for the symmetric Gaussian mixture, and reward functions.

Base distribution: X_0 ~ N(0, 1). Target: rho_1 = 0.5 N(mu, sigma^2) + 0.5 N(-mu, sigma^2).
Interpolant: I_t = (1 - t) X_0 + t X_1.
"""

import torch
import numpy as np


def _sigma_t_sq(t, sigma):
    """Variance of I_t: sigma_t^2 = (1-t)^2 + t^2 sigma^2."""
    return (1.0 - t) ** 2 + t ** 2 * sigma ** 2


def analytic_denoiser(t, x, mu, sigma):
    """D_t(x) = E[X_1 | I_t = x] for the symmetric Gaussian mixture with N(0,1) base."""
    s2 = _sigma_t_sq(t, sigma)
    return t * sigma ** 2 * x / s2 + mu * (1.0 - t) ** 2 * torch.tanh(t * mu * x / s2) / s2


def analytic_velocity(t, x, mu, sigma):
    """b_t(x) = E[X_1 - X_0 | I_t = x]; see Theorem 5.1 of the paper."""
    s2 = _sigma_t_sq(t, sigma)
    drift = (t * sigma ** 2 - (1.0 - t)) * x / s2
    mixing = mu * (1.0 - t) * torch.tanh(t * mu * x / s2) / s2
    return drift + mixing


def sample_gaussian_mixture(n, mu=3.0, sigma=1.0, device="cpu"):
    """Sample from rho_1(x) = 0.5 N(mu, sigma^2) + 0.5 N(-mu, sigma^2)."""
    components = torch.randint(0, 2, (n,), device=device)
    means = torch.where(components == 0, mu, -mu).float()
    return means + sigma * torch.randn(n, device=device)


def reward_step(x, R=10.0):
    """r(x) = 0 if x >= 0, else -R. Gradient is zero a.e."""
    x_flat = x.squeeze(-1) if x.dim() > 1 else x
    return torch.where(x_flat >= 0, torch.zeros_like(x_flat), -R * torch.ones_like(x_flat))


def reward_gaussian(x, center=3.0, sigma_r=1.0):
    """r(x) = exp(-(x - center)^2 / (2 sigma_r^2))."""
    x_flat = x.squeeze(-1) if x.dim() > 1 else x
    return torch.exp(-(x_flat - center) ** 2 / (2 * sigma_r ** 2))


def reward_linear(x):
    """r(x) = x."""
    x_flat = x.squeeze(-1) if x.dim() > 1 else x
    return x_flat


REWARD_FUNCTIONS = {
    "step": reward_step,
    "gaussian": reward_gaussian,
    "linear": reward_linear,
}
