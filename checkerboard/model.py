"""MLP velocity network and checkerboard distribution sampling.

The VelocityMLP architecture (input scaling, GELU activations, hidden width)
and the adaptive Gaussian rescaling convention follow Nicholas Boffi's jax-interpolants
reference implementation (https://github.com/nmboffi/jax-interpolants).
"""

import torch
import torch.nn as nn
import numpy as np


class VelocityMLP(nn.Module):
    """MLP that predicts the velocity field b(t, x) for flow matching.

    Follows nmboffi/jax-interpolants: input is [x/rescale, t], GELU activations.
    """

    def __init__(self, hidden_dim=256, num_layers=4, rescale=1.0):
        super().__init__()
        self.rescale = rescale
        # Input: (x1/rescale, x2/rescale, t) -> 3 dimensions
        layers = [nn.Linear(3, hidden_dim), nn.GELU()]
        for _ in range(num_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, t, x):
        """
        Args:
            t: (B,) or scalar
            x: (B, 2)
        Returns:
            velocity: (B, 2)
        """
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        tx = torch.cat([x / self.rescale, t.unsqueeze(-1)], dim=-1)
        return self.net(tx)


def denoiser(velocity_net, t, x):
    """Compute D_t(x) = x + (1 - t) * b_theta(t, x)."""
    if t.dim() == 0:
        t = t.unsqueeze(0)
    b = velocity_net(t, x)
    return x + (1.0 - t.unsqueeze(-1)) * b


def sample_checkerboard(n, device="cpu"):
    """Sample n points from the 6x6 checkerboard on [-3, 3]^2.

    Filled squares are those where floor(x + 3) + floor(y + 3) is even.
    Each filled square has side length 1. Direct sampling (no rejection).
    """
    # Enumerate the 18 filled squares
    squares = []
    for i in range(6):
        for j in range(6):
            if (i + j) % 2 == 0:
                squares.append((i - 3, j - 3))  # lower-left corner
    squares = np.array(squares, dtype=np.float64)  # (18, 2)

    # Pick a random square for each sample, then uniform within it
    idx = np.random.randint(0, len(squares), size=n)
    offsets = np.random.uniform(0, 1, size=(n, 2))
    samples = squares[idx] + offsets
    return torch.tensor(samples, dtype=torch.float32, device=device)


def checkerboard_rescale():
    """Compute the standard deviation of the checkerboard distribution.

    Used to set the adaptive Gaussian base distribution, following
    nmboffi/jax-interpolants.
    """
    samples = sample_checkerboard(200_000)
    return float(samples.std())


def checkerboard_density(x):
    """Evaluate the (unnormalized) checkerboard density.

    Args:
        x: (..., 2) array or tensor
    Returns:
        density: (...) with 1.0 on filled squares, 0.0 elsewhere
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    i = np.floor(x[..., 0] + 3).astype(int)
    j = np.floor(x[..., 1] + 3).astype(int)
    in_bounds = (i >= 0) & (i < 6) & (j >= 0) & (j < 6)
    filled = (i + j) % 2 == 0
    return (in_bounds & filled).astype(np.float64)


def reward_fn(x, center, sigma_r=0.5):
    """Gaussian bump reward: r(x) = exp(-||x - center||^2 / (2 * sigma_r^2)).

    Args:
        x: (B, 2) tensor
        center: (2,) tensor
        sigma_r: width of the bump
    Returns:
        reward: (B,) tensor
    """
    return torch.exp(-torch.sum((x - center) ** 2, dim=-1) / (2 * sigma_r ** 2))


# Center of a filled square near the middle.
# The square [0, 1] x [0, 1] has i=3, j=3, sum=6 (even) -> filled.
DEFAULT_REWARD_CENTER = torch.tensor([0.5, 0.5])
