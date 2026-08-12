"""Feature-Space Second-Order Reward Guidance — Core Mathematics.

Implements three levels of reward-guided correction for the conditional
distribution p_t(x | x_0) in flow matching:

    Level 0 (First-order):   shift mean by ∇r, keep covariance
    Level 1 (Full 2nd-order): shift mean + reshape cov via full Hessian ∇²r  [O(D³)]
    Level 2 (Feature-space):  shift mean + reshape cov via J_f^T H_g J_f    [O(d³)]

The key insight of Level 2 is the Gauss-Newton decomposition:
    For r(x) = g(f(x)), where f: R^D → R^d (feature extractor, d << D):
        ∇²_x r ≈ J_f^T · ∇²_z g · J_f    (Gauss-Newton approximation)

    Combined with Woodbury identity, this reduces the D×D matrix inversion
    to a d×d inversion, making second-order correction tractable even for
    D ~ 10^5 (images).

Reference:
    Trung Le, "Reward Guided Flow Matching", 2026.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


# ============================================================
# Part 1: Decomposable Reward Model
# ============================================================


class DecomposableReward(nn.Module):
    """A reward model r(x) = g(f(x)) with explicit feature/head split.

    The reward is decomposed into:
        f: R^D → R^d   (feature extractor, possibly a large neural net)
        g: R^d → R      (reward head, typically a small MLP or linear layer)

    This decomposition enables efficient second-order corrections when
    the bottleneck dimension d is much smaller than the input dimension D.

    Example:
        For CLIP-based reward: f = CLIP image encoder (d=768), g = cosine similarity head.
        For ImageReward: f = BLIP encoder, g = MLP reward head.

    Args:
        feature_extractor: Module mapping R^D → R^d
        reward_head: Module mapping R^d → R (output should be (B,) shaped)
        feature_dim: Dimension d of the feature space (inferred if not given)
    """

    def __init__(self, feature_extractor: nn.Module, reward_head: nn.Module,
                 feature_dim: Optional[int] = None):
        super().__init__()
        self.f = feature_extractor
        self.g = reward_head
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute r(x) = g(f(x)). Returns (B,) tensor of rewards."""
        z = self.f(x)           # (B, d)
        return self.g(z).squeeze(-1)  # (B,)

    def compute_feature_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute J_f = ∂f/∂x at x.

        Args:
            x: (B, D) input tensor

        Returns:
            J_f: (B, d, D) Jacobian matrix

        Complexity: O(d × backprop) time, O(D×d) memory.
        This is the most expensive step but only involves d backward passes,
        where d is the (small) feature dimension.
        """
        x = x.detach().requires_grad_(True)
        z = self.f(x)  # (B, d)
        B, d = z.shape
        D = x.shape[1]

        J = torch.zeros(B, d, D, device=x.device, dtype=x.dtype)
        for i in range(d):
            if x.grad is not None:
                x.grad.zero_()
            grad_i = torch.autograd.grad(
                z[:, i].sum(), x,
                retain_graph=(i < d - 1),
                create_graph=False,
            )[0]  # (B, D)
            J[:, i, :] = grad_i
        return J

    def compute_head_hessian(self, z: torch.Tensor) -> torch.Tensor:
        """Compute H_g = ∂²g/∂z² at z.

        Args:
            z: (B, d) feature tensor

        Returns:
            H_g: (B, d, d) Hessian matrix of the reward head

        Complexity: O(d² × backprop_of_g) — very cheap since g is small.
        """
        z = z.detach().requires_grad_(True)
        g_val = self.g(z).squeeze(-1)  # (B,)
        B, d = z.shape

        grad_g = torch.autograd.grad(
            g_val.sum(), z, create_graph=True
        )[0]  # (B, d)

        H = torch.zeros(B, d, d, device=z.device, dtype=z.dtype)
        for i in range(d):
            grad2_i = torch.autograd.grad(
                grad_g[:, i].sum(), z,
                retain_graph=(i < d - 1),
            )[0]  # (B, d)
            H[:, i, :] = grad2_i
        return H

    def compute_reward_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ∇_x r(x). Returns (B, D) tensor."""
        x = x.detach().requires_grad_(True)
        r = self.forward(x)
        return torch.autograd.grad(r.sum(), x)[0].detach()

    def compute_full_hessian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the full Hessian ∇²_x r(x). Shape (B, D, D).

        WARNING: O(D²) memory, O(D² × backprop) time.
        Only use for small D (validation/debugging).
        """
        x = x.detach().requires_grad_(True)
        r = self.forward(x)
        B, D = x.shape

        grad_r = torch.autograd.grad(r.sum(), x, create_graph=True)[0]

        H = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
        for i in range(D):
            grad2_i = torch.autograd.grad(
                grad_r[:, i].sum(), x,
                retain_graph=(i < D - 1),
            )[0]
            H[:, i, :] = grad2_i
        return H.detach()

    def compute_gauss_newton_hessian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the Gauss-Newton approximation J_f^T H_g J_f.

        Shape (B, D, D). For validation against feature_space correction.
        Only use for small D.
        """
        z = self.f(x.detach())
        J_f = self.compute_feature_jacobian(x)    # (B, d, D)
        H_g = self.compute_head_hessian(z)         # (B, d, d)

        # J_f^T H_g J_f = (B, D, d) @ (B, d, d) @ (B, d, D) = (B, D, D)
        return torch.bmm(
            J_f.transpose(1, 2),
            torch.bmm(H_g, J_f)
        )


# ============================================================
# Part 2: Correction Functions
# ============================================================


def first_order_correction(
    mu_t: torch.Tensor,
    sigma_t_sq: float,
    beta_t: float,
    grad_r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute exact first-order (plug-in) correction for target Gaussian.

    Args:
        mu_t:       (B, D) conditional mean
        sigma_t_sq: scalar, conditional variance
        beta_t:     scalar, reward weight at time t
        grad_r:     (B, D) reward gradient ∇r evaluated at mu_t

    Returns:
        new_mean: (B, D)
        new_cov:  (B, D, D) diagonal covariance matrix
    """
    B, D = mu_t.shape
    device = mu_t.device

    new_mean = mu_t + beta_t * sigma_t_sq * grad_r
    new_cov = sigma_t_sq * torch.eye(D, device=device).unsqueeze(0).expand(B, -1, -1)

    return new_mean, new_cov


def first_order_damped_correction(
    mu_t: torch.Tensor,
    sigma_t_sq: float,
    beta_t: float,
    grad_r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scalar damping (Dandapanthula & Boffi). Baseline quyết định.
    
    Args:
        mu_t:       (B, D) conditional mean
        sigma_t_sq: scalar, conditional variance
        beta_t:     scalar, reward weight at time t
        grad_r:     (B, D) reward gradient ∇r evaluated at mu_t
    """
    B, D = mu_t.shape
    device = mu_t.device
    
    beta_eff = beta_t / (1.0 + 2.0 * beta_t * sigma_t_sq)
    new_mean = mu_t + beta_eff * sigma_t_sq * grad_r
    new_cov = sigma_t_sq * torch.eye(D, device=device).unsqueeze(0).expand(B, -1, -1)
    
    return new_mean, new_cov


def full_second_order_correction(
    mu_t: torch.Tensor,
    sigma_t_sq: float,
    beta_t: float,
    grad_r: torch.Tensor,
    hessian_r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full second-order Laplace approximation (with corrected signs).

    Modifies both mean AND covariance using the full Hessian.

        Σ = (σ_t⁻² I - β_t ∇²r)⁻¹
        μ_new = μ_t + β_t Σ ∇r(μ_t)

    WARNING: Requires O(D³) for matrix inversion. Not scalable for large D.

    Args:
        mu_t:       (B, D) conditional mean
        sigma_t_sq: scalar, conditional variance
        beta_t:     scalar, reward weight
        grad_r:     (B, D) reward gradient
        hessian_r:  (B, D, D) full Hessian of reward

    Returns:
        new_mean: (B, D)
        new_cov:  (B, D, D) full covariance matrix
    """
    B, D = mu_t.shape
    device = mu_t.device

    # Precision = σ_t⁻² I - β_t H  (Notice the minus sign)
    I_D = torch.eye(D, device=device).unsqueeze(0).expand(B, -1, -1)
    precision = (1.0 / sigma_t_sq) * I_D - beta_t * hessian_r

    # Σ = precision⁻¹  [O(D³) — the bottleneck]
    new_cov = torch.linalg.inv(precision)

    # μ_new = μ_t + β_t Σ ∇r
    new_mean = mu_t + beta_t * torch.bmm(new_cov, grad_r.unsqueeze(-1)).squeeze(-1)

    return new_mean, new_cov


def feature_space_second_order_correction(
    mu_t: torch.Tensor,
    sigma_t_sq: float,
    beta_t: float,
    grad_r: torch.Tensor,
    J_f: torch.Tensor,
    H_g: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Feature-space second-order correction (OUR CONTRIBUTION).

    Uses Gauss-Newton decomposition: ∇²r ≈ J_f^T H_g J_f
    Combined with Woodbury identity to reduce D×D inversion to d×d.

        Σ = (σ_t⁻² I - β_t J_f^T H_g J_f)⁻¹
          = σ_t² I - σ_t⁴ J_f^T (-H_g⁻¹/β_t + σ_t² J_f J_f^T)⁻¹ J_f

    The matrix being inverted is d×d (feature dim), NOT D×D (input dim).

    Args:
        mu_t:       (B, D) conditional mean
        sigma_t_sq: scalar, conditional variance
        beta_t:     scalar, reward weight
        grad_r:     (B, D) reward gradient ∇_x r at mu_t
        J_f:        (B, d, D) Jacobian of feature extractor at mu_t
        H_g:        (B, d, d) Hessian of reward head in feature space

    Returns:
        new_mean: (B, D)
        new_cov:  (B, D, D) corrected covariance

    Complexity: O(d³ + d²D) instead of O(D³).
        For CLIP (d=768, D=786432): speedup ~ (786432/768)³ ≈ 10⁹×
    """
    B, d, D = J_f.shape
    device = mu_t.device

    # ── Step 1: J_f J_f^T  (B, d, d)
    #    This captures how the feature space "stretches" the input space.
    JJT = torch.bmm(J_f, J_f.transpose(1, 2))

    # ── Step 2: Inner matrix  (B, d, d)
    #    M = -H_g⁻¹/β_t + σ_t² J_f J_f^T
    #    For concave rewards (H_g negative definite), -H_g⁻¹ is positive definite,
    #    making M positive definite and easily invertible.
    H_g_inv = torch.linalg.inv(H_g)
    inner = -H_g_inv / beta_t + sigma_t_sq * JJT
    inner_inv = torch.linalg.inv(inner)

    # ── Step 3: Covariance via Woodbury
    #    Σ = σ_t² I_D - σ_t⁴ J_f^T inner⁻¹ J_f
    sigma_t_4 = sigma_t_sq ** 2
    correction = sigma_t_4 * torch.bmm(
        J_f.transpose(1, 2),
        torch.bmm(inner_inv, J_f),
    )
    I_D = torch.eye(D, device=device).unsqueeze(0).expand(B, -1, -1)
    new_cov = sigma_t_sq * I_D - correction

    # ── Step 4: Corrected mean
    #    μ_new = μ_t + β_t Σ ∇r
    new_mean = mu_t + beta_t * torch.bmm(new_cov, grad_r.unsqueeze(-1)).squeeze(-1)

    return new_mean, new_cov


# ============================================================
# Part 3: Efficient Sampling (without forming D×D matrix)
# ============================================================


def sample_from_correction(
    mean: torch.Tensor,
    sigma_t_sq: float,
    J_f: torch.Tensor,
    inner_inv: torch.Tensor,
    n_samples: int = 1,
) -> torch.Tensor:
    """Sample from N(mean, Σ) WITHOUT forming the D×D covariance matrix.

    Uses the identity:
        Σ = σ_t² I - σ_t⁴ J_f^T inner⁻¹ J_f

    Sampling via:
        x = mean + σ_t ε - σ_t² J_f^T L (J_f ε)

    where inner⁻¹ = L L^T (Cholesky) and ε ~ N(0, I_D).

    This avoids forming the D×D matrix entirely.

    Args:
        mean:       (B, D) corrected mean
        sigma_t_sq: scalar
        J_f:        (B, d, D)
        inner_inv:  (B, d, d) inverse of the inner matrix
        n_samples:  number of samples per batch element

    Returns:
        samples: (B, n_samples, D)

    Complexity: O(B × n_samples × d × D) — linear in D!
    """
    B, D = mean.shape
    d = J_f.shape[1]
    device = mean.device
    sigma_t = sigma_t_sq ** 0.5

    # Cholesky of inner_inv for the correction term
    # inner_inv = L L^T  →  correction factor = σ_t² J_f^T L
    L = torch.linalg.cholesky(inner_inv + 1e-8 * torch.eye(d, device=device))  # (B, d, d)

    samples = []
    for _ in range(n_samples):
        eps = torch.randn(B, D, device=device)  # standard normal noise

        # Uncorrected sample: σ_t ε
        base = sigma_t * eps

        # Correction: σ_t² J_f^T L (L^T J_f ε)
        Jf_eps = torch.bmm(J_f, eps.unsqueeze(-1)).squeeze(-1)          # (B, d)
        LT_Jf_eps = torch.bmm(L.transpose(1, 2), Jf_eps.unsqueeze(-1)).squeeze(-1)  # (B, d)
        corr = sigma_t_sq * torch.bmm(
            J_f.transpose(1, 2), LT_Jf_eps.unsqueeze(-1)
        ).squeeze(-1)  # (B, D)

        sample = mean + base - corr
        samples.append(sample)

    return torch.stack(samples, dim=1)  # (B, n_samples, D)


# ============================================================
# Part 4: Helper — Exact tilted density (for validation on toy problems)
# ============================================================


def exact_tilted_log_density(
    x_grid: torch.Tensor,
    mu_t: torch.Tensor,
    sigma_t_sq: float,
    beta_t: float,
    reward_fn,
) -> torch.Tensor:
    """Compute log p_t(x) ∝ N(x|μ_t, σ_t²I) · exp(β_t r(x)) on a grid.

    For 2D validation only. Computes the unnormalized log-density.

    Args:
        x_grid:     (G, 2) grid points
        mu_t:       (1, 2) or (2,) conditional mean
        sigma_t_sq: scalar
        beta_t:     scalar
        reward_fn:  callable, x → r(x)

    Returns:
        log_density: (G,) unnormalized log p
    """
    if mu_t.dim() == 1:
        mu_t = mu_t.unsqueeze(0)

    diff = x_grid - mu_t  # (G, 2)
    log_gauss = -0.5 * torch.sum(diff ** 2, dim=-1) / sigma_t_sq
    log_reward = beta_t * reward_fn(x_grid)
    return log_gauss + log_reward
