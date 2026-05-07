"""GMM model: GaussianMixture, LinearInterpolant, and reward functions."""

import numpy as np
import torch
import torch.nn as nn


class GaussianMixture(nn.Module):
    def __init__(self, weights, means, covs):
        super().__init__()
        self.K = len(weights)
        self.dim = len(means[0])
        w = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", w / w.sum())
        self.register_buffer("means", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("covs", torch.tensor(covs, dtype=torch.float32))


class LinearInterpolant(nn.Module):
    """Analytic linear interpolant for a Gaussian mixture.

    The interpolant is X_t = t X_0 + (1-t) Z, Z ~ N(0, I), so the marginal
    p_t(x) is a mixture of Gaussians with means t*mu_k and covariances
    t^2 Sigma_k + (1-t)^2 I.
    """

    def __init__(self, gmm):
        super().__init__()
        self.gmm = gmm
        self.dim = gmm.dim

    def get_posterior_components(self, x_t, t):
        """Compute posterior means, covariances, and weights p(k | X_t = x_t)."""
        B, D = x_t.shape
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x_t.device, dtype=torch.float32)

        t = torch.clamp(t, 1e-4, 1.0 - 1e-4)
        var_noise = (1 - t) ** 2

        post_means, post_covs_chol, post_covs, log_probs = [], [], [], []

        for k in range(self.gmm.K):
            mu_0 = self.gmm.means[k]
            Sigma_0 = self.gmm.covs[k]

            mu_t = t * mu_0
            Sigma_t = (t**2) * Sigma_0 + var_noise * torch.eye(D, device=x_t.device)

            diff = x_t - mu_t
            L = torch.linalg.cholesky(Sigma_t)
            log_det = 2 * torch.sum(torch.log(torch.diagonal(L)))
            y = torch.linalg.solve_triangular(L, diff.T, upper=False)
            quad = torch.sum(y**2, dim=0)
            const = D * torch.log(torch.tensor(2 * np.pi, device=x_t.device))
            log_pdf = -0.5 * (const + log_det + quad)
            log_probs.append(log_pdf + torch.log(self.gmm.weights[k]))

            z = torch.linalg.solve_triangular(L.T, y, upper=True)
            gain_term = t * Sigma_0 @ z
            cond_mu = mu_0.unsqueeze(1) + gain_term
            cond_mu = cond_mu.T

            A = t * Sigma_0
            M = torch.linalg.solve_triangular(L, A.T, upper=False)
            cond_Sigma = Sigma_0 - M.T @ M

            post_means.append(cond_mu)
            post_covs.append(cond_Sigma)

            # Jitter scales with Sigma_0's magnitude so the Cholesky stays
            # well-defined for wide-covariance configs (e.g. variance ~16)
            # where Sigma_0 - M^T M cancels to numerical zero near t=1.
            jitter_scale = max(1.0, float(Sigma_0.diagonal().abs().max()))
            jitter = 1e-5 * jitter_scale * torch.eye(D, device=x_t.device)
            post_covs_chol.append(torch.linalg.cholesky(cond_Sigma + jitter))

        log_probs = torch.stack(log_probs, dim=1)
        lse = torch.logsumexp(log_probs, dim=1, keepdim=True)
        post_weights = torch.exp(log_probs - lse)

        return torch.stack(post_means, dim=1), post_covs, post_covs_chol, post_weights

    def denoise(self, x, t):
        means, _, _, weights = self.get_posterior_components(x, t)
        return torch.sum(weights.unsqueeze(-1) * means, dim=1)

    def velocity(self, x, t):
        cond_mean = self.denoise(x, t)
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device)
        return (cond_mean - x) / (1.0 - t + 1e-6)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def reward_quadratic(x, target):
    """Quadratic reward r(x) = -||x - target||^2."""
    return -torch.sum((x - target) ** 2, dim=-1)


def reward_double_well(x):
    """Double-well reward with local maxima at (-2.5, -1.0) and (2.5, 1.0).

    r(x) = -0.125(x1^4 - 12.5 x1^2) - (x2 - 0.4 x1)^2
    """
    x1 = x[..., 0]
    x2 = x[..., 1]
    return -0.125 * (x1**4 - 12.5 * x1**2) - (x2 - 0.4 * x1) ** 2


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

# Shared defaults
_ISO_COV = [[0.5, 0.0], [0.0, 0.5]]
_EQUAL_WEIGHTS = [0.5, 0.5]
_DEFAULT_MEANS = [[-2.5, 0.0], [2.5, 0.0]]
_DEFAULT_TARGET = torch.tensor([0.0, 2.5])
_DEFAULT_SIGMA_HYPER = float(np.sqrt(0.5))

# Default: symmetric GMM at (±2.5, 0), isotropic covariance, quadratic reward
QUADRATIC_WEIGHTS = _EQUAL_WEIGHTS
QUADRATIC_MEANS = _DEFAULT_MEANS
QUADRATIC_COVS = [_ISO_COV, _ISO_COV]
QUADRATIC_TARGET = _DEFAULT_TARGET
QUADRATIC_SIGMA_HYPER = _DEFAULT_SIGMA_HYPER

# Double-well reward (from gmm-experiment-nonquadratic.py); same GMM as default
DOUBLE_WELL_WEIGHTS = _EQUAL_WEIGHTS
DOUBLE_WELL_MEANS = _DEFAULT_MEANS
DOUBLE_WELL_COVS = [_ISO_COV, _ISO_COV]
DOUBLE_WELL_SIGMA_HYPER = 50.0

# Non-isotropic covariances: off-diagonal ±0.25 so both components tilt toward the target
# Left component at (-2.5, 0) tilts northeast; right component at (2.5, 0) tilts northwest
NONISO_WEIGHTS = _EQUAL_WEIGHTS
NONISO_MEANS = _DEFAULT_MEANS
NONISO_COVS = [[[0.5, 0.25], [0.25, 0.5]], [[0.5, -0.25], [-0.25, 0.5]]]
NONISO_TARGET = _DEFAULT_TARGET
NONISO_SIGMA_HYPER = _DEFAULT_SIGMA_HYPER

# Unequal component weights
UNEQUAL_WEIGHTS = [0.2, 0.8]
UNEQUAL_MEANS = _DEFAULT_MEANS
UNEQUAL_COVS = [_ISO_COV, _ISO_COV]
UNEQUAL_TARGET = _DEFAULT_TARGET
UNEQUAL_SIGMA_HYPER = _DEFAULT_SIGMA_HYPER

# Uncentered: left component shifted further left (from gmm-experiment-glass.py)
UNCENTERED_WEIGHTS = _EQUAL_WEIGHTS
UNCENTERED_MEANS = [[-4.0, 0.0], [1.0, 0.0]]
UNCENTERED_COVS = [_ISO_COV, _ISO_COV]
UNCENTERED_TARGET = _DEFAULT_TARGET
UNCENTERED_SIGMA_HYPER = _DEFAULT_SIGMA_HYPER

# Single Gaussian: one component at origin, same covariance and target as quadratic
GAUSSIAN_WEIGHTS = [1.0]
GAUSSIAN_MEANS = [[0.0, 0.0]]
GAUSSIAN_COVS = [_ISO_COV]
GAUSSIAN_TARGET = _DEFAULT_TARGET
GAUSSIAN_SIGMA_HYPER = _DEFAULT_SIGMA_HYPER
