"""Phase 1 — ridge trên checkerboard. Bản đã sửa.

FIX 1: sigma_t_sq -> phương sai hậu nghiệm thật Var[X_1|X_t]
FIX 2: beta_t -> memoryless schedule (điều kiện để h-transform hợp lệ)
FIX 3: plugin_damping -> công thức damping đúng của Dandapanthula & Boffi
FIX 4: thêm Jacobian dmu/dx qua surrogate (trước đây bị bỏ)
Thêm: unguided baseline, W2 floor, diagnostic giới hạn.

CHƯA CHẠY THỬ (môi trường không có torch). Chạy check_limits() trước run_phase1().
"""

import math
import time

import numpy as np
import ot
import pandas as pd
import torch

from model import VelocityMLP, sample_checkerboard, checkerboard_density

torch.set_num_threads(1)
device = "cpu"
rescale = 1.73

reward_center = torch.tensor([0.5, 1.5], dtype=torch.float32)
A = torch.tensor([[10.0, 0.0], [0.0, 0.1]], dtype=torch.float32)

# convention: t=0 nhiễu, t=1 data


# ---------------------------------------------------------------- schedules

def posterior_var(t, sigma_data=rescale):
    """FIX 1. Var[X_1 | X_t]. Trước đây dùng (1-t)^2*rescale^2 — sai ~2.5x gần t=1.

    Biên: t=0 -> sigma_data^2 (chưa biết gì);  t=1 -> 0 (biết chính xác).
    """
    a = (1.0 - t) ** 2
    return sigma_data ** 2 * a / (a + t ** 2 * sigma_data ** 2 + 1e-12)


def memoryless_var(t, dt):
    """FIX 2. sigma_t^2 = 2(1-t)/t, chặn trên tại t=dt.

    h-transform chỉ hợp lệ khi X_0 ⊥ X_1, và điều đó đòi hỏi đúng schedule này
    (Domingo-Enrich et al.). Bản cũ dùng 2(1-t)*rescale^2 — phá điều kiện đó.
    """
    cap = 2.0 * (1.0 - dt) / dt
    return torch.clamp(2.0 * (1.0 - t) / torch.clamp(t, min=dt), max=cap)


# ---------------------------------------------------------------- reward

def ridge_reward(x, c=reward_center, A=A):
    diff = x - c
    return torch.exp(-0.5 * (diff @ A * diff).sum(-1))


def _grad_H(mu, c=reward_center, A=A):
    """grad r và Gauss-Newton Hessian (bỏ số hạng rank-1 lồi -> luôn âm xác định)."""
    diff = mu - c
    val = ridge_reward(mu, c, A)
    grad = -(diff @ A) * val.unsqueeze(-1)
    B = mu.shape[0]
    H = -A.unsqueeze(0).expand(B, -1, -1) * val.view(-1, 1, 1)
    return grad, H, val


# ---------------------------------------------------------------- guidance

def _guidance_at_mu(mu, t, lam, method):
    """Trả về vector guidance TẠI mu (chưa nhân Jacobian). Shape (B,2)."""
    grad, H, _ = _grad_H(mu)

    if method == "plugin":
        return lam * grad

    if method == "plugin_damping":
        # FIX 3. Công thức đúng: lam_t = lam / (1 + 2*lam*sigma_{1|t}^2).
        # Bản cũ dùng g*(1-t) — không phải damping schedule của paper gốc.
        s2 = posterior_var(t).view(-1, 1)
        lam_t = lam / (1.0 + 2.0 * lam * s2)
        return lam_t * grad

    if method == "second_order":
        s2 = posterior_var(t).view(-1, 1, 1)
        eye = torch.eye(2, device=mu.device).unsqueeze(0)
        # grad log h = lam * Sigma * grad / sigma^2  =  (I - lam*s2*H)^{-1} (lam*grad)
        M = eye - lam * s2 * H
        return torch.linalg.solve(M, (lam * grad).unsqueeze(-1)).squeeze(-1)

    raise ValueError(method)


def compute_guidance(x, t, velocity_net, lam, method):
    """FIX 4. Chain rule qua mu(x) = x + v(t,x)(1-t).

    Bản cũ bỏ hẳn Jacobian (ngầm coi dmu/dx = I). Surrogate dưới đây cho
    J^T @ g mà KHÔNG sinh đạo hàm bậc ba, vì g đã được detach.
    """
    if method == "unguided":
        return torch.zeros_like(x)

    with torch.enable_grad():
        xi = x.detach().requires_grad_(True)
        mu = xi + velocity_net(t, xi) * (1.0 - t).unsqueeze(-1)
        g = _guidance_at_mu(mu, t, lam, method).detach()
        surrogate = (mu * g).sum()
        return torch.autograd.grad(surrogate, xi)[0].detach()


# ---------------------------------------------------------------- sampler

@torch.no_grad()
def sample_ridge(velocity_net, num_samples, method, num_ode_steps=200, lam=1.5, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    dt = 1.0 / num_ode_steps
    x = torch.randn(num_samples, 2) * rescale

    for step in range(num_ode_steps):
        t = torch.full((num_samples,), step * dt)
        v = velocity_net(t, x)

        if step < num_ode_steps - 1:
            g = compute_guidance(x, t, velocity_net, lam, method)
            b = memoryless_var(t, dt).view(-1, 1)
            x = x + (v + 0.5 * b * g) * dt
        else:
            x = x + v * dt
    return x


def generate_target_samples(lam, num_samples=2000, seed=0):
    torch.manual_seed(seed)
    base = sample_checkerboard(num_samples * 50)
    logw = lam * ridge_reward(base)
    w = torch.softmax(logw, dim=0)          # ổn định số học, thay cho exp/sum
    idx = torch.multinomial(w, num_samples, replacement=True)
    return base[idx]


# ---------------------------------------------------------------- metrics

def analyze(samples, target):
    bad = (torch.isnan(samples).any() or torch.isinf(samples).any()
           or samples.abs().max() > 1e4)
    if bad:
        return dict(ratio=float("nan"), reward=float("nan"),
                    w2=float("nan"), in_dist=0.0)

    L = torch.linalg.eigvalsh(torch.cov(samples.T))
    ratio = math.sqrt(max(L[1].item(), 1e-12) / max(L[0].item(), 1e-12))

    M = ot.dist(samples.numpy(), target.numpy())
    a = np.ones(len(samples)) / len(samples)
    b = np.ones(len(target)) / len(target)
    w2 = math.sqrt(ot.emd2(a, b, M, numItermax=500000))

    return dict(ratio=ratio,
                reward=ridge_reward(samples).mean().item(),
                w2=w2,
                in_dist=(checkerboard_density(samples) > 0.5).astype(float).mean().item())


# ---------------------------------------------------------------- diagnostics

def check_limits(velocity_net):
    """Hai kiểm tra giới hạn. Chạy TRƯỚC mọi sweep."""
    x = torch.randn(256, 2) * rescale
    t = torch.full((256,), 0.5)

    print("\n[1] lam -> 0 : second_order phải TRÙNG plugin")
    for lam in [1e-3, 1e-2]:
        g1 = compute_guidance(x, t, velocity_net, lam, "plugin")
        g2 = compute_guidance(x, t, velocity_net, lam, "second_order")
        rel = ((g2 - g1).norm() / g1.norm()).item()
        print(f"  lam={lam:<7} sai số tương đối = {rel:.2e}   {'OK' if rel < 1e-2 else 'FAIL'}")

    print("\n[2] lam -> inf : ||g|| phải TĂNG rồi BÃO HOÀ (không giảm)")
    prev = None
    for lam in [0.5, 1.5, 5.0, 15.0, 50.0, 200.0]:
        n1 = compute_guidance(x, t, velocity_net, lam, "plugin").norm(dim=-1).mean().item()
        n2 = compute_guidance(x, t, velocity_net, lam, "second_order").norm(dim=-1).mean().item()
        flag = "" if prev is None or n2 >= prev - 1e-6 else "  <-- GIẢM, nghi bug"
        print(f"  lam={lam:<7} ||g_plugin||={n1:9.3f}  ||g_2nd||={n2:8.4f}{flag}")
        prev = n2

    print("\n[3] giới hạn giải tích: g_2nd -> (c - mu)/sigma_t^2 khi lam -> inf")
    with torch.no_grad():
        mu = x + velocity_net(t, x) * (1.0 - t).unsqueeze(-1)
    g = _guidance_at_mu(mu, t, 1e4, "second_order")      # KHÔNG qua Jacobian
    lim = (reward_center - mu) / posterior_var(t).view(-1, 1)
    print(f"  ||g_mu(lam=1e4) - (c-mu)/s2|| / ||(c-mu)/s2|| = "
          f"{((g - lim).norm() / lim.norm()).item():.2e}")


# ---------------------------------------------------------------- driver

def run_phase1(lams=(1.5, 5.0, 15.0), seeds=(0, 1, 2), n=2000, steps=200):
    ck = torch.load("results/velocity_net.pt", map_location=device, weights_only=False)
    net = VelocityMLP(hidden_dim=256, num_layers=4, rescale=rescale).to(device)
    net.load_state_dict(ck["model"])
    net.eval()

    check_limits(net)

    rows = []
    for lam in lams:
        tgt = generate_target_samples(lam, n, seed=0)
        tm = analyze(tgt, tgt)

        # W2 floor: hai tập độc lập cùng rút từ target
        floor = analyze(generate_target_samples(lam, n, seed=1), tgt)["w2"]

        print(f"\n=== lam={lam}  target: ratio={tm['ratio']:.2f} "
              f"reward={tm['reward']:.3f} in_dist={tm['in_dist']:.3f} floor={floor:.3f}")
        rows.append(dict(lam=lam, method="target", seed=-1, floor=floor, **tm))

        for method in ["unguided", "plugin", "plugin_damping", "second_order"]:
            for sd in seeds:
                t0 = time.time()
                s = sample_ridge(net, n, method, steps, lam, seed=sd)
                m = analyze(s, tgt)
                rows.append(dict(lam=lam, method=method, seed=sd, floor=floor, **m))
                print(f"  {method:<16} seed={sd} W2={m['w2']:.3f} "
                      f"(-floor={m['w2']-floor:+.3f}) ratio={m['ratio']:.2f} "
                      f"dR={m['reward']-tm['reward']:+.3f} "
                      f"in_dist={m['in_dist']:.3f} [{time.time()-t0:.0f}s]")

            pd.DataFrame(rows).to_csv("phase1_results.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n", df.groupby(["lam", "method"])[["w2", "ratio", "reward", "in_dist"]]
          .mean().to_string())
    return df


if __name__ == "__main__":
    run_phase1()
