"""TANG 1 — Mixture-posterior second-order guidance tren checkerboard.

Y tuong: hau nghiem p(x_1 | x_t) cua qua trinh GOC, khi p_data la GMM,
CUNG LA mot GMM — dang dong, KHONG can lay mau, KHONG goi glass_flow.

  h_t(x) = E[e^{lam r(X_1)} | X_t = x]
         = [ sum_k pi_k N_k(x) Z_k(x) ] / p_base(x)

  => grad log h = grad log( sum_k pi_k N_k Z_k )  -  grad log p_base

Ca hai so hang deu dang dong. Chi phi them so voi second-order 1 Gaussian:
K lan giai he 2x2 moi buoc. Khong goi mang them lan nao.

Convention khop repo: t=0 nhieu, t=1 data.  x_t = t*x_1 + (1-t)*rescale*eps
"""
import math
import numpy as np
import torch

torch.set_num_threads(1)

# ----------------------------------------------------------------- setup
RESCALE = 1.73
S_CELL = 1.0 / math.sqrt(12.0)          # std cua phan bo deu tren o 1x1
SIGMA_R = 0.5


def gray_cell_centers(half=3):
    """Tam cac o XAM. Quy uoc: xam <=> floor(x)+floor(y) chan."""
    cs = [(i + 0.5, j + 0.5)
          for i in range(-half, half) for j in range(-half, half)
          if (i + j) % 2 == 0]
    return torch.tensor(cs, dtype=torch.float32)          # (M,2)


CENTERS = gray_cell_centers()
M = CENTERS.shape[0]

# sanity: std cua GMM nay phai ~ RESCALE
_var = (CENTERS ** 2).mean().item() + S_CELL ** 2
assert abs(math.sqrt(_var) - RESCALE) < 0.05, f"std GMM = {math.sqrt(_var):.3f}"


def is_gray(x):
    return ((torch.floor(x[:, 0]) + torch.floor(x[:, 1])) % 2 == 0) & \
           (x.abs().max(-1).values < 3.0)


# ----------------------------------------------------------------- reward
def reward(x, c, sr=SIGMA_R):
    return torch.exp(-((x - c) ** 2).sum(-1) / (2 * sr ** 2))


def reward_grad_hess(x, c, sr=SIGMA_R):
    """grad r va Gauss-Newton Hessian (bo so hang rank-1 loi -> luon am xac dinh).

    r  = exp(-|d|^2/2sr^2),  d = x-c
    gr = -d/sr^2 * r
    H_true = -(r/sr^2)(I - d d^T/sr^2)   -> phan +r d d^T/sr^4 la LOI
    H_GN   = -(r/sr^2) I                 -> am xac dinh, KHONG can clamp
    """
    d = x - c
    r = torch.exp(-(d ** 2).sum(-1) / (2 * sr ** 2))
    g = -d / sr ** 2 * r.unsqueeze(-1)
    H = -torch.eye(2, device=x.device).expand(*x.shape[:-1], 2, 2) \
        * (r / sr ** 2).unsqueeze(-1).unsqueeze(-1)
    return g, H, r


# ----------------------------------------------------------------- hau nghiem GMM
def posterior_gmm(x, t):
    """Tra ve (log_w, nu, tau2, dnu_dx, V_t) cho p(x_1|x_t=x).

    x: (B,2)   t: scalar
    log_w: (B,M)   nu: (B,M,2)   tau2, dnu_dx, V_t: scalar
    """
    a, b = t, (1.0 - t) * RESCALE
    b2 = max(b ** 2, 1e-12)
    tau2 = 1.0 / (1.0 / S_CELL ** 2 + a ** 2 / b2)
    V = a ** 2 * S_CELL ** 2 + b2                       # var cua x_t | thanh phan k

    C = CENTERS.to(x.device)                            # (M,2)
    nu = tau2 * (C / S_CELL ** 2 + a * x.unsqueeze(1) / b2)   # (B,M,2)
    lw = -((x.unsqueeze(1) - a * C) ** 2).sum(-1) / (2 * V)   # (B,M), pi_k deu
    return lw, nu, tau2, tau2 * a / b2, V


# ----------------------------------------------------------------- log Z (bac hai)
def log_Z(nu, tau2, lam, c):
    """log E_{N(nu, tau2 I)}[e^{lam r}] va grad theo nu, xap xi bac hai.

    log Z = lam*r(nu) - 0.5*logdet(I - lam*tau2*H) + 0.5*lam^2*tau2*g^T M^{-1} g
    grad_nu log Z = lam * M^{-1} g          (bo dao ham cua H -> khong bac ba)
    """
    g, H, r = reward_grad_hess(nu, c)                   # (B,M,2), (B,M,2,2)
    eye = torch.eye(2, device=nu.device).expand_as(H)
    Mx = eye - lam * tau2 * H                           # luon xac dinh duong (H<=0)
    Minv_g = torch.linalg.solve(Mx, g.unsqueeze(-1)).squeeze(-1)

    logdet = torch.log(torch.linalg.det(Mx).clamp_min(1e-30))
    quad = 0.5 * lam ** 2 * tau2 * (g * Minv_g).sum(-1)
    return lam * r - 0.5 * logdet + quad, lam * Minv_g


# ----------------------------------------------------------------- guidance
def mixture_guidance(x, t, lam, c, K=None):
    """grad_x log h_t(x) voi hau nghiem hon hop.

    K=None  -> dung TAT CA M thanh phan (chinh xac)
    K=int   -> chi giu K thanh phan co trong so lon nhat
    K=1     -> suy bien ve second-order mot Gaussian
    """
    lw, nu, tau2, dnu, V = posterior_gmm(x, t)

    if K is not None and K < lw.shape[1]:
        idx = lw.topk(K, dim=1).indices                              # (B,K)
        lw = lw.gather(1, idx)
        nu = nu.gather(1, idx.unsqueeze(-1).expand(-1, -1, 2))
        Csel = CENTERS.to(x.device)[idx]                             # (B,K,2)
    else:
        Csel = CENTERS.to(x.device).expand(x.shape[0], -1, -1)

    lZ, G = log_Z(nu, tau2, lam, c)                                  # (B,K),(B,K,2)

    # grad log( sum_k pi_k N_k Z_k )
    wt = torch.softmax(lw + lZ, dim=1).unsqueeze(-1)                 # (B,K,1)
    term_N = -(x.unsqueeze(1) - t * Csel) / V                        # grad log N_k
    grad_num = (wt * (term_N + dnu * G)).sum(1)

    # grad log p_base  (cung hon hop, KHONG co Z)
    wb = torch.softmax(lw, dim=1).unsqueeze(-1)
    grad_den = (wb * term_N).sum(1)

    return grad_num - grad_den


# ----------------------------------------------------------------- sampler
@torch.no_grad()
def sample(velocity_net, n, lam, c, K=None, steps=200, seed=0, guided=True):
    torch.manual_seed(seed)
    dt = 1.0 / steps
    x = torch.randn(n, 2) * RESCALE
    for i in range(steps):
        t = i * dt
        tt = torch.full((n,), t)
        v = velocity_net(tt, x)
        if guided and i < steps - 1:
            g = mixture_guidance(x, t, lam, c, K)
            beta = 2.0 * (1.0 - t) * RESCALE ** 2       # khop phase1 cu
            x = x + (v + 0.5 * beta * g) * dt
        else:
            x = x + v * dt
    return x


# ----------------------------------------------------------------- target + metrics
def target_samples(lam, c, n=4000, seed=0):
    """Mau tu p_data * e^{lam r}, bang luoi roi rac (chinh xac, on dinh moi lam)."""
    torch.manual_seed(seed)
    h = 6.0 / 600
    g = torch.linspace(-3 + h/2, 3 - h/2, 600)
    G = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    mask = is_gray(G)
    lw = lam * reward(G, c)
    lw = torch.where(mask, lw, torch.full_like(lw, -1e30))
    w = torch.softmax(lw, 0)
    idx = torch.multinomial(w, n, replacement=True)
    return G[idx] + (torch.rand(n, 2) - 0.5) * h


def cell_id(x):
    return (torch.floor(x[:, 0]) + 100 * torch.floor(x[:, 1])).long()


def metrics(s, tgt, c):
    if torch.isnan(s).any() or s.abs().max() > 1e4:
        return dict(entropy=float("nan"), coverage=0, in_dist=0.0, w2=float("nan"), reward=0.0)
    ids = cell_id(s[is_gray(s)])
    _, cnt = torch.unique(ids, return_counts=True)
    p = cnt.float() / cnt.sum()
    try:
        import ot
        Mx = ot.dist(s.numpy(), tgt.numpy())
        a = np.ones(len(s)) / len(s); b = np.ones(len(tgt)) / len(tgt)
        w2 = math.sqrt(ot.emd2(a, b, Mx, numItermax=500000))
    except Exception:
        w2 = float("nan")
    return dict(entropy=-(p * p.log()).sum().item(),
                coverage=len(cnt),
                in_dist=is_gray(s).float().mean().item(),
                w2=w2,
                reward=reward(s, c).mean().item())


# ----------------------------------------------------------------- gate
@torch.no_grad()
def gate_distinct_modes(velocity_net, glass_flow, n=512, K=8, steps=200):
    """CHI SO CONG: K hat rut tu hau nghiem THAT phu duoc bao nhieu o phan biet?

    ~1  -> hat cung roi mot mode, mixture suy bien, tang 2 vo dung
    >=2 -> tang 2 song
    """
    dt = 1.0 / steps
    x = torch.randn(n, 2) * RESCALE
    out = []
    for i in range(steps):
        t = i * dt
        tt = torch.full((n,), t)
        if i % 20 == 0 and i > 0:
            xs = x.repeat(K, 1)
            x1 = glass_flow(velocity_net, t, xs, num_steps=50)
            ids = cell_id(x1).view(K, n)
            d = torch.tensor([len(torch.unique(ids[:, j])) for j in range(n)])
            out.append((t, d.float().mean().item()))
        x = x + velocity_net(tt, x) * dt
    return out


# ----------------------------------------------------------------- main
def main():
    from model import VelocityMLP
    ck = torch.load("results/velocity_net.pt", map_location="cpu", weights_only=False)
    net = VelocityMLP(hidden_dim=256, num_layers=4, rescale=RESCALE)
    net.load_state_dict(ck["model"]); net.eval()

    for name, c, lam in [("out-of-support", torch.tensor([0.5, 1.5]), 50.0),
                         ("in-support",     torch.tensor([0.5, 0.5]), 10.0)]:
        tgt = target_samples(lam, c)
        tm = metrics(tgt, tgt, c)
        ln4 = math.log(4)
        print(f"\n=== {name}  lam={lam} ===")
        print(f"  {'method':<24} {'entropy':>9} {'cov':>5} {'in-dist':>8} {'W2':>8} {'reward':>8}")
        print(f"  {'TARGET':<24} {tm['entropy']:9.4f} {tm['coverage']:5d} "
              f"{tm['in_dist']:8.3f} {'-':>8} {tm['reward']:8.3f} (ln4={ln4:.4f})")

        s = sample(net, 4000, lam, c, guided=False)
        m = metrics(s, tgt, c)
        print(f"  {'unguided':<24} {m['entropy']:9.4f} {m['coverage']:5d} "
              f"{m['in_dist']:8.3f} {m['w2']:8.3f} {m['reward']:8.3f}")

        for K in [1, 2, 4, 8, None]:
            for bo4 in [False, True]:
                if bo4:
                    s_all = sample(net, 4000 * 4, lam, c, K=K)
                    r_all = reward(s_all, c)
                    s_all = s_all.view(4000, 4, 2)
                    r_all = r_all.view(4000, 4)
                    best_idx = r_all.argmax(dim=1)
                    s = s_all[torch.arange(4000), best_idx]
                else:
                    s = sample(net, 4000, lam, c, K=K)
                
                m = metrics(s, tgt, c)
                lab = f"mix K={K if K else 'all'} {'+ Bo4' if bo4 else ''}"
                print(f"  {lab:<24} {m['entropy']:9.4f} {m['coverage']:5d} "
                      f"{m['in_dist']:8.3f} {m['w2']:8.3f} {m['reward']:8.3f}")


if __name__ == "__main__":
    main()