"""Smoke test cho second-order guidance — chay tren CPU, khong can FLUX weights.

Muc dich: bat het loi RUNTIME truoc khi ton GPU-hour tren H200.

Khac voi DummyPipe (dung ham tuyen tinh tam thuong), file nay dung MANG THAT
nho — co conv, attention, upsample — nen no bat duoc:
  - retain_graph qua k lan VJP tren do thi sau
  - torch.func.hessian co xung dot voi autograd truyen thong khong
  - eigh voi d=768 va dtype thap
  - is_grads_batched co chay tren do thi co attention khong
  - so tri rieng am giu lai co > 0 khong (neu = 0 thi second-order khong lam gi)

Chay:  python3 smoke_test_second_order.py
"""

import math
import importlib.util
import sys
import time
import traceback
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def _install_diffusers_test_stubs():
    """Allow math smoke tests to run before the heavyweight diffusers install."""
    if importlib.util.find_spec("diffusers") is not None:
        return

    pipeline_flux = types.ModuleType("diffusers.pipelines.flux.pipeline_flux")
    pipeline_output = types.ModuleType("diffusers.pipelines.flux.pipeline_output")

    class FluxPipeline:
        pass

    class FluxPipelineOutput:
        def __init__(self, images=None):
            self.images = images

    pipeline_flux.FluxPipeline = FluxPipeline
    pipeline_flux.calculate_shift = lambda *args, **kwargs: 0.0
    pipeline_flux.retrieve_timesteps = lambda *args, **kwargs: ([], 0)
    pipeline_output.FluxPipelineOutput = FluxPipelineOutput

    modules = {
        "diffusers": types.ModuleType("diffusers"),
        "diffusers.pipelines": types.ModuleType("diffusers.pipelines"),
        "diffusers.pipelines.flux": types.ModuleType("diffusers.pipelines.flux"),
        "diffusers.pipelines.flux.pipeline_flux": pipeline_flux,
        "diffusers.pipelines.flux.pipeline_output": pipeline_output,
    }
    sys.modules.update(modules)


_install_diffusers_test_stubs()

PASS, FAIL = [], []


def check(name, fn):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    try:
        fn()
        PASS.append(name)
        print(f"[PASS] {name}")
    except Exception as e:
        FAIL.append((name, repr(e)))
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()


# --------------------------------------------------------------- mang gia lap
class TinyTransformer(nn.Module):
    """Dung lai attention + residual de do thi du sau va co op giong FLUX."""

    def __init__(self, c=64, heads=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))
        self.temb = nn.Linear(1, c)

    def forward(self, x, sigma):  # x: (B,N,C)
        h = self.n1(x) + self.temb(sigma.view(-1, 1, 1))
        a, _ = self.attn(h, h, h)
        x = x + a
        return x + self.mlp(self.n2(x))


class TinyVAE(nn.Module):
    """Decode packed latents -> anh. Co conv + upsample nhu VAE that."""

    def __init__(self, c=64):
        super().__init__()
        self.p = nn.Linear(c, 32)
        self.c1 = nn.Conv2d(32, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x, h, w):  # x: (B,N,C), N = (h/8)*(w/8)
        B, N, _ = x.shape
        s = int(math.sqrt(N))
        z = self.p(x).transpose(1, 2).reshape(B, 32, s, s)
        z = F.silu(self.c1(z))
        z = F.interpolate(z, scale_factor=2, mode="nearest")
        return torch.tanh(self.c2(z))


class FakeCLIPHead(nn.Module):
    """Encoder -> feature d chieu -> MLP head. Mo phong ImageReward."""

    def __init__(self, d=768):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(128, d)
        self.head = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))
        self.d = d

    def features(self, img):
        return self.proj(self.enc(img).mean(dim=(2, 3)))  # (B,d)

    def head_fn(self, z):
        return self.head(z).squeeze(-1)


# --------------------------------------------------------------- pipeline gia
class SmokePipe:
    """Gan cac method THAT tu pipeline.py vao. Neu import loi -> bao ro."""

    def __init__(self, d_feat=768, dtype=torch.float32):
        self.tr = TinyTransformer().to(dtype)
        self.vae = TinyVAE().to(dtype)
        self.rw = FakeCLIPHead(d_feat).to(dtype)
        for p in (
            list(self.tr.parameters())
            + list(self.vae.parameters())
            + list(self.rw.parameters())
        ):
            p.requires_grad_(False)

    def _flow_map_x0(
        self, z, sigma_t, prompt_embeds, pooled, text_ids, img_ids, guidance
    ):
        return self.tr(z, sigma_t)

    def _decode_packed(self, x0, h, w):
        return self.vae(x0, h, w)


def bind_real_methods(pipe):
    from pipeline import GuidedFluxPipeline

    SmokePipe._compute_grad_k1 = GuidedFluxPipeline._compute_grad_k1
    SmokePipe._compute_grad_second_order = GuidedFluxPipeline._compute_grad_second_order
    return pipe


# --------------------------------------------------------------- reward fns
def make_reward(rw, kind="clip"):
    if kind == "palette":
        target = torch.tensor([[0.2, 0.5, 0.9]])

        def fn(image, return_features=False):
            z = image.mean(dim=(2, 3))

            def head_fn(zi):
                return -((zi - target.to(zi)) ** 2).sum(-1)

            s = head_fn(z)
            return (s, z, head_fn) if return_features else s

    else:

        def fn(image, return_features=False):
            reward_dtype = next(rw.parameters()).dtype
            z = rw.features(image.to(reward_dtype))
            s = rw.head_fn(z)
            return (s, z, rw.head_fn) if return_features else s

    fn.supports_features = True
    return fn


# --------------------------------------------------------------- args chung
def make_args(pipe, dtype=torch.float32, N=64, C=64):
    lat = torch.randn(1, N, C, dtype=dtype)
    return dict(
        latents=lat,
        sigma_value=0.5,
        snr_factor=5.0,
        reward_scale=2.0,
        gradient_norm_scale=None,
        height=64,
        width=64,
        prompt_embeds=torch.randn(1, 8, C, dtype=dtype),
        pooled_prompt_embeds=torch.randn(1, C, dtype=dtype),
        text_ids=torch.randn(8, 3, dtype=dtype),
        latent_image_ids=torch.randn(N, 3, dtype=dtype),
        guidance=None,
    )


# =============================================================== CAC BAI TEST


def t_posterior_variance():
    """Phuong sai hau nghiem PHAI la sigma^2/(sigma^2+(1-sigma)^2), khong phai sigma^2."""
    import pipeline as P

    fname = [n for n in dir(P) if "posterior" in n.lower() or "post_var" in n.lower()]
    if not fname:
        raise AssertionError(
            "Khong tim thay ham posterior variance trong pipeline.py.\n"
            "Can mot ham dung chung:\n"
            "    def posterior_var(s): return s**2 / (s**2 + (1-s)**2)\n"
            "va dung no cho `c`, KHONG dung sigma**2 tran."
        )
    f = getattr(P, fname[0])
    assert abs(f(1.0) - 1.0) < 1e-6, f"sigma=1 (nhieu) phai cho 1.0, duoc {f(1.0)}"
    assert abs(f(0.0) - 0.0) < 1e-6, f"sigma=0 (data) phai cho 0.0, duoc {f(0.0)}"
    assert abs(f(0.5) - 0.5) < 1e-6, f"sigma=0.5 phai cho 0.5, duoc {f(0.5)}"
    print(
        f"  ham: {fname[0]}   f(.9)={f(0.9):.4f}  f(.5)={f(0.5):.4f}  f(.1)={f(0.1):.4f}"
    )


def t_lambda_zero(kind="clip"):
    """lam -> 0: second-order phai trung plug-in, sai so ti le TUYEN TINH voi lam."""
    torch.manual_seed(0)
    pipe = bind_real_methods(SmokePipe(dtype=torch.float64))
    rfn = make_reward(pipe.rw, kind)
    a = make_args(pipe, torch.float64)

    ratios = []
    for lam in [1e-2, 1e-3, 1e-4]:
        g1, _, _ = pipe._compute_grad_k1(
            **a, reward_fn=rfn, generator=torch.Generator().manual_seed(7)
        )
        g2, _, _ = pipe._compute_grad_second_order(
            **a,
            reward_fn=rfn,
            generator=torch.Generator().manual_seed(7),
            lam=lam,
            k_eig=16,
        )
        rel = ((g2 - lam * g1).norm() / (lam * g1).norm()).item()
        ratios.append(rel / lam)
        print(f"  lam={lam:<8.0e} rel_err={rel:.3e}  rel_err/lam={rel/lam:.5f}")

    spread = max(ratios) / max(min(ratios), 1e-30)
    assert (
        spread < 1.15
    ), f"rel_err/lam khong hang (spread {spread:.3f}). Con thieu/thua he so."
    print(f"  -> ti le hang (spread {spread:.4f}) OK")


def t_eigenvalues_kept():
    """So tri rieng am giu lai phai > 0. Neu = 0 thi second-order KHONG lam gi."""
    torch.manual_seed(0)
    pipe = SmokePipe(dtype=torch.float64)
    rfn = make_reward(pipe.rw, "clip")
    img = pipe.vae(
        pipe.tr(
            torch.randn(1, 64, 64, dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
        ),
        64,
        64,
    )
    _, z, head = rfn(img, return_features=True)

    from torch.func import hessian

    H = hessian(lambda v: head(v.unsqueeze(0)).squeeze(0))(z[0])
    ev = torch.linalg.eigvalsh(H.double())
    neg = int((ev < -1e-6).sum())
    print(
        f"  d={H.shape[0]}  tri rieng am: {neg}/{len(ev)}  "
        f"min={ev.min():.3e}  max={ev.max():.3e}"
    )
    assert neg > 0, (
        "KHONG co tri rieng am nao. Second-order se lang le suy bien ve first-order.\n"
        "Them dong log nay vao pipeline.py de theo doi luc chay that."
    )


def t_hessian_backends_agree():
    """torch.func.hessian phai khop autograd.functional.hessian."""
    torch.manual_seed(0)
    rw = FakeCLIPHead(64).to(torch.float64)
    z = torch.randn(64, dtype=torch.float64)
    f = lambda v: rw.head_fn(v.unsqueeze(0)).squeeze(0)
    from torch.func import hessian

    H1 = hessian(f)(z)
    H2 = torch.autograd.functional.hessian(f, z)
    err = (H1 - H2).abs().max().item()
    print(f"  sai lech toi da: {err:.3e}")
    assert err < 1e-8, "hai backend Hessian khong khop"


def t_dtype_bf16():
    """eigh tren bfloat16 mat chinh xac nghiem trong -> phai ep .double() truoc."""
    torch.manual_seed(0)
    rw = FakeCLIPHead(256)
    rw.double()
    z = torch.randn(256)
    f = lambda v: rw.head_fn(v.unsqueeze(0)).squeeze(0)
    from torch.func import hessian

    H64 = hessian(f)(z.double())
    ev64 = torch.linalg.eigvalsh(H64)
    rw.float()
    z32 = z.float()
    f32 = lambda v: rw.head_fn(v.unsqueeze(0)).squeeze(0)
    ev32 = torch.linalg.eigvalsh(hessian(f32)(z32).float())
    err = (ev64 - ev32.double()).abs().max().item()
    print(f"  |eig(f64) - eig(f32)| max = {err:.3e}")
    try:
        torch.linalg.eigvalsh(H64.bfloat16())
        print("  bfloat16 eigh: chay duoc (van nen ep .double() truoc khi eigh)")
    except Exception as e:
        print(
            f"  bfloat16 eigh: KHONG chay ({type(e).__name__}) -> BAT BUOC ep .double()"
        )


def t_retain_graph_k_vjp():
    """k lan VJP voi retain_graph tren do thi that. Do thoi gian + bo nho."""
    torch.manual_seed(0)
    pipe = SmokePipe()
    x = torch.randn(1, 64, 64, requires_grad=True)
    img = pipe.vae(pipe.tr(x, torch.tensor([0.5])), 64, 64)
    z = pipe.rw.features(img)

    t0 = time.time()
    _ = torch.autograd.grad(z, x, grad_outputs=torch.randn_like(z), retain_graph=True)[
        0
    ]
    t1 = time.time()
    for _ in range(15):
        torch.autograd.grad(z, x, grad_outputs=torch.randn_like(z), retain_graph=True)
    t2 = time.time()
    print(
        f"  1 VJP: {t1-t0:.4f}s   16 VJP: {t2-t0:.4f}s   "
        f"ti le: {(t2-t0)/max(t1-t0,1e-9):.2f}x"
    )
    print("  (tren H200 con so nay quyet dinh k. >20x thi phai giam k)")


def t_batched_vjp():
    """is_grads_batched thuong FAIL tren do thi co attention. Biet truoc thi tot."""
    torch.manual_seed(0)
    pipe = SmokePipe()
    x = torch.randn(1, 64, 64, requires_grad=True)
    img = pipe.vae(pipe.tr(x, torch.tensor([0.5])), 64, 64)
    z = pipe.rw.features(img)
    V = torch.randn(8, *z.shape)
    try:
        g = torch.autograd.grad(
            z, x, grad_outputs=V, retain_graph=True, is_grads_batched=True
        )[0]
        ref = torch.stack(
            [
                torch.autograd.grad(z, x, grad_outputs=V[i], retain_graph=True)[0]
                for i in range(8)
            ]
        )
        err = (g - ref).abs().max().item()
        print(f"  is_grads_batched CHAY, sai lech vs tuan tu = {err:.3e}")
        assert err < 1e-4, "batched VJP cho ket qua khac tuan tu"
    except RuntimeError as e:
        print(f"  is_grads_batched KHONG chay: {e}")
        print("  -> giu vong lap tuan tu, khong sao")


def t_batch_assert():
    """batch > 1 phai bao loi ro rang, khong im lang cho ket qua sai."""
    pipe = bind_real_methods(SmokePipe())
    rfn = make_reward(pipe.rw, "clip")
    a = make_args(pipe)
    a["latents"] = torch.randn(2, 64, 64)
    try:
        pipe._compute_grad_second_order(**a, reward_fn=rfn, generator=None, lam=1.0)
        raise AssertionError("batch=2 KHONG bao loi -> H_g se lay sai block")
    except (AssertionError, ValueError) as e:
        if "batch" not in str(e).lower():
            raise
        print(f"  bao loi dung: {e}")


def t_no_nan():
    """Chay that voi lam lon, kiem tra NaN/Inf va do lon guidance."""
    torch.manual_seed(0)
    pipe = bind_real_methods(SmokePipe())
    rfn = make_reward(pipe.rw, "clip")
    for lam in [1.0, 10.0, 100.0]:
        for s in [0.9, 0.5, 0.1]:
            a = make_args(pipe)
            a["sigma_value"] = s
            g, _, _ = pipe._compute_grad_second_order(
                **a,
                reward_fn=rfn,
                generator=torch.Generator().manual_seed(1),
                lam=lam,
                k_eig=16,
            )
            assert torch.isfinite(g).all(), f"NaN/Inf tai lam={lam}, sigma={s}"
            print(f"  lam={lam:<6} sigma={s}  ||g||={g.norm():.4e}")


def t_saturation():
    """||g|| phai TANG roi BAO HOA theo lam, khong bao gio GIAM."""
    torch.manual_seed(0)
    pipe = bind_real_methods(SmokePipe())
    rfn = make_reward(pipe.rw, "clip")
    prev, norms = None, []
    for lam in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        a = make_args(pipe)
        g, _, _ = pipe._compute_grad_second_order(
            **a,
            reward_fn=rfn,
            generator=torch.Generator().manual_seed(1),
            lam=lam,
            k_eig=16,
        )
        n = g.norm().item()
        norms.append(n)
        flag = "  <-- GIAM, nghi bug" if prev and n < prev * 0.99 else ""
        print(f"  lam={lam:<8} ||g||={n:.4e}{flag}")
        prev = n
    assert all(
        norms[i + 1] >= norms[i] * 0.99 for i in range(len(norms) - 1)
    ), "||g|| giam theo lam -> con thieu he so"


# =============================================================== main
if __name__ == "__main__":
    print(f"torch {torch.__version__}  |  device: CPU  |  khong can FLUX weights\n")

    check("1. Phuong sai hau nghiem dung cong thuc", t_posterior_variance)
    check("2. Hai backend Hessian khop nhau", t_hessian_backends_agree)
    check("3. dtype: eigh can double", t_dtype_bf16)
    check("4. So tri rieng am giu lai > 0", t_eigenvalues_kept)
    check(
        "5. lam->0 trung plug-in (reward CLIP gia, d=768)",
        lambda: t_lambda_zero("clip"),
    )
    check("6. lam->0 trung plug-in (palette, d=3)", lambda: t_lambda_zero("palette"))
    check("7. retain_graph qua 16 VJP", t_retain_graph_k_vjp)
    check("8. is_grads_batched", t_batched_vjp)
    check("9. batch>1 bao loi ro rang", t_batch_assert)
    check("10. khong NaN o lam lon / sigma nho", t_no_nan)
    check("11. ||g|| tang roi bao hoa theo lam", t_saturation)

    print(f"\n{'='*70}\nPASS {len(PASS)}  |  FAIL {len(FAIL)}\n{'='*70}")
    for n, e in FAIL:
        print(f"  FAIL  {n}\n        {e}")
    sys.exit(1 if FAIL else 0)
