import torch
import traceback
from smoke_test_second_order import (
    SmokePipe, bind_real_methods, make_reward, make_args
)

def test_lambda(kind, lambdas):
    print(f"--- Running {kind} ---")
    torch.manual_seed(0)
    pipe = bind_real_methods(SmokePipe(dtype=torch.float64))
    rfn = make_reward(pipe.rw, kind)
    a = make_args(pipe, torch.float64)

    ratios = []
    for lam in lambdas:
        # compute first-order
        g1, _, _ = pipe._compute_grad_k1(
            **a, reward_fn=rfn, generator=torch.Generator().manual_seed(7))
        
        # intercept the eigenvalues
        import pipeline
        original_eigh = torch.linalg.eigh
        num_neg = []
        def intercept_eigh(*args, **kwargs):
            evals, evecs = original_eigh(*args, **kwargs)
            k = int((evals < -1e-6).sum())
            num_neg.append(k)
            return evals, evecs
        torch.linalg.eigh = intercept_eigh
        
        g2, _, _ = pipe._compute_grad_second_order(
            **a, reward_fn=rfn, generator=torch.Generator().manual_seed(7),
            lam=lam, k_eig=16)
        
        torch.linalg.eigh = original_eigh
        
        kept = num_neg[0] if num_neg else 0
        
        rel = ((g2 - lam * g1).norm() / (lam * g1).norm()).item()
        ratios.append(rel / lam)
        print(f"  lam={lam:<8.0e} rel_err={rel:.3e}  rel_err/lam={rel/lam:.5f}  (U_kept={kept})")

test_lambda("clip", [1e-1, 1e-2, 1e-3, 1e-4])
test_lambda("palette", [1e-1, 1e-2, 1e-3, 1e-4])
