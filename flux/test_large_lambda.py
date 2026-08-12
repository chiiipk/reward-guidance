import torch
from smoke_test_second_order import SmokePipe, bind_real_methods, make_reward, make_args

def test_large_lambda():
    print(f"\n--- Testing Large Lambda Magnitude ---")
    torch.manual_seed(0)
    pipe = bind_real_methods(SmokePipe(dtype=torch.float64))
    rfn = make_reward(pipe.rw, "clip")
    a = make_args(pipe, torch.float64)

    for lam in [1.0, 10.0, 100.0, 1000.0]:
        generator = lambda: torch.Generator().manual_seed(7)
        g1, _, _ = pipe._compute_grad_k1(**a, reward_fn=rfn, generator=generator())
        g2, _, _ = pipe._compute_grad_second_order(**a, reward_fn=rfn,
                                                   generator=generator(), lam=lam, k_eig=16)
        
        c = pipe.last_vars['c']
        W = pipe.last_vars['W']
        mu = pipe.last_vars['mu']
        
        ratio = (g2.norm() / (lam * g1.norm())).item()
        
        W_norm = W.norm().item() if W is not None else 0.0
        mu_max = mu.abs().max().item() if mu is not None and len(mu) > 0 else 0.0
        strength = c * (W_norm ** 2) * mu_max if c is not None else 0.0
        
        print(f"lam={lam:<6} ||g2||/||lam*g1|| = {ratio:.4f}")
        print(f"       c = {c:.4e}, ||W|| = {W_norm:.4e}, max|mu| = {mu_max:.4e} => c*||W||^2*|mu| = {strength:.4e}")

if __name__ == '__main__':
    test_large_lambda()
