import torch
import math

# Use float64 for exact precision tests
torch.set_default_dtype(torch.float64)

def get_gn_hessian_and_grad(x, func, A):
    x.requires_grad_(True)
    z = func(x)
    
    J = []
    for i in range(z.shape[1]):
        grad_x = torch.autograd.grad(z[0, i], x, retain_graph=True)[0]
        J.append(grad_x.view(-1))
    J = torch.stack(J, dim=0) # d x D
    
    H_g = -A # d x d
    H_GN = J.T @ H_g @ J # D x D
    
    r = -0.5 * (z @ A @ z.T).squeeze()
    grad_r = torch.autograd.grad(r, x)[0].view(-1)
    
    return J, H_g, H_GN, grad_r

def test_woodbury(D=100, d=20, k=20):
    print(f"\n--- Testing Woodbury Exactness (D={D}, d={d}, k={k}) ---")
    torch.manual_seed(42)
    x = torch.randn(1, D)
    
    layer1 = torch.nn.Linear(D, D)
    layer2 = torch.nn.Linear(D, d)
    func = lambda x: layer2(torch.tanh(layer1(x)))
    
    B = torch.randn(d, d)
    A = B @ B.T
    
    J, H_g, H_GN, grad_r = get_gn_hessian_and_grad(x, func, A)
    
    c = 0.05
    lam = 1.0
    
    exact_update = torch.linalg.solve(torch.eye(D) - c * H_GN, lam * grad_r)
    
    evals, evecs = torch.linalg.eigh(H_g)
    mu = evals[:k]
    U = evecs[:, :k]
    
    W = J.T @ U 
    
    M_inv = torch.diag(1.0 / mu)
    K = -M_inv / c + W.T @ W 
    
    W_K_inv = W @ torch.linalg.inv(K) 
    
    g_1st = lam * grad_r
    woodbury_update = g_1st - W_K_inv @ (W.T @ g_1st)
    
    error = torch.norm(exact_update - woodbury_update).item()
    print(f"Error between Exact (I-cH)^(-1)g and Woodbury Low-Rank: {error:.2e}")
    if error < 1e-10:
        print("✅ Quadratic Reward Exactness Test Passed!")
    else:
        print("❌ Quadratic Reward Exactness Test Failed!")
        
def test_lambda_limit():
    print(f"\n--- Testing lambda -> 0 Limit ---")
    torch.manual_seed(42)
    D, d, k = 100, 20, 20
    x = torch.randn(1, D)
    layer1 = torch.nn.Linear(D, d)
    func = lambda x: layer1(x)
    
    B = torch.randn(d, d)
    A = B @ B.T
    J, H_g, H_GN, grad_r = get_gn_hessian_and_grad(x, func, A)
    
    sigma_t = 0.9
    c_base = sigma_t ** 2
    
    for lam in [1.0, 0.1, 0.01, 0.001, 0.0001, 1e-5]:
        c = lam * c_base
        g_1st = lam * grad_r
        g_2nd = torch.linalg.solve(torch.eye(D) - c * H_GN, g_1st)
        
        diff = torch.norm(g_2nd - g_1st).item()
        base = torch.norm(g_1st).item()
        rel_diff = diff / base
        
        const = rel_diff / lam
        print(f"lam={lam:7.5f} | rel_err={rel_diff:.2e} | rel_err/lam={const:.4f}")
            
if __name__ == "__main__":
    test_woodbury(D=100, d=20, k=20)
    test_lambda_limit()
