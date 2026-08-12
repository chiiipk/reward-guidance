import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from checkerboard.sample import _posterior_var

def _damped_lam(lam, t, sigma_data):
    # This must match what is in checkerboard/sample.py
    # Wait, let's just copy the logic from sample.py
    sigma_1t_sq = _posterior_var(t, sigma_data)
    return lam / (1 + 2 * lam * sigma_1t_sq)

sigma_damp = 1.73

for t in [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]:
    lam_t = _damped_lam(50.0, t, sigma_damp)
    sig_1t_sq = _posterior_var(t, sigma_damp)
    print(f"t={t:.2f} | lam_t={lam_t:.4f} | sig2_1t={sig_1t_sq:.4f}")
