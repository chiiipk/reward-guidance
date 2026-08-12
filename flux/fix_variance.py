with open("pipeline.py", "r") as f:
    code = f.read()

# Add posterior_variance
var_code = """
def posterior_variance(sigma: float) -> float:
    \"\"\"Var[x_1 | x_t] = σ² / (σ² + (1−σ)²) for x_t = (1−σ)x_1 + σε.\"\"\"
    t_interp = 1.0 - sigma
    num = sigma ** 2
    den = num + t_interp ** 2 + 1e-12
    return num / den

assert abs(posterior_variance(1.0) - 1.0) < 1e-5
assert posterior_variance(0.0) < 1e-5

def damped_lambda_factor(sigma_t: float, sigma_damp: float,
                          lambda_eff: float) -> float:
    \"\"\"λ_t / λ = 1 / (1 + 2 λ σ²_{1|t}).\"\"\"
    # Incorporate sigma_damp by scaling the variance
    t_interp = 1.0 - sigma_t
    s = sigma_damp
    num = s * s * (1.0 - t_interp) ** 2
    den = (1.0 - t_interp) ** 2 + t_interp ** 2 * s * s + 1e-12
    sigma_1t_sq = num / den
    return 1.0 / (1.0 + 2.0 * lambda_eff * sigma_1t_sq)
"""
import re
code = re.sub(r"def damped_lambda_factor.*?return 1\.0 / \(1\.0 \+ 2\.0 \* lambda_eff \* sigma_1t_sq\)", var_code, code, flags=re.DOTALL)

# Fix c = lam * (sigma_value ** 2) * reward_scale -> c = lam * posterior_variance(sigma_prime) * reward_scale
code = code.replace("c = lam * (sigma_value ** 2) * reward_scale", "c = lam * posterior_variance(sigma_prime) * reward_scale")

with open("pipeline.py", "w") as f:
    f.write(code)
print("Fixed variance!")
