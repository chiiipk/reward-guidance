import re

with open("pipeline.py", "r") as f:
    code = f.read()

# Replace hessian computation and add assert
new_hessian = """            assert latents.shape[0] == 1, "Second-order guidance currently only supports batch size 1"
            def head_1d(z1d):
                return head_fn(z1d.unsqueeze(0)).squeeze(0)
            
            from torch.func import hessian
            H_g = hessian(head_1d)(z_feat[0])"""

code = re.sub(
    r"            def scalar_head_fn\(feat\):.*?H_g = H_g\[0, :, 0, :\]",
    new_hessian,
    code,
    flags=re.DOTALL
)

# Fix raw_grad_norm
old_norm = """        raw_grad_norm = float(g_2nd.norm().item())
        if gradient_norm_scale is not None:"""
new_norm = """        raw_grad_norm = float(g_1st.norm().item())
        if gradient_norm_scale is not None:"""

code = code.replace(old_norm, new_norm)

with open("pipeline.py", "w") as f:
    f.write(code)

print("Fixed pipeline.py")
