import re

with open("pipeline.py", "r") as f:
    code = f.read()

second_order_code = """
    def _compute_grad_second_order(
        self,
        latents: torch.Tensor,
        sigma_value: float,
        sigma_next_value: float,
        snr_factor: float,
        reward_scale: float,
        gradient_norm_scale: Optional[float],
        height: int,
        width: int,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance: Optional[torch.Tensor],
        generator: Optional[torch.Generator],
        grad_divisor: Optional[float] = None,
        lam: float = 1.0,
        k_eig: int = 16,
    ) -> tuple[torch.Tensor, float, float]:
        \"\"\"Second-Order Woodbury Guidance: g_2nd = (I - c J_f^T H_g J_f)^{-1} g_1st.\"\"\"
        if not getattr(reward_fn, "supports_features", False):
            raise ValueError("Second-order requires reward_fn with return_features=True.")

        sqrt_l = math.sqrt(snr_factor)
        sigma_prime = (sqrt_l * sigma_value) / (sqrt_l * sigma_value + 1.0 - sigma_value)
        sigma_prime = min(sigma_prime, 0.9999)

        alpha_t = 1.0 - sigma_value
        alpha_prev = 1.0 - sigma_prime
        var_t = sigma_value ** 2
        var_prev = sigma_prime ** 2
        scale = alpha_prev / max(alpha_t, 1e-8)
        var_q = max(var_prev - scale ** 2 * var_t, 1e-8)
        std_q = math.sqrt(var_q)

        eps = torch.randn(
            latents.shape, device=latents.device, dtype=latents.dtype,
            generator=generator,
        )

        sigma_prime_t = torch.full(
            (latents.shape[0],), sigma_prime,
            device=latents.device, dtype=latents.dtype,
        )

        with torch.enable_grad():
            x_input = latents.detach().requires_grad_(True)
            z = scale * x_input + std_q * eps
            x0_hat = self._flow_map_x0(
                z, sigma_prime_t, prompt_embeds,
                pooled_prompt_embeds, text_ids, latent_image_ids, guidance,
            )

            image = self._decode_packed(x0_hat, height, width).float()
            score, z_feat, head_fn = reward_fn(image, return_features=True)
            
            def scalar_head_fn(feat):
                return head_fn(feat).sum()
            
            H_g = torch.autograd.functional.hessian(scalar_head_fn, z_feat).squeeze()
            if H_g.ndim > 2:
                H_g = H_g[0, :, 0, :]
                
            g_feat = torch.autograd.grad(score.sum(), z_feat, retain_graph=True)[0]
            
            evals, evecs = torch.linalg.eigh(H_g)
            k = min(k_eig, H_g.shape[-1])
            mu = evals[:k]
            U = evecs[:, :k]
            mask = mu < -1e-6
            mu = mu[mask]
            U = U[:, mask]
            
            W_cols = []
            for j in range(U.shape[1]):
                v = U[:, j].view_as(z_feat)
                vjp = torch.autograd.grad(z_feat, x_input, grad_outputs=v, retain_graph=True)[0]
                W_cols.append(vjp.view(-1))
                
            g_1st = torch.autograd.grad(z_feat, x_input, grad_outputs=g_feat, retain_graph=False)[0]
            g_1st = g_1st * reward_scale * lam
            
            if len(W_cols) > 0:
                W = torch.stack(W_cols, dim=1)
                c = lam * (sigma_value ** 2) * reward_scale
                
                M_inv = torch.diag(1.0 / mu).to(W.device)
                K = -M_inv / c + W.T @ W 
                
                W_K_inv = W @ torch.linalg.inv(K)
                
                g_1st_flat = g_1st.view(-1)
                g_2nd_flat = g_1st_flat - W_K_inv @ (W.T @ g_1st_flat)
                g_2nd = g_2nd_flat.view_as(g_1st)
            else:
                g_2nd = g_1st
                
        raw_grad_norm = float(g_2nd.norm().item())
        if gradient_norm_scale is not None:
            grad_norm = g_2nd.norm().clamp_min(1e-8)
            grad = g_2nd / grad_norm * gradient_norm_scale
        elif grad_divisor is not None:
            grad = g_2nd / grad_divisor
        else:
            grad = g_2nd

        return grad.detach(), float(score.detach().mean().item()), raw_grad_norm
"""

# Insert _compute_grad_second_order after _compute_grad_k1
if "_compute_grad_second_order" not in code:
    code = code.replace(
        "    def _compute_grad_k_particles(",
        second_order_code + "\n    def _compute_grad_k_particles("
    )
    
    # Now we need to update __call__ to accept `method` argument
    code = code.replace(
        "        lam: float = 1.0,",
        "        lam: float = 1.0,\n        method: str = 'plugin',"
    )
    
    call_logic = """
            if method == "second_order":
                grad, r_val, raw_norm = self._compute_grad_second_order(
                    latents, sigmas[i], sigmas[i + 1],
                    snr_factor, reward_scale, gradient_norm_scale,
                    height, width, reward_fn,
                    prompt_embeds, pooled_prompt_embeds,
                    text_ids, latent_image_ids, guidance, generator,
                    grad_divisor, lam=lam,
                )
            elif num_particles > 1:
"""
    code = code.replace("            if num_particles > 1:", call_logic)
    
    with open("pipeline.py", "w") as f:
        f.write(code)
    print("Injected second_order into pipeline.py")
else:
    print("second_order already injected")
