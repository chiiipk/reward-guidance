import torch
import math
from pipeline import GuidedFluxPipeline

class DummyPipe:
    def _flow_map_x0(self, z, sigma_t, prompt_embeds, pooled_prompt_embeds,
                     text_ids, latent_image_ids, guidance):
        return z * 0.5 + 0.1

    def _decode_packed(self, x0_hat, height, width):
        B = x0_hat.shape[0]
        return x0_hat.view(B, -1)[:, :12].view(B, 3, 2, 2)
        
    _compute_grad_k1 = GuidedFluxPipeline._compute_grad_k1
    _compute_grad_second_order = GuidedFluxPipeline._compute_grad_second_order

def mock_reward_fn(image, return_features=False):
    z = image.mean(dim=(2, 3))
    def head_fn(z_in):
        return -(z_in ** 2).sum(dim=-1)
    
    score = head_fn(z)
    if return_features:
        return score, z, head_fn
    return score
mock_reward_fn.supports_features = True

def test_integration():
    torch.manual_seed(42)
    # Ensure float64 for exactness
    torch.set_default_dtype(torch.float64)
    pipe = DummyPipe()
    
    latents = torch.randn(1, 16, 2, 2, dtype=torch.float64)
    sigma_value = 0.5
    sigma_next_value = 0.4
    snr_factor = 5.0
    reward_scale = 2.0
    gradient_norm_scale = None
    height = 2
    width = 2
    prompt_embeds = torch.randn(1, 4, dtype=torch.float64)
    pooled_prompt_embeds = torch.randn(1, 4, dtype=torch.float64)
    text_ids = torch.randn(1, 4, dtype=torch.float64)
    latent_image_ids = torch.randn(1, 4, dtype=torch.float64)
    guidance = torch.randn(1, dtype=torch.float64)
    
    for lam in [1.0, 0.1, 0.01, 0.001, 0.0001, 1e-5]:
        generator = torch.Generator().manual_seed(123)
        g_1st, r_val1, raw_norm1 = pipe._compute_grad_k1(
            latents, sigma_value, sigma_next_value, snr_factor, reward_scale,
            gradient_norm_scale, height, width, mock_reward_fn,
            prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids,
            guidance, generator=generator
        )
        
        generator = torch.Generator().manual_seed(123)
        g_2nd, r_val2, raw_norm2 = pipe._compute_grad_second_order(
            latents, sigma_value, sigma_next_value, snr_factor, reward_scale,
            gradient_norm_scale, height, width, mock_reward_fn,
            prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids,
            guidance, generator=generator, lam=lam, k_eig=16
        )
        
        scaled_g_1st = g_1st * lam
        diff = torch.norm(g_2nd - scaled_g_1st).item()
        base = torch.norm(scaled_g_1st).item()
        rel_diff = diff / base
        const = rel_diff / lam
        print(f"lam={lam:7.5f} | rel_err={rel_diff:.2e} | rel_err/lam={const:.4f}")

if __name__ == "__main__":
    test_integration()
