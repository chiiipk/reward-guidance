import torch
import time
from pipeline import GuidedFluxPipeline

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading pipeline on {device}...")
    pipe = GuidedFluxPipeline.from_flowmap_pretrained(
        model_id="black-forest-labs/FLUX.1-dev",
        device=device
    )
    
    # dummy latents
    latents = torch.randn(1, 16, 64, 64, device=device, dtype=torch.bfloat16)
    sigma = torch.tensor([0.9], device=device, dtype=torch.bfloat16)
    prompt_embeds = torch.randn(1, 77, 4096, device=device, dtype=torch.bfloat16)
    pooled = torch.randn(1, 768, device=device, dtype=torch.bfloat16)
    text_ids = torch.randn(77, 3, device=device, dtype=torch.bfloat16)
    img_ids = torch.randn(4096, 3, device=device, dtype=torch.bfloat16)
    
    print("Running forward pass...")
    latents.requires_grad_(True)
    t0 = time.time()
    x0_hat = pipe._flow_map_x0(
        latents, sigma, prompt_embeds, pooled, text_ids, img_ids, None
    )
    
    # fake reward (e.g. MSE on pixels)
    # decode to image
    image = pipe._decode_packed(x0_hat, 512, 512)
    # fake VLM feature
    feature = image.mean(dim=(2,3)) # 1 x 3
    t1 = time.time()
    print(f"Forward pass took {t1 - t0:.3f}s")
    
    # test 1 VJP
    v = torch.randn_like(feature)
    t0 = time.time()
    grad = torch.autograd.grad(feature, latents, grad_outputs=v, retain_graph=True)[0]
    t1 = time.time()
    print(f"1 VJP took {t1 - t0:.3f}s")
    
    # test 16 VJPs
    t0 = time.time()
    for _ in range(16):
        v = torch.randn_like(feature)
        grad = torch.autograd.grad(feature, latents, grad_outputs=v, retain_graph=True)[0]
    t1 = time.time()
    print(f"16 VJPs sequentially took {t1 - t0:.3f}s")
    
    # test vectorized VJP
    # we can use torch.autograd.functional.vjp if we encapsulate the function
    # but for now sequential is easy
    
if __name__ == "__main__":
    main()
