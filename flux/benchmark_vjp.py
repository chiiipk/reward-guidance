import torch
import time
import argparse
from pipeline import GuidedFluxPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16, help="Number of VJPs to run")
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.1-dev")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading pipeline on {device}...")
    pipe = GuidedFluxPipeline.from_flowmap_pretrained(
        model_id=args.model_id,
        device=device,
        torch_dtype=torch.bfloat16
    )
    
    # Dummy latents and conditionings
    batch_size = 1
    latents = torch.randn(batch_size, 4096, 64, device=device, dtype=torch.bfloat16)
    sigma = torch.tensor([0.9], device=device, dtype=torch.bfloat16)
    prompt_embeds = torch.randn(batch_size, 512, 4096, device=device, dtype=torch.bfloat16)
    pooled = torch.randn(batch_size, 768, device=device, dtype=torch.bfloat16)
    text_ids = torch.randn(512, 3, device=device, dtype=torch.bfloat16)
    img_ids = torch.randn(4096, 3, device=device, dtype=torch.bfloat16)
    
    print("Running forward pass...")
    latents.requires_grad_(True)
    t0 = time.time()
    
    # 1. Forward through transformer
    x0_hat = pipe._flow_map_x0(
        latents, sigma, prompt_embeds, pooled, text_ids, img_ids, None
    )
    
    # 2. Decode to image
    image = pipe._decode_packed(x0_hat, 512, 512)
    

    # 3. Load ImageReward and extract true feature (d=768)
    # We will just instantiate the pipeline's reward_fn to get the real feature size if it's available
    from rewards import load_imagereward, get_imagereward_text_input, reward_imagereward
    
    print("Loading ImageReward model for benchmarking...")
    # Mocking ImageReward to not require the real model download, but wait!
    # The user is running this on H200 where the models are cached.
    # We can just load it.
    try:
        ir_model = load_imagereward(device)
        text_input = get_imagereward_text_input(ir_model, "a cute fluffy cat", device)
        
        # We need to simulate the forward pass through ImageReward to get the feature
        # image_tensor -> resize/normalize -> blip.visual_encoder -> blip.text_encoder -> feature (d=768)
        import torch.nn.functional as F
        img = F.interpolate(image, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        # Just pass it through IR visually
        image_embeds = ir_model.blip.visual_encoder(img)
        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=img.device)
        
        text_output = ir_model.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        feature = text_output.last_hidden_state[:, 0, :] # 1 x 768
        
        def head_fn(feat):
            return ir_model.mlp(feat).squeeze(-1)
            
    except Exception as e:
        print("Could not load ImageReward:", e)
        print("Falling back to dummy d=768 feature for benchmarking.")
        feature = torch.randn(1, 768, device=device, dtype=image.dtype)
        feature = feature * image.mean() # Connect it to the graph
        def head_fn(feat):
            return (feat ** 2).sum()

    feature = feature.squeeze(0) # d=768
    
    t1 = time.time()
    print(f"Forward pass (Transformer + VAE + Reward Feature) took {t1 - t0:.3f}s")
    
    # 4. Measure Hessian of the head
    print("Measuring Hessian of the head (d=768)...")
    t_hess_start = time.time()
    
    def head_1d(z1d):
        return head_fn(z1d.unsqueeze(0)).squeeze(0)
    
    try:
        from torch.func import hessian
        H_g = hessian(head_1d)(feature)
    except Exception as e:
        print("torch.func.hessian failed:", e)
        H_g = torch.autograd.functional.hessian(head_1d, feature)
        
    torch.cuda.synchronize() if device == "cuda" else None
    t_hess_end = time.time()
    print(f"Hessian of the head took {t_hess_end - t_hess_start:.3f}s")
    
    print(f"Running {args.k} VJPs sequentially...")

    vjp_times = []
    
    # Pre-generate random vectors in feature space
    vs = [torch.randn_like(feature) for _ in range(args.k)]
    
    # We use retain_graph=True so we can call grad multiple times
    for i, v in enumerate(vs):
        torch.cuda.synchronize() if device == "cuda" else None
        t_start = time.time()
        
        grad = torch.autograd.grad(
            outputs=feature, 
            inputs=latents, 
            grad_outputs=v, 
            retain_graph=True
        )[0]
        
        torch.cuda.synchronize() if device == "cuda" else None
        t_end = time.time()
        vjp_times.append(t_end - t_start)
    

    if device == "cuda":
        print(f"Max GPU Memory allocated: {torch.cuda.max_memory_allocated()/1e9:.3f} GB")
    total_vjp_time = sum(vjp_times)
    print(f"Total time for {args.k} VJPs: {total_vjp_time:.3f}s")
    print(f"Average time per VJP: {total_vjp_time / args.k:.3f}s")
    print(f"Cost ratio ({args.k} VJPs / Forward pass): {total_vjp_time / (t1 - t0):.2f}x")

    # Test batched VJP
    try:
        print(f"Testing batched VJP (is_grads_batched=True)...")
        # Stack vs into a single tensor of shape (k, d)
        vs_tensor = torch.stack(vs, dim=0) # k x d
        
        t_batch_start = time.time()
        grad_batched = torch.autograd.grad(
            outputs=feature,
            inputs=latents,
            grad_outputs=vs_tensor,
            retain_graph=True,
            is_grads_batched=True
        )[0]
        torch.cuda.synchronize() if device == "cuda" else None
        t_batch_end = time.time()
        print(f"Batched VJP took {t_batch_end - t_batch_start:.3f}s")
    except Exception as e:
        print("Batched VJP failed:", e)

    
if __name__ == "__main__":
    main()
