with open("benchmark_vjp.py", "r") as f:
    code = f.read()

# Replace dummy feature with imagereward
new_feature = """
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
"""

code = code.replace(
    "    # 3. Fake feature extraction (e.g. pooling)\n    feature = image.mean(dim=(2,3)).flatten() # 1 x 3 -> 3\n    t1 = time.time()\n    print(f\"Forward pass (Transformer + VAE + Feature) took {t1 - t0:.3f}s\")\n    \n    print(f\"Running {args.k} VJPs sequentially...\")",
    new_feature
)

# Add memory printing
mem_print = """
    if device == "cuda":
        print(f"Max GPU Memory allocated: {torch.cuda.max_memory_allocated()/1e9:.3f} GB")
"""
code = code.replace("    total_vjp_time = sum(vjp_times)", mem_print + "    total_vjp_time = sum(vjp_times)")

# Add batched VJP test
batched_test = """
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
"""
code = code.replace("    print(f\"Cost ratio ({args.k} VJPs / Forward pass): {total_vjp_time / (t1 - t0):.2f}x\")", "    print(f\"Cost ratio ({args.k} VJPs / Forward pass): {total_vjp_time / (t1 - t0):.2f}x\")\n" + batched_test)


with open("benchmark_vjp.py", "w") as f:
    f.write(code)

print("Updated benchmark_vjp.py")
