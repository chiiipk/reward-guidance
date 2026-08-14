"""CLI driver for reward-guided FLUX sampling.

Run from this directory. Examples:

    python sample.py --reward blueness --gradient-norm-scale 10 --num-images 4
    python sample.py --reward imagereward --prompt "..." --gradient-norm-scale 10
    python sample.py --reward skywork --prompt "a stop sign that says GO" \\
        --skywork-question "Does this image show a stop sign with the word GO on it?"
"""

import argparse
import os

import numpy as np

# Triton invokes GCC at runtime to build its CUDA driver helper. Configure the
# compiler and dynamic-loader paths before importing PyTorch/Triton so Linux
# containers can find the host-mounted libcuda.so.1.
from h200_runtime import configure_linux_libcuda_path, validate_h200_runtime

configure_linux_libcuda_path()

import torch

validate_h200_runtime(torch)

from pipeline import GuidedFluxPipeline
from rewards import (
    DEFAULT_PROMPT,
    PALETTES,
    REWARDS,
    get_imagereward_text_input,
    load_imagereward,
    load_skywork_vlm,
    make_masked_brightness_reward_fn,
    make_masked_reward_fn,
    make_palette_reward_fn,
    prepare_vlm_static_inputs,
    reward_imagereward,
    reward_skywork_vh,
)

PIXEL_REWARDS = set(REWARDS.keys())


def _write_metadata(args, cache_dir: str):
    """Write a human-readable metadata.txt summarizing this run's config."""
    lines = []
    lines.append(f"prompt:                 {args.prompt!r}")
    lines.append(f"reward:                 {args.reward}")
    lines.append(f"method:                 {args.method}")
    lines.append(f"reward_scale:           {args.reward_scale}")
    if args.grad_divisor is not None:
        lines.append(
            f"grad_divisor:           {args.grad_divisor:g}  "
            f"(rescaled grad = raw_grad / divisor)"
        )
    elif args.gradient_norm_scale is not None:
        lines.append(
            f"gradient_norm_scale:    {args.gradient_norm_scale}  "
            f"(rescaled grad has fixed L2 = gradient_norm_scale)"
        )
    else:
        lines.append("gradient_norm_scale:    disabled  (using raw gradient)")
    if args.sigma_damp is not None:
        lines.append(
            f"sigma_damp:             {args.sigma_damp:g}  "
            f"(λ_t = λ/(1 + 2λ σ²_{{1|t}}); heavier damping near noise)"
        )
    if args.num_particles > 1:
        lines.append(
            f"num_particles:          {args.num_particles}  "
            f"(plug-in k; ∇log ĥ ≈ Σ softmax(λ r_i) ∇r_i)"
        )
    if args.num_particles > 1 or args.method == "second_order":
        lines.append(
            f"lam:                    {args.lam:g}  "
            f"(particle temperature / second-order tilt scale)"
        )
    lines.append(f"snr_factor:             {args.snr_factor}")
    lines.append(f"num_guidance_steps:     {args.num_guidance_steps}")
    lines.append(f"guidance_start_step:    {args.guidance_start_step}")
    lines.append(
        f"max_abs_b:              {args.max_abs_b}  " f"(cap on b_t = sigma/(1-sigma))"
    )
    lines.append(f"num_images:             {args.num_images}")
    lines.append(f"num_inference_steps:    {args.num_steps}")
    lines.append(f"image_size:             {args.height}x{args.width}")
    lines.append(
        f"cfg_scale:              {args.cfg_scale}  "
        f"(FLUX guidance-distillation scalar)"
    )
    lines.append(f"seed:                   {args.seed}")
    lines.append(f"model_id:               {args.model_id}")

    if args.reward == "imagereward":
        ir_prompt = args.ir_prompt or args.prompt
        lines.append(
            f"ir_prompt (scoring):    {ir_prompt!r}  "
            f"(text the BLIP scorer is asked to align with)"
        )
    elif args.reward == "skywork":
        lines.append(f"skywork_model_id:       {args.skywork_model_id}")
        lines.append(
            f"skywork_question:       {args.skywork_question!r}  "
            f"(yes/no question fed to the VLM)"
        )
    elif args.reward in ("masked", "masked_brightness"):
        lines.append(f"mask_region:            {args.mask_region}")
    elif args.reward == "palette":
        lines.append(f"palette:                {args.palette}")

    with open(os.path.join(cache_dir, "metadata.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def build_reward_fn(args):
    """Return a callable reward_fn(image_tensor) -> [B] tensor of scores."""
    if args.reward in PIXEL_REWARDS:
        return REWARDS[args.reward]
    if args.reward == "masked":
        return make_masked_reward_fn(
            args.height, args.width, args.mask_region, args.device
        )
    if args.reward == "masked_brightness":
        return make_masked_brightness_reward_fn(
            args.height, args.width, args.mask_region, args.device
        )
    if args.reward == "palette":
        if "," in args.palette:
            rgb = [float(x) for x in args.palette.split(",")]
            if len(rgb) != 3 or any(x < 0.0 or x > 1.0 for x in rgb):
                raise ValueError(
                    "--palette RGB form must contain exactly three values in [0, 1]."
                )
            return make_palette_reward_fn(rgb)
        return make_palette_reward_fn(args.palette)
    if args.reward == "imagereward":
        ir_model = load_imagereward(device=args.device)
        text_input = get_imagereward_text_input(
            ir_model, args.ir_prompt or args.prompt, args.device
        )

        def _ir_fn(image, return_features=False):
            return reward_imagereward(
                image, text_input, ir_model, return_features=return_features
            )

        _ir_fn.supports_features = True
        return _ir_fn
    if args.reward == "skywork":
        if not args.skywork_question:
            raise ValueError("--skywork-question is required when --reward skywork")
        model, processor = load_skywork_vlm(
            model_id=args.skywork_model_id,
            device=args.device,
            torch_dtype=torch.bfloat16,
        )
        static = prepare_vlm_static_inputs(
            processor,
            args.skywork_question,
            args.height,
            args.width,
            args.device,
        )
        return lambda image: reward_skywork_vh(image, model, static)
    raise ValueError(f"Unknown reward '{args.reward}'.")


def validate_args(args):
    """Reject invalid combinations before loading multi-GB model weights."""
    if args.num_images < 1 or args.num_steps < 1:
        raise ValueError("--num-images and --num-steps must be positive.")
    if args.height % 16 or args.width % 16:
        raise ValueError("FLUX image height and width must be divisible by 16.")
    if args.snr_factor <= 0.0:
        raise ValueError("--snr-factor must be positive.")
    if args.grad_divisor is not None and args.grad_divisor <= 0.0:
        raise ValueError("--grad-divisor must be positive.")
    if args.gradient_norm_scale is not None and args.gradient_norm_scale < 0.0:
        raise ValueError("--gradient-norm-scale must be non-negative.")
    if args.num_particles < 1:
        raise ValueError("--num-particles must be at least 1.")
    if args.num_particles > 1 and args.gradient_norm_scale is None:
        raise ValueError(
            "--num-particles > 1 requires a positive --gradient-norm-scale."
        )
    if args.method == "second_order":
        if args.reward not in {"imagereward", "palette"}:
            raise ValueError(
                "--method second_order currently supports only "
                "--reward imagereward or --reward palette."
            )
        if args.num_particles != 1:
            raise ValueError(
                "--method second_order currently requires --num-particles 1."
            )
        if args.lam <= 0.0:
            raise ValueError("--lam must be positive for second-order guidance.")


def main(args):
    # Backward-compatible CLI convention: zero disables L2 normalization.
    # Normalize it here as well as in the command-line entry point so callers
    # invoking main(args) programmatically get identical semantics.
    if args.gradient_norm_scale == 0.0:
        args.gradient_norm_scale = None
    validate_args(args)
    torch.manual_seed(args.seed)

    print(f"Loading FLUX-1-dev from {args.model_id} + Flow Map LoRA ...")
    pipe = GuidedFluxPipeline.from_flowmap_pretrained(
        model_id=args.model_id,
        torch_dtype=torch.bfloat16,
        device=args.device,
    )
    print("Pipeline loaded.")

    print(f"Building reward fn ({args.reward}) ...")
    reward_fn = build_reward_fn(args)
    print("Reward fn ready.")

    if args.reward_scale == 0.0:
        cache_dir = os.path.join(args.output_dir, "unguided")
    elif args.grad_divisor is not None:
        cache_dir = os.path.join(
            args.output_dir,
            f"guided_{args.reward}_div{args.grad_divisor:g}"
            f"_snr{args.snr_factor}_steps{args.num_guidance_steps}"
            f"_start{args.guidance_start_step}",
        )
    elif args.gradient_norm_scale is not None:
        cache_dir = os.path.join(
            args.output_dir,
            f"guided_{args.reward}_gns{args.gradient_norm_scale}"
            f"_snr{args.snr_factor}_steps{args.num_guidance_steps}"
            f"_start{args.guidance_start_step}",
        )
    else:
        cache_dir = os.path.join(
            args.output_dir,
            f"guided_{args.reward}_raw"
            f"_snr{args.snr_factor}_steps{args.num_guidance_steps}"
            f"_start{args.guidance_start_step}",
        )
    if args.method != "plugin":
        cache_dir += f"_{args.method}"
    if args.method == "second_order":
        cache_dir += f"_lam{args.lam:g}"
    if args.sigma_damp is not None:
        cache_dir += f"_damp{args.sigma_damp:g}"
    if args.num_particles > 1:
        cache_dir += f"_k{args.num_particles}_lam{args.lam:g}"
    os.makedirs(cache_dir, exist_ok=True)
    _write_metadata(args, cache_dir)

    rewards = []
    for i in range(args.num_images):
        gen = torch.Generator(device=args.device).manual_seed(args.seed + i)
        out = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_steps,
            guidance_scale=args.cfg_scale,
            generator=gen,
            reward_fn=reward_fn,
            reward_scale=args.reward_scale,
            gradient_norm_scale=args.gradient_norm_scale,
            snr_factor=args.snr_factor,
            num_guidance_steps=args.num_guidance_steps,
            guidance_start_step=args.guidance_start_step,
            max_abs_b=args.max_abs_b,
            grad_divisor=args.grad_divisor,
            sigma_damp=args.sigma_damp,
            num_particles=args.num_particles,
            lam=args.lam,
            method=args.method,
            verbose=args.verbose,
        )
        img = out.images[0]
        img.save(os.path.join(cache_dir, f"{i:04d}.png"))

        # Score with the same reward function (no autograd here).
        with torch.no_grad():
            arr = np.array(img).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(args.device)
            r = float(reward_fn(t).mean().item())
        rewards.append(r)
        print(f"image {i+1}/{args.num_images}: reward = {r:+.4f}")

    np.save(os.path.join(cache_dir, "rewards.npy"), np.array(rewards))
    print(f"Saved {len(rewards)} images and rewards to {cache_dir}")
    print(f"mean reward: {np.mean(rewards):+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reward-guided FLUX sampling")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--reward",
        type=str,
        default="blueness",
        choices=list(REWARDS)
        + ["imagereward", "skywork", "masked", "masked_brightness", "palette"],
    )
    parser.add_argument(
        "--mask-region",
        type=str,
        default="topright_circle",
        choices=["topright_circle", "center_circle", "bottom_half"],
        help="Named mask region for --reward masked.",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default="cool_ocean",
        help=(
            "Named palette for --reward palette. "
            f"Choices: {list(PALETTES)}, or pass "
            "comma-separated R,G,B in [0,1] e.g. '0.2,0.4,0.7'."
        ),
    )
    parser.add_argument(
        "--ir-prompt",
        type=str,
        default=None,
        help="Override prompt used by ImageReward scorer "
        "(default: same as --prompt).",
    )
    parser.add_argument(
        "--skywork-question",
        type=str,
        default=None,
        help="Question fed to Skywork-VL-Reward-7B, e.g. "
        "'Is this image a stop sign with the word GO "
        "clearly written on it? Answer Yes or No.'",
    )
    parser.add_argument(
        "--skywork-model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument(
        "--gradient-norm-scale",
        type=float,
        default=10.0,
        help="Positive target L2 norm; pass 0 to keep the raw gradient.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="plugin",
        choices=["plugin", "second_order"],
        help="Guidance method to use.",
    )
    parser.add_argument(
        "--snr-factor",
        type=float,
        default=5.0,
        help="Renoising SNR factor (larger → smaller σ' shift).",
    )
    parser.add_argument("--num-guidance-steps", type=int, default=5)
    parser.add_argument("--guidance-start-step", type=int, default=1)
    parser.add_argument(
        "--max-abs-b", type=float, default=20.0, help="Cap on b_t = σ/(1−σ)."
    )
    parser.add_argument(
        "--grad-divisor",
        type=float,
        default=None,
        help="If set, divide gradient by this constant instead "
        "of normalizing to gradient_norm_scale L2.",
    )
    parser.add_argument(
        "--sigma-damp",
        type=float,
        default=None,
        help="Damping schedule sigma. λ_t = λ/(1 + 2λ σ²_{1|t}) "
        "with σ²_{1|t} = σ²(1−t)²/((1−t)² + t²σ²). "
        "Heavy damping near noise (high σ_t), no damping "
        "near data (low σ_t).",
    )
    parser.add_argument(
        "--num-particles",
        type=int,
        default=1,
        help="Number of plug-in particles k. With k>1, "
        "guidance becomes Σ softmax(λ r_i) ∇r_i where each "
        "∇r_i is L2-normalized to gradient_norm_scale.",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help=(
            "Softmax temperature for k-particle aggregation; "
            "reward tilt scale for second-order guidance."
        ),
    )
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--cfg-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
