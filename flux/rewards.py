"""Differentiable reward functions for FLUX guidance.

Pixel rewards: blueness, blue_minus_rg, blue_minus_half_rg.
Learned rewards: ImageReward (BLIP-based scorer made differentiable),
                 Skywork-VL (VLM yes/no probe made differentiable).
"""

from collections.abc import Callable
from typing import Union

import torch
import torch.nn.functional as F

DEFAULT_PROMPT = "a baby fox wearing a cozy knitted sweater"

# BLIP preprocessing constants (also used by Skywork-VL because it inherits
# the OpenAI/CLIP image normalization).
_BLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_BLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# ---------------------------------------------------------------------------
# Pixel rewards
# ---------------------------------------------------------------------------


def reward_blueness(image: torch.Tensor) -> torch.Tensor:
    """Mean blue channel; image in [0, 1] with shape [B, 3, H, W]. Returns [B]."""
    return image[:, 2].mean(dim=(1, 2))


def reward_blue_minus_rg(image: torch.Tensor) -> torch.Tensor:
    """Mean of B − R − G. Rewards blue dominance, not just brightness."""
    return (image[:, 2] - image[:, 0] - image[:, 1]).mean(dim=(1, 2))


def reward_blue_minus_half_rg(image: torch.Tensor) -> torch.Tensor:
    """Mean of B − ½(R + G). Diamond Maps' blueness reward."""
    return (image[:, 2] - 0.5 * (image[:, 0] + image[:, 1])).mean(dim=(1, 2))


# ---------------------------------------------------------------------------
# Geometric / structural rewards (FMTT-style)
# ---------------------------------------------------------------------------


def make_circle_mask(
    h: int, w: int, cx: int, cy: int, r: int, device: str = "cuda"
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return ((yy - cy) ** 2 + (xx - cx) ** 2 < r**2).float().unsqueeze(0)


def reward_masked(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Saliency inside mask − saliency outside.

    Saliency at each pixel is its squared deviation from the per-image mean
    color, summed over channels — a simple proxy for "image content".
    Rewards images whose interesting content lies inside the mask region.

    image: [B, 3, H, W] in [0, 1]; mask: [1, H, W] in [0, 1].
    Returns [B].
    """
    mu = image.mean(dim=(2, 3), keepdim=True)
    saliency = ((image - mu) ** 2).sum(dim=1, keepdim=True)
    m = mask.unsqueeze(0).to(image.device).to(image.dtype)
    in_mask = (saliency * m).sum(dim=(1, 2, 3)) / (m.sum() + 1e-6)
    out_mask = (saliency * (1 - m)).sum(dim=(1, 2, 3)) / ((1 - m).sum() + 1e-6)
    return in_mask - out_mask


def reward_masked_brightness(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean intensity inside mask − mean intensity outside mask.

    Intensity is the mean over RGB channels, so r in [-1, 1].

    image: [B, 3, H, W] in [0, 1]; mask: [1, H, W] in [0, 1].
    Returns [B].
    """
    intensity = image.mean(dim=1, keepdim=True)
    m = mask.unsqueeze(0).to(image.device).to(image.dtype)
    in_mask = (intensity * m).sum(dim=(1, 2, 3)) / (m.sum() + 1e-6)
    out_mask = (intensity * (1 - m)).sum(dim=(1, 2, 3)) / ((1 - m).sum() + 1e-6)
    return in_mask - out_mask


def reward_symmetric(image: torch.Tensor) -> torch.Tensor:
    """−MSE(image, horizontal_flip(image)). Maxed at perfect mirror symmetry."""
    flipped = image.flip(dims=[-1])
    return -((image - flipped) ** 2).mean(dim=(1, 2, 3))


def reward_antisymmetric(image: torch.Tensor) -> torch.Tensor:
    """+MSE(image, horizontal_flip(image)). Maxed at maximally asymmetric."""
    flipped = image.flip(dims=[-1])
    return ((image - flipped) ** 2).mean(dim=(1, 2, 3))


def reward_rotation_invariant(image: torch.Tensor) -> torch.Tensor:
    """−MSE(image, rotated(image, 180°)). Maxed at 180°-rotation-invariant."""
    rotated = image.flip(dims=[-2, -1])
    return -((image - rotated) ** 2).mean(dim=(1, 2, 3))


def make_palette_reward_fn(target_rgb: Union[str, list[float]]) -> Callable:
    """Return a reward_fn that scores how close the image's mean RGB is to a target.

    Can optionally return `(score, z, head_fn)` if `return_features=True` is passed,
    where `z` is the Bx3 mean color and `head_fn` is the function mapping z to score.
    """
    if isinstance(target_rgb, str):
        target_rgb = PALETTES[target_rgb]

    target_tensor = torch.tensor(target_rgb, dtype=torch.float32).view(1, 3)

    def reward_fn(image: torch.Tensor, return_features: bool = False):
        z = image.mean(dim=(2, 3))

        def head_fn(z_in):
            t = target_tensor.to(z_in.device).to(z_in.dtype)
            return -((z_in - t) ** 2).sum(dim=-1)

        score = head_fn(z)
        if return_features:
            return score, z, head_fn
        return score

    reward_fn.supports_features = True
    return reward_fn


# Named palettes for convenience.
PALETTES = {
    "warm_autumn": [0.78, 0.42, 0.18],  # burnt orange
    "cool_ocean": [0.20, 0.40, 0.70],  # ocean blue
    "rose_pink": [0.90, 0.55, 0.70],  # rose
    "forest_green": [0.18, 0.50, 0.28],  # forest
    "monochrome_gray": [0.50, 0.50, 0.50],
}


# ---------------------------------------------------------------------------
# ImageReward (BLIP) — verbatim from flux_old, with batch-dim preserved
# ---------------------------------------------------------------------------


def _shim_imagereward_imports():
    """Patch Transformers/BLIP API mismatches before importing ImageReward.

    1. ImageReward imports pruning/chunking helpers from `modeling_utils`, but
       Transformers 4.57 moved them to `pytorch_utils`.
    2. BLIP's `init_tokenizer` reads
       `tokenizer.additional_special_tokens_ids[0]`, which returns an empty
       list under newer Transformers. Replace `init_tokenizer` itself with a
       version that uses `convert_tokens_to_ids('[ENC]')` directly. We patch
       both the source module (`blip`) AND the consumer module
       (`blip_pretrain`, which does `from .blip import init_tokenizer` at
       import time and would otherwise keep the old reference).
    """
    import transformers.modeling_utils as tmu
    import transformers.pytorch_utils as tpu

    if not hasattr(tpu, "find_pruneable_heads_and_indices"):

        def _stub(heads, n_heads, head_size, already_pruned_heads):
            return set(), torch.tensor([], dtype=torch.long)

        tpu.find_pruneable_heads_and_indices = _stub

    for helper in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(tmu, helper) and hasattr(tpu, helper):
            setattr(tmu, helper, getattr(tpu, helper))

    import ImageReward.models.BLIP.blip as _b
    import ImageReward.models.BLIP.blip_pretrain as _bp
    import ImageReward.models.BLIP.med as _med
    from transformers import BertTokenizer

    def _patched_init_tokenizer():
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
        return tokenizer

    _b.init_tokenizer = _patched_init_tokenizer
    _bp.init_tokenizer = _patched_init_tokenizer

    # Newer transformers' init_weights calls tie_weights() which reads
    # `all_tied_weights_keys` (a 5.x property) that BLIP's custom BertModel
    # doesn't expose. BLIP doesn't tie weights anyway, so make tie_weights
    # a no-op on its custom BertModel/BertLMHeadModel.
    def _noop_tie_weights(self, *args, **kwargs):
        pass

    # `get_head_mask` was a method on PreTrainedModel in 4.x; in 5.x it was
    # moved to a module-level utility, so BLIP's `self.get_head_mask(...)`
    # raises AttributeError. Provide a minimal stub that returns
    # `[None] * num_hidden_layers` when head_mask is None (the typical case),
    # otherwise broadcasts a single mask across all layers.
    def _get_head_mask(self, head_mask, num_hidden_layers, *args, **kwargs):
        if head_mask is None:
            return [None] * num_hidden_layers
        if isinstance(head_mask, (list, tuple)):
            return list(head_mask)
        return [head_mask] * num_hidden_layers

    for cls_name in (
        "BertModel",
        "BertLMHeadModel",
        "BertEncoder",
        "BertLayer",
        "BertAttention",
    ):
        cls = getattr(_med, cls_name, None)
        if cls is not None:
            cls.tie_weights = _noop_tie_weights
            cls.get_head_mask = _get_head_mask


def load_imagereward(device: str = "cuda"):
    """Load ImageReward-v1.0; exposes .blip and .mlp for differentiable scoring."""
    _shim_imagereward_imports()
    import ImageReward as RM

    ir_model = RM.load("ImageReward-v1.0", device=device)
    ir_model.eval()
    # We only need gradients with respect to the input image/features.  Freezing
    # model parameters avoids allocating parameter-gradient buffers on the GPU.
    for parameter in ir_model.parameters():
        parameter.requires_grad_(False)
    return ir_model


def get_imagereward_text_input(ir_model, prompt: str, device: str):
    """Tokenize prompt for BLIP; call once and reuse across denoising steps."""
    return ir_model.blip.tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=35,
        return_tensors="pt",
    ).to(device)


def reward_imagereward(
    image: torch.Tensor, text_input, ir_model, return_features: bool = False
):
    """Score images with ImageReward while keeping gradients.

    image_tensor → resize/normalize → blip.visual_encoder → blip.text_encoder
    (cross-attn on image_embeds) → mlp → scalar per batch element.

    If return_features=True, returns (score, z_feat, head_fn) where:
      - z_feat: [B, 768] CLS token embedding from BLIP text_encoder
      - head_fn: callable z -> scalar, i.e. ir_model.mlp
    This enables second-order guidance via Woodbury on the MLP head's Hessian.
    """
    img = F.interpolate(
        image, size=(224, 224), mode="bicubic", align_corners=False, antialias=True
    )
    mean = torch.tensor(_BLIP_MEAN, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_BLIP_STD, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    img = (img - mean) / std

    image_embeds = ir_model.blip.visual_encoder(img)
    image_atts = torch.ones(
        image_embeds.shape[:-1], dtype=torch.long, device=img.device
    )

    # BLIP text_encoder expects token-id batch matching image batch; tile if needed.
    B = img.shape[0]
    if text_input.input_ids.shape[0] != B:
        input_ids = text_input.input_ids.expand(B, -1)
        attention_mask = text_input.attention_mask.expand(B, -1)
    else:
        input_ids = text_input.input_ids
        attention_mask = text_input.attention_mask

    text_output = ir_model.blip.text_encoder(
        input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )
    z_feat = text_output.last_hidden_state[:, 0, :]  # [B, 768]
    score = ir_model.mlp(z_feat).squeeze(-1)

    if return_features:

        def head_fn(z_in):
            return ir_model.mlp(z_in).squeeze(-1)

        return score, z_feat, head_fn
    return score


# ---------------------------------------------------------------------------
# Skywork VLM — log-probability of "yes" answer to a yes/no question
# ---------------------------------------------------------------------------


def load_skywork_vlm(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load Qwen2.5-VL-3B-Instruct as the VLM reward backbone.

    Skywork-VL-Reward-7B + FLUX exceeds 44 GB GPU memory; we fall back to
    Qwen2.5-VL-3B-Instruct (the same family the flux_old experiments used)
    and score with sigmoid(logit("Yes") − logit("No")) at the assistant
    next-token position. No value head, no trl, no gating.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def prepare_vlm_static_inputs(
    processor, question: str, image_h: int, image_w: int, device: str
):
    """Pre-compute text tokenization + image-grid info once before sampling.

    Mirrors flux_old/flux_mode_selection/rewards.prepare_vlm_inputs:
    the prompt asks a yes/no question and the next-token logit at the
    assistant position is used to compute the reward.
    """
    import numpy as np
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.fromarray(
                        np.zeros((image_h, image_w, 3), dtype=np.uint8)
                    ),
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt")

    ip = processor.image_processor
    yes_ids = processor.tokenizer("Yes", add_special_tokens=False).input_ids
    no_ids = processor.tokenizer("No", add_special_tokens=False).input_ids
    thw = inputs["image_grid_thw"][0]
    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "image_grid_thw": inputs["image_grid_thw"].to(device),
        "grid_t": int(thw[0]),
        "grid_h": int(thw[1]),
        "grid_w": int(thw[2]),
        "merge_size": ip.merge_size,
        "img_mean": torch.tensor(ip.image_mean, dtype=torch.float32),
        "img_std": torch.tensor(ip.image_std, dtype=torch.float32),
        "yes_token_id": yes_ids[0],
        "no_token_id": no_ids[0],
    }


def reward_skywork_vh(image: torch.Tensor, model, static) -> torch.Tensor:
    """VLM yes/no reward = logit("Yes") − logit("No"), differentiable through image.

    (Name retained for backwards compatibility with sample.py; despite the
    name, this is the Qwen2.5-VL yes/no-logit recipe from flux_old, not the
    Skywork value-head scorer.)
    """
    device = image.device
    patch_size = 14
    grid_t = static["grid_t"]
    grid_h = static["grid_h"]
    grid_w = static["grid_w"]
    merge = static["merge_size"]
    n_patches = grid_t * grid_h * grid_w
    target_h = grid_h * patch_size
    target_w = grid_w * patch_size

    img = F.interpolate(
        image,
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    mean = static["img_mean"].to(device).view(1, 3, 1, 1)
    std = static["img_std"].to(device).view(1, 3, 1, 1)
    img = (img - mean) / std

    img = img[0]
    frames = img.unsqueeze(0).expand(2, -1, -1, -1)
    patches = frames.reshape(
        grid_t,
        2,
        3,
        grid_h // merge,
        merge,
        patch_size,
        grid_w // merge,
        merge,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
    pixel_values = patches.reshape(n_patches, 3 * 2 * patch_size * patch_size)

    out = model(
        input_ids=static["input_ids"],
        attention_mask=static["attention_mask"],
        pixel_values=pixel_values.to(dtype=next(model.parameters()).dtype),
        image_grid_thw=static["image_grid_thw"],
        return_dict=True,
        use_cache=False,
    )
    last_logits = out.logits[0, -1, :].float()
    return (
        last_logits[static["yes_token_id"]] - last_logits[static["no_token_id"]]
    ).unsqueeze(0)


REWARDS: dict[str, Callable] = {
    "blueness": reward_blueness,
    "blue_minus_rg": reward_blue_minus_rg,
    "blue_minus_half_rg": reward_blue_minus_half_rg,
    "symmetric": reward_symmetric,
    "antisymmetric": reward_antisymmetric,
    "rotation_invariant": reward_rotation_invariant,
}


def make_masked_reward_fn(image_h: int, image_w: int, region: str, device: str):
    """Build a reward_fn(image) closure for a named geometric mask region."""
    mask = _make_named_mask(image_h, image_w, region, device)
    return lambda image: reward_masked(image, mask)


def _make_named_mask(image_h: int, image_w: int, region: str, device: str):
    """Create one of the masks exposed by the FLUX CLI."""
    if region == "topright_circle":
        return make_circle_mask(
            image_h,
            image_w,
            cx=int(0.75 * image_w),
            cy=int(0.25 * image_h),
            r=int(0.18 * min(image_h, image_w)),
            device=device,
        )
    if region == "center_circle":
        return make_circle_mask(
            image_h,
            image_w,
            cx=image_w // 2,
            cy=image_h // 2,
            r=int(0.20 * min(image_h, image_w)),
            device=device,
        )
    if region == "bottom_half":
        mask = torch.zeros(1, image_h, image_w, device=device)
        mask[:, image_h // 2 :, :] = 1.0
        return mask
    raise ValueError(f"unknown geometric region: {region}")


def make_masked_brightness_reward_fn(
    image_h: int, image_w: int, region: str, device: str
):
    """Build a reward_fn(image) closure for masked-brightness on a region."""
    mask = _make_named_mask(image_h, image_w, region, device)
    return lambda image: reward_masked_brightness(image, mask)
