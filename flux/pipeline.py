"""Reward-guided FLUX-1-dev pipeline (k=1 plug-in, Diamond-Maps style).

Implements a single-particle reward-guided FluxPipeline that follows the structure
of Holderrieth et al.'s Diamond Maps repository
(https://github.com/PeterHolderrieth/diamond_maps), simplified to:

    - k=1 single particle (no IS weighting)
    - No likelihood correction (`include_likelihood=False`)
    - No score correction (`include_score=False`)
    - Engineering tricks only: gradient normalization, b_t cap, guidance window.

REQUIRES the Flow Map LoRA (`gabeguofanclub/flux-1-dev-flowmap-lsd`) +
DualTimeEmbedder patch. Without these, the dual-time x_0 prediction is
undefined and guidance is too noisy to converge.

At each guided step it
(1) renoises x_t to a higher noise level x_{t'} via a Gaussian forward kernel,
(2) calls the flow-map model to predict x_0 from x_{t'} in one shot
    (model takes dual timestep `[σ', 0]`),
(3) computes the gradient of reward(decode(x_0)) w.r.t. x_t,
(4) normalizes the gradient to fixed L2 magnitude,
(5) modifies the velocity prediction by `noise_pred -= b_t * grad`
    where `b_t = σ/(1−σ)` capped at `max_abs_b`,
(6) runs the standard scheduler.step(noise_pred, t, latents).

See README.md in this folder for the full list of approximations vs. theory.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import List, Optional, Union

import numpy as np
import torch

from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipeline,
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

from dual_time_embedder import add_dual_time_embedder


DEFAULT_FLUX_LORA_SOURCE = "gabeguofanclub/flux-1-dev-flowmap-lsd"
DEFAULT_FLUX_LORA_WEIGHT_NAME = (
    "01-12-26/runs/res_512_steps_50k_rank_64_lr_1e-4/checkpoint-43000/"
    "pytorch_lora_weights.safetensors"
)


def damped_lambda_factor(sigma_t: float, sigma_damp: float,
                          lambda_eff: float) -> float:
    """λ_t / λ = 1 / (1 + 2 λ σ²_{1|t}).

    σ²_{1|t} = σ²(1−t)² / ((1−t)² + t²σ²) where t = interpolant time = 1 − σ_t.
    Heavy damping near noise (σ_t → 1), no damping near data (σ_t → 0).
    """
    t_interp = 1.0 - sigma_t
    s = sigma_damp
    num = s * s * (1.0 - t_interp) ** 2
    den = (1.0 - t_interp) ** 2 + t_interp ** 2 * s * s + 1e-12
    sigma_1t_sq = num / den
    return 1.0 / (1.0 + 2.0 * lambda_eff * sigma_1t_sq)


def compute_b_coefficient(sigma: float, max_abs_b: float = 20.0,
                          eps: float = 1e-6) -> float:
    """b_t = σ/(1−σ), clamped to [0, max_abs_b].

    The Doob h-transform Euler step in FLUX's σ-time is
        x_{σ_next} = x_σ + dσ · (v_θ − b_t · ∇log h_t),    dσ < 0
    where b_t = σ/(1−σ) = ½ · σ_ml² (memoryless schedule).
    The cap matches the Diamond-Maps reference implementation.
    """
    return min(max(sigma / max(1.0 - sigma, eps), 0.0), max_abs_b)


def transformer_dtype(module: torch.nn.Module) -> torch.dtype:
    for p in module.parameters():
        return p.dtype
    return torch.float32


def vae_dtype(module: torch.nn.Module) -> torch.dtype:
    if hasattr(module, "decoder"):
        for p in module.decoder.parameters():
            return p.dtype
    for p in module.parameters():
        return p.dtype
    return torch.float32


class GuidedFluxPipeline(FluxPipeline):
    """k=1 plug-in reward-guided FLUX flow-map pipeline.

    Subclass of FluxPipeline. Use `from_flowmap_pretrained` to load with the
    dual-time-embedder patch and Flow Map LoRA applied.
    """

    @classmethod
    def from_flowmap_pretrained(
        cls,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        lora_source: str = DEFAULT_FLUX_LORA_SOURCE,
        weight_name: str = DEFAULT_FLUX_LORA_WEIGHT_NAME,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> "GuidedFluxPipeline":
        """Load FLUX-1-dev, apply dual-time-embedder patch, load Flow Map LoRA."""
        pipe = cls.from_pretrained(model_id, torch_dtype=torch_dtype)
        pipe.transformer = add_dual_time_embedder(pipe.transformer)
        pipe = pipe.to(device, torch_dtype)
        pipe.load_lora_weights(lora_source, weight_name=weight_name)
        pipe = pipe.to(device, torch_dtype)
        return pipe

    def _run_transformer_dual(
        self,
        latents_packed: torch.Tensor,
        sigma_from: torch.Tensor,
        sigma_to: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """One transformer call with dual-time input `[σ_from, σ_to]`.

        Both `sigma_from` and `sigma_to` are (B,) tensors in [0, 1]. The
        DualTimeEmbedder averages embeddings of both timesteps; the LoRA
        is fine-tuned to interpret the result as a flow-map velocity that
        integrates from σ_from to σ_to in one step.
        """
        dtype = transformer_dtype(self.transformer)
        timestep = torch.stack([sigma_from, sigma_to], dim=-1).to(dtype)
        return self.transformer(
            hidden_states=latents_packed.to(dtype),
            timestep=timestep,
            encoder_hidden_states=prompt_embeds.to(dtype),
            pooled_projections=pooled_prompt_embeds.to(dtype),
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            guidance=guidance.to(dtype) if guidance is not None else None,
            return_dict=False,
        )[0]

    def _flow_map_x0(
        self,
        latents: torch.Tensor,
        sigma_from: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """One flow-map model query at `[σ_from, 0]`; returns x0 prediction.

        For k=1 we only need x0; the noise_pred at [σ_from, σ_to] that the
        upstream `compute_model_prediction` also produces is unused (it would
        only feed the score-correction term, which we explicitly omit).
        Skipping it halves activation memory in the guided pass.
        """
        zero_t = torch.zeros_like(sigma_from)
        v0_pred = self._run_transformer_dual(
            latents, sigma_from, zero_t, prompt_embeds,
            pooled_prompt_embeds, text_ids, latent_image_ids, guidance,
        )
        sigma_view = sigma_from.view(-1, *([1] * (latents.ndim - 1)))
        return (latents - sigma_view * v0_pred).to(latents.dtype)

    def _decode_packed(self, latents_packed: torch.Tensor, height: int,
                        width: int) -> torch.Tensor:
        """Decode packed latents to image tensor in [0, 1]; differentiable."""
        unpacked = self._unpack_latents(latents_packed, height, width,
                                         self.vae_scale_factor)
        unpacked = (unpacked / self.vae.config.scaling_factor
                    + self.vae.config.shift_factor)
        decoded = self.vae.decode(unpacked.to(vae_dtype(self.vae)),
                                   return_dict=False)[0]
        return (decoded / 2 + 0.5).clamp(0, 1)

    def _compute_grad_k1(
        self,
        latents: torch.Tensor,
        sigma_value: float,
        sigma_next_value: float,
        snr_factor: float,
        reward_scale: float,
        gradient_norm_scale: float,
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
    ) -> tuple[torch.Tensor, float]:
        """k=1 plug-in: renoise → flow-map x0 → reward → grad → normalize."""
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
            reward = reward_fn(image)
            objective = (reward * reward_scale).sum()
            grad = torch.autograd.grad(objective, x_input,
                                        retain_graph=False)[0]

        raw_grad_norm = float(grad.norm().item())
        if grad_divisor is not None:
            grad = grad / grad_divisor
        else:
            grad_norm = grad.norm().clamp_min(1e-8)
            grad = grad / grad_norm * gradient_norm_scale

        return grad.detach(), float(reward.detach().mean().item()), raw_grad_norm

    def _compute_grad_k_particles(
        self,
        latents: torch.Tensor,
        sigma_value: float,
        snr_factor: float,
        reward_scale: float,
        gradient_norm_scale: float,
        height: int,
        width: int,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance: Optional[torch.Tensor],
        generator: Optional[torch.Generator],
        num_particles: int,
        lam: float,
    ) -> tuple[torch.Tensor, float, float]:
        """k-particle plug-in: ∇log ĥ ≈ Σ_i softmax(λ r_i) ∇r_i.

        Each particle i samples its own renoise ε_i, predicts x0, scores reward,
        backprops to get ∇r_i. We L2-normalize each ∇r_i to gradient_norm_scale
        before applying the softmax weights. With per-particle backprop and
        retain_graph=False the peak memory equals the k=1 case.
        """
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

        sigma_prime_t = torch.full(
            (latents.shape[0],), sigma_prime,
            device=latents.device, dtype=latents.dtype,
        )

        rewards = []
        normalized_grads = []
        raw_grad_norms = []
        for _ in range(num_particles):
            eps = torch.randn(
                latents.shape, device=latents.device, dtype=latents.dtype,
                generator=generator,
            )
            with torch.enable_grad():
                x_input = latents.detach().requires_grad_(True)
                z = scale * x_input + std_q * eps
                x0_hat = self._flow_map_x0(
                    z, sigma_prime_t, prompt_embeds,
                    pooled_prompt_embeds, text_ids, latent_image_ids, guidance,
                )
                image = self._decode_packed(x0_hat, height, width).float()
                reward = reward_fn(image)
                objective = (reward * reward_scale).sum()
                grad = torch.autograd.grad(objective, x_input,
                                            retain_graph=False)[0]

            r_scalar = float(reward.detach().mean().item())
            raw_norm = float(grad.norm().item())
            unit_grad = grad.detach() / max(raw_norm, 1e-8)
            normalized_grads.append(unit_grad * gradient_norm_scale)
            rewards.append(r_scalar)
            raw_grad_norms.append(raw_norm)

            del z, x0_hat, image, reward, objective, grad, x_input, eps
            torch.cuda.empty_cache()

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        log_w = lam * rewards_tensor
        log_w = log_w - log_w.max()  # numerical stability
        weights = torch.softmax(log_w, dim=0).to(latents.device)

        agg = torch.zeros_like(normalized_grads[0])
        for w_i, g_i in zip(weights, normalized_grads):
            agg = agg + w_i * g_i

        return (
            agg,
            float(rewards_tensor.mean().item()),
            float(sum(raw_grad_norms) / len(raw_grad_norms)),
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        output_type: str = "pil",
        # Reward-guidance kwargs ------------------------------------------------
        reward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        reward_scale: float = 0.0,
        gradient_norm_scale: float = 10.0,
        snr_factor: float = 5.0,
        num_guidance_steps: int = 5,
        guidance_start_step: int = 1,
        max_abs_b: float = 20.0,
        grad_divisor: Optional[float] = None,
        sigma_damp: Optional[float] = None,
        num_particles: int = 1,
        lam: float = 1.0,
        verbose: bool = False,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt, prompt_2=None, height=height, width=width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=["latents"],
            max_sequence_length=max_sequence_length,
        )

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        # Pin to the transformer's device. self._execution_device walks the
        # pipeline's modules and returns the first one's device — after the
        # first __call__ leaves text encoders on CPU, that property reports
        # "cpu" while the generator and transformer are still on cuda.
        device = self.transformer.device
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            self.text_encoder.to(device)
        if hasattr(self, "text_encoder_2") and self.text_encoder_2 is not None:
            self.text_encoder_2.to(device)

        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt, prompt_2=None,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device, num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length, lora_scale=None,
        )

        # Offload text encoders (CLIP + T5-XXL) to CPU after encoding — they
        # are not needed for the rest of the trajectory and free ~5 GB so the
        # VLM reward model can fit alongside FLUX.
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            self.text_encoder.to("cpu")
        if hasattr(self, "text_encoder_2") and self.text_encoder_2 is not None:
            self.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents,
            height, width, prompt_embeds.dtype, device, generator, latents,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if (hasattr(self.scheduler.config, "use_flow_sigmas")
                and self.scheduler.config.use_flow_sigmas):
            sigmas = None

        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )

        guidance_start_idx = int(guidance_start_step)
        guidance_end_idx = min(num_inference_steps,
                                guidance_start_idx + int(num_guidance_steps))
        is_guided_run = (reward_fn is not None and reward_scale != 0.0
                         and num_guidance_steps > 0)

        # Gradient checkpointing on the transformer halves activation memory
        # at the cost of one extra forward per backward call. Required to
        # avoid OOM on a 48 GB GPU at 512x512.
        if is_guided_run and hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()

        if self.transformer.config.guidance_embeds:
            guidance_vec = torch.full([1], guidance_scale, device=device,
                                       dtype=torch.float32).expand(latents.shape[0])
        else:
            guidance_vec = None

        gen_for_noise = generator if isinstance(generator, torch.Generator) else None

        self.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            sigma = float(t / 1000.0)
            sigma_next = (float(timesteps[i + 1] / 1000.0)
                           if i + 1 < num_inference_steps else 0.0)
            sigma_t = torch.full(
                (latents.shape[0],), sigma,
                device=device, dtype=latents.dtype,
            )
            sigma_next_t = torch.full(
                (latents.shape[0],), sigma_next,
                device=device, dtype=latents.dtype,
            )

            with torch.no_grad():
                noise_pred = self._run_transformer_dual(
                    latents, sigma_t, sigma_next_t, prompt_embeds,
                    pooled_prompt_embeds, text_ids, latent_image_ids,
                    guidance_vec,
                ).to(latents.dtype)

            if is_guided_run and guidance_start_idx <= i < guidance_end_idx:
                if num_particles > 1:
                    grad, reward_value, raw_grad_norm = self._compute_grad_k_particles(
                        latents, sigma, snr_factor, reward_scale,
                        gradient_norm_scale, height, width, reward_fn,
                        prompt_embeds, pooled_prompt_embeds, text_ids,
                        latent_image_ids, guidance_vec, gen_for_noise,
                        num_particles=num_particles, lam=lam,
                    )
                else:
                    grad, reward_value, raw_grad_norm = self._compute_grad_k1(
                        latents, sigma, sigma_next, snr_factor, reward_scale,
                        gradient_norm_scale, height, width, reward_fn,
                        prompt_embeds, pooled_prompt_embeds, text_ids,
                        latent_image_ids, guidance_vec, gen_for_noise,
                        grad_divisor=grad_divisor,
                    )
                b_t = compute_b_coefficient(sigma, max_abs_b=max_abs_b)
                dt = sigma_next - sigma  # negative (σ decreases)
                if sigma_damp is not None:
                    # Effective lambda = magnitude of the rescaled gradient.
                    if grad_divisor is not None:
                        lambda_eff = raw_grad_norm / grad_divisor
                    else:
                        lambda_eff = gradient_norm_scale
                    damp_factor = damped_lambda_factor(sigma, sigma_damp,
                                                         lambda_eff)
                    grad = grad * damp_factor
                else:
                    damp_factor = 1.0
                noise_pred = noise_pred - b_t * grad

                if verbose:
                    eff_step = abs(dt) * b_t * gradient_norm_scale * damp_factor
                    rescale_factor = gradient_norm_scale / max(raw_grad_norm, 1e-12)
                    print(
                        f"DIAG step={i:3d} sigma={sigma:.4f} "
                        f"raw_grad_norm={raw_grad_norm:.4e} "
                        f"rescale_factor={rescale_factor:.4e} "
                        f"b_t={b_t:.3f} "
                        f"abs_dt={abs(dt):.4f} "
                        f"damp={damp_factor:.4f} "
                        f"eff_per_step={eff_step:.3f} "
                        f"reward={reward_value:+.4f}"
                    )

            latents = self.scheduler.step(noise_pred, t, latents,
                                            return_dict=False)[0]

        latents_unpacked = self._unpack_latents(latents, height, width,
                                                 self.vae_scale_factor)
        latents_unpacked = (latents_unpacked / self.vae.config.scaling_factor
                             + self.vae.config.shift_factor)
        image = self.vae.decode(latents_unpacked.to(vae_dtype(self.vae)),
                                  return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        return FluxPipelineOutput(images=image)
