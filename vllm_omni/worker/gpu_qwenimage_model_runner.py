"""Qwen-Image Model Runner for text+embedding to image generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from .gpu_diffusion_model_runner import DiffusionRunnerOutput


@dataclass
class QwenImageRunnerOutput(DiffusionRunnerOutput):
    """Output from Qwen-Image generation with embedding support."""

    embedding_used: bool = False  # Whether pre-generated embeddings were used
    original_prompt: Optional[str] = None  # Original prompt before enhancement


class QwenImageModelRunner:
    """Model runner for Qwen-Image pipeline.

    Extends DiffusionModelRunner to support:
    - Pre-generated embeddings from QwenVL model
    - Qwen-Image specific preprocessing
    - Text enhancement and prompt templating
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        """Initialize Qwen-Image runner.

        Args:
            model_path: Path to Qwen-Image model (e.g., "Qwen/Qwen-Image")
            device: Device to use (cuda/cpu/mps)
            dtype: Data type (bfloat16/float16/float32)
        """
        self.model_path = model_path
        self.device, self.torch_dtype = self._resolve_device_and_dtype(device, dtype)
        self._pipeline = self._load_pipeline()

    def _resolve_device_and_dtype(
        self,
        device: Optional[str],
        dtype: Optional[str],
    ):
        """Resolve device and dtype for Qwen-Image."""
        import torch

        resolved_device = device
        if resolved_device is None:
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"

        # Qwen-Image works best with bfloat16
        if dtype is not None:
            resolved_dtype = getattr(torch, dtype, None)
        else:
            if resolved_device == "cuda" and torch.cuda.is_bf16_supported():
                resolved_dtype = torch.bfloat16
            elif resolved_device in {"cuda", "mps"}:
                resolved_dtype = torch.float16
            else:
                resolved_dtype = torch.float32

        return resolved_device, resolved_dtype

    def _load_pipeline(self):
        """Load Qwen-Image Edit pipeline for dual-path support."""
        from diffusers import QwenImageEditPipeline

        pipeline = QwenImageEditPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
        )

        try:
            pipeline = pipeline.to(self.device)
        except Exception:
            pipeline = pipeline.to("cpu")

        # Enable optimizations
        self._apply_optimizations(pipeline)
        return pipeline

    @staticmethod
    def _apply_optimizations(pipeline) -> None:
        """Apply memory optimizations to Qwen-Image pipeline."""
        try:
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
        except Exception:
            pass

        try:
            vae = getattr(pipeline, "vae", None)
            if vae and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
        except Exception:
            pass

    def _encode_vae_image(self, image, generator=None):
        """Encode image using VAE for dual-path conditioning.

        Args:
            image: PIL Image or torch.Tensor
            generator: Random generator for VAE encoding

        Returns:
            image_latents: VAE-encoded latents [B, C, H, W]
        """
        import torch
        from PIL import Image

        # Ensure we have a pipeline with VAE
        if not hasattr(self._pipeline, 'vae'):
            return None

        # Convert PIL to tensor if needed
        if isinstance(image, Image.Image):
            # Use pipeline's image processor
            image = self._pipeline.image_processor.preprocess(image)
            if image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dim
            image = image.unsqueeze(2)  # Add time dim for QwenImage VAE

        # Move to device
        image = image.to(device=self.device, dtype=self.torch_dtype)

        # Encode with VAE
        if isinstance(generator, list):
            image_latents = [
                self._retrieve_latents(
                    self._pipeline.vae.encode(image[i:i+1]),
                    generator=generator[i],
                    sample_mode="argmax"
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self._retrieve_latents(
                self._pipeline.vae.encode(image),
                generator=generator,
                sample_mode="argmax"
            )

        # Normalize latents
        latent_channels = self._pipeline.vae.config.z_dim
        latents_mean = (
            torch.tensor(self._pipeline.vae.config.latents_mean)
            .view(1, latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self._pipeline.vae.config.latents_std)
            .view(1, latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents

    @staticmethod
    def _retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
        """Retrieve latents from VAE encoder output."""
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            return encoder_output.latents
        else:
            raise AttributeError("Could not access latents of provided encoder_output")

    def generate(
        self,
        prompt: Optional[str] = None,
        *,
        # Image inputs (for dual-path conditioning)
        image: Optional[Any] = None,  # PIL Image or list of images for VAE encoding
        image_latents: Optional[torch.Tensor] = None,  # Pre-encoded VAE latents
        # Embedding inputs (from QwenVL)
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        # Image generation params
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        guidance_scale: float = 1.0,
        negative_prompt: Optional[str] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        max_sequence_length: int = 512,
        **kwargs,
    ) -> QwenImageRunnerOutput:
        """Generate images using Qwen-Image with dual-path conditioning.

        Args:
            prompt: Text prompt (or None if using prompt_embeds)
            image: Input image for dual-path conditioning (VAE encoding)
            image_latents: Pre-encoded VAE latents (alternative to image)
            prompt_embeds: Pre-generated embeddings from QwenVL [batch, seq_len, hidden_dim]
            prompt_embeds_mask: Attention mask for embeddings [batch, seq_len]
            height: Output image height (should be divisible by 16)
            width: Output image width (should be divisible by 16)
            num_inference_steps: Number of denoising steps
            true_cfg_scale: Classifier-free guidance scale
            guidance_scale: Additional guidance scale (usually 1.0)
            negative_prompt: Negative prompt for CFG
            negative_prompt_embeds: Pre-generated negative embeddings
            negative_prompt_embeds_mask: Mask for negative embeddings
            seed: Random seed for reproducibility
            max_sequence_length: Max sequence length for embeddings
            **kwargs: Additional pipeline arguments

        Returns:
            QwenImageRunnerOutput with generated images
        """
        import torch

        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))
            except Exception:
                generator = torch.Generator().manual_seed(int(seed))

        # Process image input for dual-path conditioning
        if image is not None and image_latents is None:
            # Encode image with VAE for visual path
            image_latents = self._encode_vae_image(image, generator)

        # Move embeddings to correct device/dtype if provided
        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.torch_dtype)
            embedding_used = True
        else:
            embedding_used = False

        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.to(device=self.device)

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device=self.device, dtype=self.torch_dtype
            )

        if negative_prompt_embeds_mask is not None:
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(device=self.device)

        # Call Qwen-Image Edit pipeline with dual-path inputs
        output = self._pipeline(
            image=image_latents,  # VAE latents for visual path
            prompt=prompt,
            prompt_embeds=prompt_embeds,  # Semantic embeddings from QwenVL
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt=negative_prompt,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            height=int(height),
            width=int(width),
            num_inference_steps=int(num_inference_steps),
            true_cfg_scale=float(true_cfg_scale),
            guidance_scale=float(guidance_scale),
            generator=generator,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )

        return QwenImageRunnerOutput(
            prompt=prompt or "Generated from embeddings",
            images=getattr(output, "images", []) or [],
            embedding_used=embedding_used,
            original_prompt=prompt,
        )

    def generate_from_qwenvl_output(
        self,
        qwenvl_embeddings: torch.Tensor,
        qwenvl_mask: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> QwenImageRunnerOutput:
        """Generate images directly from QwenVL embeddings.

        This is a convenience method for using QwenVL output directly.

        Args:
            qwenvl_embeddings: Embeddings from QwenVLModelRunner [batch, seq_len, hidden_dim]
            qwenvl_mask: Attention mask from QwenVL
            **generation_kwargs: Additional generation parameters

        Returns:
            QwenImageRunnerOutput with generated images
        """
        return self.generate(
            prompt_embeds=qwenvl_embeddings,
            prompt_embeds_mask=qwenvl_mask,
            **generation_kwargs,
        )
