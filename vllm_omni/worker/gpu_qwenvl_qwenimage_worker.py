"""Combined worker for QwenVL (embedding extraction) + Qwen-Image (generation)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.core.sched.output import SchedulerOutput

from .gpu_qwen_vl_model_runner import QwenVLModelRunner, QwenVLEmbeddingOutput
from .gpu_qwenimage_model_runner import QwenImageModelRunner, QwenImageRunnerOutput

logger = logging.getLogger(__name__)


@dataclass
class QwenVLToImageOutput:
    """Combined output from QwenVL -> Qwen-Image pipeline."""

    images: List[Any]  # Generated images
    embeddings: List[Optional[torch.Tensor]]  # QwenVL embeddings used
    modality_types: Optional[List[str]] = None  # Input modality types
    prompt: Optional[str] = None  # Original prompt
    req_ids: Optional[List[str]] = None  # Request IDs
    finished: bool = True


class QwenVLToQwenImageWorker(WorkerBase):
    """Worker combining QwenVL embedding extraction with Qwen-Image generation.

    This worker implements the full dual-path pipeline for Qwen-Image:

    **Dual-Path Architecture:**
    1. **Semantic Path**: Extract multimodal embeddings from QwenVL
       - Input: text + images → Qwen2.5-VL → semantic embeddings
       - Used as: prompt_embeds for text-to-image conditioning

    2. **Visual Path**: Encode input images with VAE
       - Input: images → VAE encoder → latent representations
       - Used as: image_latents for appearance/style conditioning

    3. **Combined Generation**: Qwen-Image DiT merges both paths
       - Semantic embeddings guide content/meaning
       - VAE latents preserve visual appearance/style
       - Enables powerful image editing and variation generation

    Follows vLLM worker patterns for integration.
    """

    def __init__(
        self,
        # QwenVL config
        vllm_config: Optional[VllmConfig] = None,
        qwenvl_model_path: Optional[str] = None,
        # Qwen-Image config
        qwenimage_model_path: str = "Qwen/Qwen-Image",
        # Shared config
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        # QwenVL embedding config
        embedding_layer: int = -1,
        pooling_method: str = "last_token",
    ):
        """Initialize combined worker.

        Args:
            vllm_config: vLLM configuration for QwenVL model runner
            qwenvl_model_path: Path to QwenVL model (fallback if no vllm_config)
            qwenimage_model_path: Path to Qwen-Image model
            device: Device to use (cuda/cpu/mps)
            dtype: Data type (bfloat16/float16/float32)
            embedding_layer: Which layer to extract embeddings from
            pooling_method: Pooling method for embeddings
        """
        if vllm_config is None and qwenvl_model_path is None:
            raise ValueError("Either vllm_config or qwenvl_model_path must be provided")

        self.vllm_config = vllm_config
        self.qwenvl_model_path = qwenvl_model_path
        self.qwenimage_model_path = qwenimage_model_path
        self.device = device
        self.dtype = dtype
        self.embedding_layer = embedding_layer
        self.pooling_method = pooling_method

        # Initialize runners
        self.qwenvl_runner: Optional[QwenVLModelRunner] = None
        self.qwenimage_runner: Optional[QwenImageModelRunner] = None
        self._initialized = False

        logger.info(
            f"QwenVLToQwenImageWorker initialized with "
            f"QwenVL: {qwenvl_model_path or 'from vllm_config'}, "
            f"Qwen-Image: {qwenimage_model_path}"
        )

    def initialize_model(self) -> None:
        """Initialize both QwenVL and Qwen-Image runners.

        This follows the WorkerBase interface for vLLM integration.
        """
        if self._initialized:
            logger.warning("Models already initialized, skipping")
            return

        try:
            # Initialize QwenVL embedding runner
            logger.info("Initializing QwenVL embedding runner...")
            if self.vllm_config is None:
                raise ValueError("vllm_config is required for QwenVL initialization")

            self.qwenvl_runner = QwenVLModelRunner(self.vllm_config)
            self.qwenvl_runner.set_embedding_config(
                layer=self.embedding_layer,
                pooling=self.pooling_method
            )
            logger.info("QwenVL runner initialized")

            # Initialize Qwen-Image runner
            logger.info("Initializing Qwen-Image runner...")
            self.qwenimage_runner = QwenImageModelRunner(
                model_path=self.qwenimage_model_path,
                device=self.device,
                dtype=self.dtype,
            )
            logger.info("Qwen-Image runner initialized")

            self._initialized = True
            logger.info("All runners initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize runners: {e}")
            raise

    def extract_embeddings(
        self,
        scheduler_output: SchedulerOutput,
    ) -> QwenVLEmbeddingOutput:
        """Extract embeddings from QwenVL.

        Args:
            scheduler_output: Scheduler output with batched requests

        Returns:
            QwenVLEmbeddingOutput with embeddings
        """
        if not self._initialized:
            self.initialize_model()

        if self.qwenvl_runner is None:
            raise RuntimeError("QwenVL runner not initialized")

        return self.qwenvl_runner.execute_model(scheduler_output)

    def generate_images(
        self,
        embedding_output: QwenVLEmbeddingOutput,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: Optional[int] = None,
        **kwargs,
    ) -> QwenVLToImageOutput:
        """Generate images from QwenVL embeddings with dual-path conditioning.

        This implements the dual-path architecture:
        1. Semantic path: QwenVL embeddings -> prompt_embeds
        2. Visual path: Input images -> VAE latents

        Args:
            embedding_output: Output from QwenVL embedding extraction
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            true_cfg_scale: Classifier-free guidance scale
            seed: Random seed
            **kwargs: Additional generation parameters

        Returns:
            QwenVLToImageOutput with generated images
        """
        if not self._initialized:
            self.initialize_model()

        if self.qwenimage_runner is None:
            raise RuntimeError("Qwen-Image runner not initialized")

        # Prepare embeddings and masks for Qwen-Image (semantic path)
        embeddings = embedding_output.pooler_output

        # Stack embeddings into batch
        valid_embeddings = [e for e in embeddings if e is not None]
        if not valid_embeddings:
            raise ValueError("No valid embeddings found")

        # Stack into batch tensor [batch, seq_len, hidden_dim]
        # For now, assume each embedding is a single vector, expand to sequence
        batch_embeddings = []
        for emb in valid_embeddings:
            if emb.dim() == 1:
                # Single vector -> [1, hidden_dim]
                emb = emb.unsqueeze(0)
            batch_embeddings.append(emb)

        # Pad to same sequence length
        max_seq_len = max(e.shape[0] for e in batch_embeddings)
        padded_embeddings = []
        attention_masks = []

        for emb in batch_embeddings:
            seq_len = emb.shape[0]
            if seq_len < max_seq_len:
                # Pad with zeros
                padding = torch.zeros(
                    max_seq_len - seq_len,
                    emb.shape[1],
                    dtype=emb.dtype,
                    device=emb.device
                )
                emb = torch.cat([emb, padding], dim=0)
                mask = torch.cat([
                    torch.ones(seq_len, dtype=torch.long, device=emb.device),
                    torch.zeros(max_seq_len - seq_len, dtype=torch.long, device=emb.device)
                ])
            else:
                mask = torch.ones(seq_len, dtype=torch.long, device=emb.device)

            padded_embeddings.append(emb)
            attention_masks.append(mask)

        prompt_embeds = torch.stack(padded_embeddings)  # [batch, seq_len, hidden_dim]
        prompt_embeds_mask = torch.stack(attention_masks)  # [batch, seq_len]

        # Prepare input images for VAE encoding (visual path)
        input_images = None
        if hasattr(embedding_output, 'input_images') and embedding_output.input_images:
            # Filter out None images
            valid_images = [img for img in embedding_output.input_images if img is not None]
            if valid_images:
                # Use the first valid image (or batch them if multiple)
                input_images = valid_images[0] if len(valid_images) == 1 else valid_images

        # Generate images with dual-path conditioning
        result = self.qwenimage_runner.generate(
            image=input_images,  # Visual path: images -> VAE latents
            prompt_embeds=prompt_embeds,  # Semantic path: QwenVL embeddings
            prompt_embeds_mask=prompt_embeds_mask,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            seed=seed,
            **kwargs,
        )

        return QwenVLToImageOutput(
            images=result.images,
            embeddings=embedding_output.pooler_output,
            modality_types=embedding_output.modality_types,
            req_ids=embedding_output.req_ids,
        )

    def generate_from_scheduler_output(
        self,
        scheduler_output: SchedulerOutput,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: Optional[int] = None,
        **kwargs,
    ) -> QwenVLToImageOutput:
        """End-to-end generation from scheduler output.

        This is the main entry point for the combined pipeline.

        Args:
            scheduler_output: Scheduler output with multimodal inputs
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            true_cfg_scale: Classifier-free guidance scale
            seed: Random seed
            **kwargs: Additional generation parameters

        Returns:
            QwenVLToImageOutput with generated images
        """
        # Step 1: Extract embeddings from QwenVL
        logger.info("Extracting embeddings from QwenVL...")
        embedding_output = self.extract_embeddings(scheduler_output)

        # Step 2: Generate images with Qwen-Image
        logger.info("Generating images with Qwen-Image...")
        return self.generate_images(
            embedding_output=embedding_output,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            seed=seed,
            **kwargs,
        )

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._initialized = False
            logger.info("Worker cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass
