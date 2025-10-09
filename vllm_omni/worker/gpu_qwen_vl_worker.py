"""Worker for Qwen2.5-VL embedding extraction for DiT."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.v1.worker.worker_base import WorkerBase
from vllm.multimodal import MultiModalDataDict
from vllm.model_executor.models.qwen2_vl import Qwen2VLForConditionalGeneration
from vllm.v1.core.sched.output import SchedulerOutput

from .gpu_qwen_vl_model_runner import (
    QwenVLModelRunner,
    QwenVLEmbeddingOutput
)

logger = logging.getLogger(__name__)


class QwenVLEmbeddingWorker(WorkerBase):
    """Worker for extracting embeddings from Qwen2.5-VL for DiT.

    This worker wraps QwenVLModelRunner and provides embedding extraction
    capabilities that integrate with vLLM's existing multimodal infrastructure.
    """

    def __init__(
        self,
        vllm_config: Optional[VllmConfig] = None,
        model_path: Optional[str] = None,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        embedding_layer: int = -1,
        pooling_method: str = "last_token"
    ):
        """Initialize Qwen2.5-VL embedding worker.

        Args:
            vllm_config: vLLM configuration object (preferred)
            model_path: Path to Qwen2.5-VL model (fallback if no config)
            device: Device to use (cuda/cpu/mps)
            dtype: Data type (float16/float32)
            embedding_layer: Which layer to extract embeddings from (-1 = last layer)
            pooling_method: How to pool embeddings (last_token/mean/cls)

        Raises:
            ValueError: If neither vllm_config nor model_path is provided
        """
        if vllm_config is None and model_path is None:
            raise ValueError("Either vllm_config or model_path must be provided")

        self.vllm_config = vllm_config
        self.model_path = model_path or (vllm_config.model_config.model if vllm_config else None)
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.embedding_layer = embedding_layer
        self.pooling_method = pooling_method

        # Model runner will be initialized lazily
        self.model_runner: Optional[QwenVLModelRunner] = None
        self._initialized = False

        logger.info(f"QwenVLEmbeddingWorker initialized with model: {self.model_path}")

    def _resolve_device(self, device: Optional[str]) -> str:
        """Resolve device to use."""
        if device:
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _resolve_dtype(self, dtype: Optional[str]) -> torch.dtype:
        """Resolve data type."""
        if dtype:
            return getattr(torch, dtype)

        # Default based on device
        if self.device in ["cuda", "mps"]:
            return torch.float16
        else:
            return torch.float32

    def _initialize_runner(self) -> QwenVLModelRunner:
        """Initialize the Qwen VL model runner.

        Returns:
            QwenVLModelRunner instance configured for embedding extraction

        Raises:
            ValueError: If vllm_config is not provided
        """
        if self.vllm_config is None:
            raise ValueError("vllm_config is required for model runner initialization")

        logger.info("Initializing QwenVLModelRunner...")

        # Create model runner with vLLM config
        runner = QwenVLModelRunner(self.vllm_config)

        # Configure embedding extraction settings
        runner.set_embedding_config(
            layer=self.embedding_layer,
            pooling=self.pooling_method
        )

        logger.info(
            f"QwenVLModelRunner initialized with embedding_layer={self.embedding_layer}, "
            f"pooling={self.pooling_method}"
        )

        return runner

    def initialize_model(self) -> None:
        """Initialize the model runner (WorkerBase interface).

        This method is called by vLLM's worker initialization process.
        """
        if self._initialized:
            logger.warning("Model already initialized, skipping")
            return

        try:
            # Initialize runner using dedicated method
            self.model_runner = self._initialize_runner()

            self._initialized = True
            logger.info("Model runner initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize model runner: {e}")
            raise

    def _get_model_hidden_dim(self) -> int:
        """Get model's hidden dimension."""
        if self.vllm_config and hasattr(self.vllm_config.model_config, 'hidden_size'):
            return self.vllm_config.model_config.hidden_size
        # Default for Qwen2.5-VL models
        return 4096

    def _ensure_initialized(self) -> None:
        """Ensure model is initialized before use."""
        if not self._initialized:
            self.initialize_model()

    def extract_embeddings(
        self,
        scheduler_output: SchedulerOutput,
    ) -> QwenVLEmbeddingOutput:
        """Extract embeddings using model runner's execute_model.

        Args:
            scheduler_output: Scheduler output containing batched requests

        Returns:
            Embeddings output from Qwen2.5-VL

        Raises:
            RuntimeError: If model is not initialized
        """
        self._ensure_initialized()

        if self.model_runner is None:
            raise RuntimeError("Model runner not initialized")

        try:
            # Execute model to get embeddings
            output = self.model_runner.execute_model(scheduler_output)

            if not isinstance(output, QwenVLEmbeddingOutput):
                raise TypeError(
                    f"Expected QwenVLEmbeddingOutput, got {type(output)}"
                )

            return output

        except Exception as e:
            logger.error(f"Failed to extract embeddings: {e}")
            raise

    async def extract_embeddings_async(
        self,
        scheduler_output: SchedulerOutput,
    ) -> QwenVLEmbeddingOutput:
        """Async version of embedding extraction.

        Args:
            scheduler_output: Scheduler output containing batched requests

        Returns:
            Embeddings output from Qwen2.5-VL
        """
        # Run in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.extract_embeddings,
            scheduler_output
        )

    def cleanup(self) -> None:
        """Clean up resources (WorkerBase interface)."""
        try:
            if self.model_runner is not None:
                # Clean up model runner resources if it has cleanup method
                if hasattr(self.model_runner, 'cleanup'):
                    self.model_runner.cleanup()

                # Clear CUDA cache if using GPU
                if self.device == "cuda" and torch.cuda.is_available():
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
            pass  # Ignore errors during cleanup in destructor


class QwenVLDiTWorker(WorkerBase):
    """Combined worker for Qwen2.5-VL embeddings + DiT generation.

    This worker handles the full pipeline from multimodal input
    to DiT-based generation.
    """

    def __init__(
        self,
        qwen_model_path: str,
        dit_model_path: str,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs
    ):
        """Initialize combined worker.

        Args:
            qwen_model_path: Path to Qwen2.5-VL model
            dit_model_path: Path to DiT model
            device: Device to use
            dtype: Data type
        """
        # Initialize Qwen embedding worker
        self.qwen_worker = QwenVLEmbeddingWorker(
            model_path=qwen_model_path,
            device=device,
            dtype=dtype,
            **kwargs
        )

        # Initialize DiT worker (uses existing diffusion runner)
        from .gpu_diffusion_model_runner import DiffusionModelRunner
        self.dit_runner = DiffusionModelRunner(
            model_path=dit_model_path,
            device=device,
            dtype=dtype
        )

    def generate(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        *,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        **kwargs
    ):
        """Generate image from multimodal input.

        Args:
            prompt: Text prompt
            images: Optional reference images
            height: Output height
            width: Output width
            num_inference_steps: DiT inference steps

        Returns:
            Generated images
        """
        # Extract embeddings from Qwen2.5-VL
        output = self.qwen_worker.extract_embeddings(
            prompt=prompt,
            images=images
        )

        # Note: In a real implementation, embeddings would be used to condition DiT
        # For now, using text prompt directly as DiT conditioning is model-specific

        output = self.dit_runner.generate(
            prompt=prompt,  # This would be replaced with embedding conditioning
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            **kwargs
        )

        return output