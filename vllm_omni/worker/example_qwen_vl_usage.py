"""Integration example for Qwen2.5-VL embeddings with DiT in vLLM-omni."""

from typing import List, Optional, Any
import torch
from PIL import Image


def example_basic_embedding_extraction():
    """Basic example of extracting embeddings from Qwen2.5-VL."""

    from vllm_omni.worker.gpu_qwen_vl_worker import QwenVLEmbeddingWorker

    # Initialize worker
    worker = QwenVLEmbeddingWorker(
        model_path="Qwen/Qwen2-VL-7B-Instruct",  # or your model path
        device="cuda",
        dtype="float16",
        dit_dim=1024,
        embedding_layer=-1,  # Last layer
        pooling_method="mean",
        use_adapter=True
    )

    # Single input example
    text = "A beautiful landscape with mountains and a lake"
    image = Image.new('RGB', (448, 448))  # Placeholder image

    # Extract embeddings
    embeddings = worker.extract_embeddings(
        prompt=text,
        images=[image]
    )

    print(f"Embeddings shape: {embeddings.shape}")  # [1024] for DiT

    # Batch example
    texts = [
        "Generate a cat playing with yarn",
        "Create a sunset over the ocean",
        "Design a modern city skyline"
    ]
    images_batch = [
        [Image.new('RGB', (448, 448))],  # Reference for first
        None,  # No reference for second
        [Image.new('RGB', (448, 448)), Image.new('RGB', (448, 448))]  # Two references for third
    ]

    batch_embeddings = worker.batch_extract_embeddings(texts, images_batch)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")  # [3, 1024]
