"""Example usage of QwenVL -> Qwen-Image pipeline for multimodal image generation.

This example demonstrates how to:
1. Use QwenVL to extract embeddings from text+image inputs
2. Feed those embeddings to Qwen-Image for image generation
3. Leverage the dual-path architecture for image editing
4. Use vLLM's infrastructure for efficient processing

**Dual-Path Architecture:**
- Semantic Path: QwenVL embeddings guide content/meaning
- Visual Path: VAE latents preserve appearance/style
"""

import torch
from PIL import Image

# Example 1: Simple text-to-image generation (standalone Qwen-Image)
def example_qwenimage_only():
    """Use Qwen-Image directly for text-to-image generation."""
    from vllm_omni.worker.gpu_qwenimage_model_runner import QwenImageModelRunner

    print("=" * 60)
    print("Example 1: Qwen-Image text-to-image generation")
    print("=" * 60)

    # Initialize runner
    runner = QwenImageModelRunner(
        model_path="Qwen/Qwen-Image",
        device="cuda",
        dtype="bfloat16",
    )

    # Generate image
    prompt = "A cat holding a sign that says 'Hello World'"
    output = runner.generate(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        seed=42,
    )

    # Save image
    if output.images:
        output.images[0].save("qwenimage_output.png")
        print(f"✓ Generated image saved to qwenimage_output.png")
        print(f"  Prompt: {prompt}")


# Example 2: QwenVL embedding extraction + Qwen-Image generation
def example_qwenvl_to_qwenimage():
    """Use QwenVL embeddings for Qwen-Image generation."""
    from vllm.config import VllmConfig, ModelConfig, CacheConfig, ParallelConfig
    from vllm_omni.worker.gpu_qwenvl_qwenimage_worker import QwenVLToQwenImageWorker

    print("\n" + "=" * 60)
    print("Example 2: QwenVL -> Qwen-Image multimodal generation")
    print("=" * 60)

    # Configure vLLM for QwenVL
    model_config = ModelConfig(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        tokenizer="Qwen/Qwen2.5-VL-7B-Instruct",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
    )

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=4,
    )

    parallel_config = ParallelConfig()

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
    )

    # Initialize combined worker
    worker = QwenVLToQwenImageWorker(
        vllm_config=vllm_config,
        qwenimage_model_path="Qwen/Qwen-Image",
        device="cuda",
        dtype="bfloat16",
        embedding_layer=-1,  # Last layer
        pooling_method="last_token",
    )

    worker.initialize_model()
    print("✓ Models initialized")

    # Note: In real usage, you would get scheduler_output from vLLM
    # For this example, we show the interface
    print("\nUsage pattern:")
    print("""
    # Get embeddings from QwenVL
    embedding_output = worker.extract_embeddings(scheduler_output)

    # Generate images from embeddings
    result = worker.generate_images(
        embedding_output=embedding_output,
        height=1024,
        width=1024,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        seed=42,
    )

    # Or do it all in one step
    result = worker.generate_from_scheduler_output(
        scheduler_output=scheduler_output,
        height=1024,
        width=1024,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        seed=42,
    )
    """)


# Example 3: Using pre-generated embeddings
def example_with_pregenerated_embeddings():
    """Use pre-generated embeddings with Qwen-Image."""
    from vllm_omni.worker.gpu_qwenimage_model_runner import QwenImageModelRunner

    print("\n" + "=" * 60)
    print("Example 3: Using pre-generated embeddings")
    print("=" * 60)

    # Initialize runner
    runner = QwenImageModelRunner(
        model_path="Qwen/Qwen-Image",
        device="cuda",
        dtype="bfloat16",
    )

    # Simulate pre-generated embeddings
    # In practice, these come from QwenVL model runner
    batch_size = 1
    seq_len = 256
    hidden_dim = 4096  # Qwen2.5-VL-7B hidden size

    # Random embeddings for demonstration
    prompt_embeds = torch.randn(
        batch_size, seq_len, hidden_dim,
        dtype=torch.bfloat16,
        device="cuda"
    )
    prompt_embeds_mask = torch.ones(
        batch_size, seq_len,
        dtype=torch.long,
        device="cuda"
    )

    # Generate from embeddings
    output = runner.generate(
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        height=1024,
        width=1024,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        seed=42,
    )

    print(f"✓ Generated {len(output.images)} image(s) from embeddings")
    print(f"  Embedding used: {output.embedding_used}")


# Example 4: Batch generation with different modalities
def example_batch_multimodal():
    """Generate images from mixed text and multimodal inputs."""
    print("\n" + "=" * 60)
    print("Example 4: Batch multimodal generation with dual-path")
    print("=" * 60)

    print("""
    This example shows the dual-path architecture in action:

    **Text-only input:**
    - Semantic path: Text → QwenVL → embeddings
    - Visual path: None (generates from scratch)

    **Text + Image input:**
    - Semantic path: Text + Image → QwenVL → multimodal embeddings
    - Visual path: Image → VAE → latents
    - Combined: Preserves style while applying text instruction

    The QwenVL runner automatically:
    1. Encodes text with Qwen2.5-VL
    2. Encodes images with vision encoder (semantic)
    3. Saves original images for VAE encoding (visual)
    4. Fuses modalities into unified embeddings

    Qwen-Image then generates using both paths:
    - Embeddings guide the semantic content
    - VAE latents preserve visual appearance
    """)

    print("\nExample batch structure:")
    print("""
    batch = [
        {
            "text": "A beautiful sunset over mountains",
            "images": None,  # Pure text → no visual path
        },
        {
            "text": "Make it look like an oil painting",
            "images": [reference_image],  # Text + image → dual path
            # Semantic: "oil painting style" guidance
            # Visual: Reference image appearance
        },
    ]

    # QwenVL processes batch → embeddings + images
    # Qwen-Image generates with dual-path conditioning
    """)


# Example 5: Image editing with dual-path
def example_image_editing():
    """Advanced image editing using dual-path architecture."""
    from vllm_omni.worker.gpu_qwenimage_model_runner import QwenImageModelRunner

    print("\n" + "=" * 60)
    print("Example 5: Image Editing with Dual-Path Architecture")
    print("=" * 60)

    # Initialize runner
    runner = QwenImageModelRunner(
        model_path="Qwen/Qwen-Image-Edit",  # Note: Edit model
        device="cuda",
        dtype="bfloat16",
    )

    print("""
    Dual-path image editing workflow:

    1. Load reference image
    2. Provide editing instruction
    3. QwenVL extracts:
       - Semantic embeddings (understanding the instruction)
       - Original image (for VAE encoding)
    4. Qwen-Image generates:
       - Uses embeddings for content guidance
       - Uses VAE latents for style preservation

    Example:
    - Reference: Photo of a cat
    - Instruction: "Make the cat wear a hat"
    - Semantic path: Understands "add hat"
    - Visual path: Preserves cat's appearance
    - Result: Same cat, now wearing a hat
    """)

    try:
        # Simulate reference image
        reference_image = Image.new('RGB', (512, 512), color='gray')

        # Simulate pre-extracted embeddings + image
        batch_size = 1
        seq_len = 256
        hidden_dim = 4096

        prompt_embeds = torch.randn(
            batch_size, seq_len, hidden_dim,
            dtype=torch.bfloat16,
            device="cuda"
        )
        prompt_embeds_mask = torch.ones(
            batch_size, seq_len,
            dtype=torch.long,
            device="cuda"
        )

        # Generate with dual-path
        output = runner.generate(
            image=reference_image,  # Visual path: VAE encoding
            prompt_embeds=prompt_embeds,  # Semantic path: QwenVL embeddings
            prompt_embeds_mask=prompt_embeds_mask,
            height=1024,
            width=1024,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            seed=42,
        )

        print(f"✓ Generated {len(output.images)} edited image(s)")
        print(f"  Used dual-path conditioning")
        print(f"  Semantic: prompt_embeds")
        print(f"  Visual: VAE latents from reference image")

    except Exception as e:
        print(f"Example failed (model may not be available): {e}")


# Example 6: Advanced dual-path configuration
def example_advanced_dual_path():
    """Advanced dual-path configuration."""
    print("\n" + "=" * 60)
    print("Example 6: Advanced Dual-Path Configuration")
    print("=" * 60)

    print("""
    Advanced dual-path options:

    # Semantic Path (QwenVL)
    - embedding_layer: Which layer to extract (-1 = last layer)
    - pooling_method: "last_token", "mean", or "cls"
    - max_sequence_length: Max tokens for embeddings (default 512)

    # Visual Path (VAE)
    - Automatically handles:
      * Image preprocessing
      * VAE encoding
      * Latent normalization
    - Optimizations:
      * VAE tiling for large images
      * Attention slicing for memory efficiency

    # Generation Control
    - true_cfg_scale: 3.0-5.0 for strong text adherence
    - num_inference_steps: 30-50 (more = better quality, slower)
    - height/width: Must be divisible by 16
      Common sizes: (1024, 1024), (1664, 928), (1472, 1140)

    # Dual-path balance:
    When both paths are active:
    - Higher true_cfg_scale → stronger semantic guidance
    - Lower steps → more visual path influence
    """)

    from vllm_omni.worker.gpu_qwenimage_model_runner import QwenImageModelRunner

    runner = QwenImageModelRunner(
        model_path="Qwen/Qwen-Image-Edit",
        device="cuda",
        dtype="bfloat16",
    )

    print("\n✓ Runner initialized with dual-path support")
    print("  Pipeline: QwenImageEditPipeline")
    print("  Semantic path: prompt_embeds input")
    print("  Visual path: image → VAE latents")


# Example 7: Understanding the dual-path flow
def example_dual_path_flow():
    """Visualize the dual-path data flow."""
    print("\n" + "=" * 60)
    print("Example 7: Dual-Path Data Flow Visualization")
    print("=" * 60)

    print("""
Data Flow Diagram:
==================

Input: Text="Make it colorful" + Image=<cat.jpg>
    │
    ├──────────────── Semantic Path ────────────────┐
    │                                               │
    │   1. QwenVL Encoding                          │
    │      - Tokenize text                          │
    │      - Process image with vision encoder      │
    │      - Fuse modalities                        │
    │      ↓                                        │
    │   2. Extract Embeddings                       │
    │      - Layer: -1 (last layer)                │
    │      - Pool: last_token                       │
    │      - Shape: [1, seq_len, 4096]             │
    │      ↓                                        │
    │   prompt_embeds (semantic meaning) ───────────┤
    │                                               │
    ├──────────────── Visual Path ──────────────────┤
    │                                               │
    │   1. Preserve Original Image                  │
    │      - Store from QwenVL input               │
    │      - Format: PIL Image / Tensor            │
    │      ↓                                        │
    │   2. VAE Encoding                            │
    │      - Preprocess image                      │
    │      - Encode with VAE                       │
    │      - Normalize latents                     │
    │      - Shape: [1, C, H, W]                   │
    │      ↓                                        │
    │   image_latents (visual features) ────────────┤
    │                                               │
    └───────────────── Merge ───────────────────────┘
                         │
                         ▼
              Qwen-Image DiT Transformer
              - Attention over embeddings (semantic)
              - Condition on latents (visual)
              - Denoising process
                         │
                         ▼
                  Generated Image
              (Colorful version of cat.jpg)

Key Points:
-----------
1. Semantic path provides WHAT to generate
2. Visual path provides HOW it should look
3. Both paths are processed independently
4. DiT combines them during generation
5. Balance controlled via cfg_scale and steps
    """)


# Example 5: Advanced configuration
def example_advanced_config():
    """Advanced configuration options."""
    print("\n" + "=" * 60)
    print("Example 5: Advanced configuration")
    print("=" * 60)

    print("""
    # QwenVL Configuration
    - embedding_layer: Which layer to extract (-1 = last layer)
    - pooling_method: "last_token", "mean", or "cls"

    # Qwen-Image Generation
    - true_cfg_scale: 3.0-5.0 for strong text adherence
    - num_inference_steps: 30-50 (more = better quality, slower)
    - height/width: Must be divisible by 16
      Common sizes: (1024, 1024), (1664, 928), (1472, 1140)

    # Advanced options
    - negative_prompt: For classifier-free guidance
    - max_sequence_length: Max tokens to use (default 512)
    - latents: Custom starting latents
    - callback_on_step_end: Progress callback
    """)

    from vllm_omni.worker.gpu_qwenimage_model_runner import QwenImageModelRunner

    runner = QwenImageModelRunner(
        model_path="Qwen/Qwen-Image",
        device="cuda",
        dtype="bfloat16",
    )

    # Advanced generation
    output = runner.generate(
        prompt="A photorealistic portrait",
        negative_prompt="blurry, low quality, distorted",
        height=1472,
        width=1140,  # 4:3 aspect ratio
        num_inference_steps=50,
        true_cfg_scale=4.5,
        guidance_scale=1.0,
        seed=123,
        max_sequence_length=512,
    )

    print("✓ Advanced generation complete")
    print(f"  Image size: {output.images[0].size if output.images else 'N/A'}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("QwenVL -> Qwen-Image Dual-Path Pipeline Examples")
    print("=" * 60)
    print("\nThese examples demonstrate the integration of:")
    print("- QwenVL: Multimodal embedding extraction (vLLM-powered)")
    print("- Qwen-Image: DiT-based image generation (diffusers)")
    print("- Dual-Path: Semantic + Visual conditioning")
    print()

    # Run examples (comment out if models not available)
    try:
        example_qwenimage_only()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    example_qwenvl_to_qwenimage()
    example_with_pregenerated_embeddings()
    example_batch_multimodal()
    example_image_editing()
    example_advanced_dual_path()
    example_dual_path_flow()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nFor more details on dual-path architecture, see:")
    print("  vllm_omni/worker/README_DUAL_PATH.md")
