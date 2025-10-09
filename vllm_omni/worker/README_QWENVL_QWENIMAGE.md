# QwenVL → Qwen-Image Integration

This module implements a complete pipeline for multimodal image generation:

1. **QwenVL**: Extract embeddings from text + image inputs (using vLLM)
2. **Qwen-Image**: Generate images from embeddings (using diffusers DiT)

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Multimodal Input (text + images)                        │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│  QwenVLModelRunner (vLLM-powered)                        │
│  - Encodes text with Qwen2.5-VL                          │
│  - Encodes images with vision encoder                    │
│  - Fuses modalities                                      │
│  - Pools to fixed embeddings                             │
└────────────────┬─────────────────────────────────────────┘
                 │
                 │ embeddings [batch, seq_len, hidden_dim]
                 │ attention_mask [batch, seq_len]
                 ▼
┌──────────────────────────────────────────────────────────┐
│  QwenImageModelRunner (diffusers)                        │
│  - Prepares embeddings for DiT                           │
│  - Runs denoising process                                │
│  - Decodes latents to images                             │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│  Generated Images                                         │
└──────────────────────────────────────────────────────────┘
```

## Files

### Core Implementation

1. **`gpu_qwen_vl_model_runner.py`**
   - Extends `GPUModelRunner` from vLLM
   - Extracts embeddings instead of generating tokens
   - Reuses vLLM's infrastructure (PP/TP, CUDA graphs, multimodal handling)

2. **`gpu_qwen_vl_worker.py`**
   - Worker wrapper for `QwenVLModelRunner`
   - Implements `WorkerBase` interface
   - Manages initialization and lifecycle

3. **`gpu_qwenimage_model_runner.py`**
   - Wraps Qwen-Image diffusers pipeline
   - Supports pre-generated embeddings
   - Handles embedding preprocessing for DiT

4. **`gpu_qwenvl_qwenimage_worker.py`**
   - Combined worker integrating both runners
   - End-to-end pipeline management
   - Handles embedding format conversion

### Examples

- **`example_qwenvl_qwenimage_usage.py`**: Comprehensive usage examples

## Usage

### Option 1: Standalone Qwen-Image (text-to-image)

```python
from vllm_omni.worker import QwenImageModelRunner

runner = QwenImageModelRunner(
    model_path="Qwen/Qwen-Image",
    device="cuda",
    dtype="bfloat16",
)

output = runner.generate(
    prompt="A cat holding a sign that says 'Hello World'",
    height=1024,
    width=1024,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    seed=42,
)

output.images[0].save("output.png")
```

### Option 2: Combined Pipeline (multimodal-to-image)

```python
from vllm.config import VllmConfig
from vllm_omni.worker import QwenVLToQwenImageWorker

# Configure vLLM for QwenVL
vllm_config = VllmConfig(...)

# Initialize combined worker
worker = QwenVLToQwenImageWorker(
    vllm_config=vllm_config,
    qwenimage_model_path="Qwen/Qwen-Image",
    device="cuda",
    dtype="bfloat16",
)

worker.initialize_model()

# End-to-end generation
result = worker.generate_from_scheduler_output(
    scheduler_output=scheduler_output,  # From vLLM
    height=1024,
    width=1024,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    seed=42,
)

# Access generated images
for img in result.images:
    img.save("output.png")
```

### Option 3: Pre-generated Embeddings

```python
from vllm_omni.worker import QwenImageModelRunner

runner = QwenImageModelRunner(...)

# Embeddings from QwenVL
prompt_embeds = torch.randn(1, 256, 4096, dtype=torch.bfloat16)
prompt_embeds_mask = torch.ones(1, 256, dtype=torch.long)

output = runner.generate(
    prompt_embeds=prompt_embeds,
    prompt_embeds_mask=prompt_embeds_mask,
    height=1024,
    width=1024,
)
```

## Key Features

### QwenVLModelRunner

- ✅ Reuses `GPUModelRunner` infrastructure
- ✅ Supports pipeline parallelism (PP)
- ✅ Supports tensor parallelism (TP)
- ✅ Multimodal input handling
- ✅ CUDA graph optimization
- ✅ Efficient memory management
- ✅ Configurable embedding extraction (layer, pooling)

### QwenImageModelRunner

- ✅ Wraps official Qwen-Image pipeline
- ✅ Accepts pre-generated embeddings
- ✅ Supports all Qwen-Image features
- ✅ Memory optimizations (VAE tiling, attention slicing)
- ✅ Flexible generation parameters

### Integration

- ✅ Seamless embedding format conversion
- ✅ Batch processing support
- ✅ Mixed modality handling (text-only, text+image)
- ✅ Error handling and validation
- ✅ Resource cleanup

## Configuration

### QwenVL Embedding Extraction

```python
embedding_layer = -1  # -1 = last layer, or specify layer index
pooling_method = "last_token"  # "last_token", "mean", or "cls"
```

### Qwen-Image Generation

```python
height = 1024  # Must be divisible by 16
width = 1024   # Common: (1024,1024), (1664,928), (1472,1140)
num_inference_steps = 50  # 30-50 recommended
true_cfg_scale = 4.0  # 3.0-5.0 for text adherence
guidance_scale = 1.0  # Usually 1.0
```

## Performance Notes

1. **Memory**: Both models are large (~7B + ~7B parameters)
   - QwenVL: ~14GB
   - Qwen-Image: ~14GB
   - Total: ~28GB VRAM recommended

2. **Speed**:
   - QwenVL embedding: ~100ms (with vLLM optimizations)
   - Qwen-Image generation: ~5-10s (50 steps)

3. **Optimizations**:
   - Use `bfloat16` for best quality/speed balance
   - Enable CUDA graphs via vLLM
   - Use VAE tiling for large images
   - Batch multiple requests together

## References

- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/)

## Future Improvements

- [ ] Support for prompt enhancement (from Qwen-Image demo)
- [ ] Image editing capabilities (Qwen-Image-Edit)
- [ ] LoRA fine-tuning support
- [ ] Multi-GPU inference
- [ ] Streaming/progressive generation
- [ ] Cached embedding reuse
