# Qwen-Image 双路径架构实现说明

## 概述

本实现完整支持 Qwen-Image 的双路径（Dual-Path）架构，用于图像编辑和生成任务。

## 双路径架构

Qwen-Image 使用两条并行路径来处理图像输入：

### 1. 语义路径 (Semantic Path)
- **输入**: 文本 prompt + 图像
- **处理**: Qwen2.5-VL 多模态模型
- **输出**: 语义 embeddings (prompt_embeds)
- **作用**: 理解图像的语义内容和文本指令的含义

### 2. 视觉路径 (Visual/Appearance Path)
- **输入**: 原始图像
- **处理**: VAE (Variational Autoencoder) 编码器
- **输出**: 视觉 latents (image_latents)
- **作用**: 保留图像的低级视觉特征（纹理、颜色、风格）

### 3. 路径融合
- Qwen-Image DiT transformer 同时接收两条路径的输出
- 语义 embeddings 指导生成内容的含义和结构
- VAE latents 保持视觉外观的一致性
- 两者结合实现精确的图像编辑和风格迁移

## 实现细节

### 核心修改

#### 1. `QwenVLEmbeddingOutput` (gpu_qwen_vl_model_runner.py)
```python
@dataclass
class QwenVLEmbeddingOutput(ModelRunnerOutput):
    pooler_output: List[Optional[torch.Tensor]]  # 语义 embeddings
    modality_types: Optional[List[str]] = None   # 模态类型
    input_images: Optional[List[Optional[Any]]] = None  # 原始图像（新增）
```

**改动**: 添加 `input_images` 字段以保存原始图像数据，供 VAE 编码使用。

#### 2. `QwenImageModelRunner` (gpu_qwenimage_model_runner.py)

**a. 使用 QwenImageEditPipeline**
```python
def _load_pipeline(self):
    from diffusers import QwenImageEditPipeline
    pipeline = QwenImageEditPipeline.from_pretrained(...)
```

**b. 添加 VAE 编码方法**
```python
def _encode_vae_image(self, image, generator=None):
    """将图像编码为 VAE latents"""
    image_latents = self._pipeline.vae.encode(image)
    # 归一化处理
    return normalized_latents
```

**c. 扩展 generate 方法**
```python
def generate(
    self,
    prompt: Optional[str] = None,
    image: Optional[Any] = None,          # 新增：输入图像
    image_latents: Optional[torch.Tensor] = None,  # 新增：预编码 latents
    prompt_embeds: Optional[torch.Tensor] = None,  # 语义 embeddings
    ...
):
    # 如果提供图像，编码为 VAE latents
    if image is not None and image_latents is None:
        image_latents = self._encode_vae_image(image, generator)

    # 调用 pipeline，传递两条路径的数据
    output = self._pipeline(
        image=image_latents,         # 视觉路径
        prompt_embeds=prompt_embeds, # 语义路径
        ...
    )
```

#### 3. `QwenVLToQwenImageWorker` (gpu_qwenvl_qwenimage_worker.py)

**修改 generate_images 方法**
```python
def generate_images(self, embedding_output, ...):
    # 提取语义 embeddings (来自 QwenVL)
    prompt_embeds = ...

    # 提取原始图像 (视觉路径)
    input_images = None
    if hasattr(embedding_output, 'input_images'):
        input_images = embedding_output.input_images[0]

    # 调用 runner，传递双路径输入
    result = self.qwenimage_runner.generate(
        image=input_images,              # 视觉路径：自动 VAE 编码
        prompt_embeds=prompt_embeds,     # 语义路径：QwenVL embeddings
        ...
    )
```

## 使用场景

### 1. 纯文本生成（仅语义路径）
```python
result = worker.generate_images(
    embedding_output=qwenvl_output,  # 包含文本 embeddings
    # input_images=None (自动)
)
```

### 2. 图像编辑（双路径）
```python
# QwenVL 处理：文本 + 图像 → 语义 embeddings + 原始图像
embedding_output = worker.extract_embeddings(scheduler_output)
# embedding_output.input_images 包含原始图像

# Qwen-Image 生成：
# - 语义路径：使用 embeddings
# - 视觉路径：使用原始图像（自动 VAE 编码）
result = worker.generate_images(embedding_output)
```

### 3. 风格迁移
```python
# 输入：参考图像 + "生成类似风格的图像" 文本
# 语义路径：理解文本指令
# 视觉路径：提取参考图像的风格特征
result = worker.generate_images(embedding_output)
```

## 数据流图

```
输入: Text + Image
        │
        ├─────────────────────────┬──────────────────────┐
        │                         │                      │
        v                         v                      v
   Qwen2.5-VL              Qwen2.5-VL            VAE Encoder
   (文本编码)              (图像编码)             (视觉特征)
        │                         │                      │
        └──────────┬──────────────┘                      │
                   v                                     v
            Semantic Embeddings                  Image Latents
            (prompt_embeds)                   (image_latents)
                   │                                     │
                   └──────────────┬──────────────────────┘
                                  v
                          Qwen-Image DiT
                          (Transformer)
                                  │
                                  v
                          Generated Image
```

## 技术优势

1. **语义保真**: 通过 QwenVL embeddings 准确理解文本指令
2. **视觉一致性**: 通过 VAE latents 保持原图的视觉风格
3. **灵活控制**: 可分别调节两条路径的权重
4. **高效推理**: 复用 vLLM 的批处理和优化能力

## 注意事项

1. **模型要求**: 需要使用 `QwenImageEditPipeline` 而非 `QwenImagePipeline`
2. **图像格式**: 支持 PIL Image 或 torch.Tensor
3. **内存占用**: 双路径会增加内存使用，建议启用 VAE tiling
4. **批处理**: 当前实现支持单图像或同构批次

## 文件修改清单

- ✅ `gpu_qwen_vl_model_runner.py`: 添加 `input_images` 提取
- ✅ `gpu_qwenimage_model_runner.py`: 添加 VAE 编码支持
- ✅ `gpu_qwenvl_qwenimage_worker.py`: 实现双路径调用
- ✅ `README_DUAL_PATH.md`: 本文档

## 参考资料

- [Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324)
- [QwenImageEditPipeline Documentation](https://huggingface.co/docs/diffusers/api/pipelines/qwenimage)
- [Diffusers QwenImage Implementation](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/qwenimage)
