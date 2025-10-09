"""快速验证脚本：测试双路径架构实现

这个脚本验证以下功能：
1. QwenVLEmbeddingOutput 包含 input_images 字段
2. QwenImageModelRunner 支持 image 参数
3. Worker 正确传递双路径数据
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试所有修改的模块能否正确导入"""
    print("=" * 60)
    print("测试 1: 模块导入")
    print("=" * 60)

    try:
        from vllm_omni.worker.gpu_qwen_vl_model_runner import (
            QwenVLModelRunner,
            QwenVLEmbeddingOutput
        )
        print("✓ gpu_qwen_vl_model_runner 导入成功")

        from vllm_omni.worker.gpu_qwenimage_model_runner import (
            QwenImageModelRunner,
            QwenImageRunnerOutput
        )
        print("✓ gpu_qwenimage_model_runner 导入成功")

        from vllm_omni.worker.gpu_qwenvl_qwenimage_worker import (
            QwenVLToQwenImageWorker,
            QwenVLToImageOutput
        )
        print("✓ gpu_qwenvl_qwenimage_worker 导入成功")

        print("\n所有模块导入成功！")
        return True

    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_dataclass_fields():
    """测试 QwenVLEmbeddingOutput 是否包含 input_images 字段"""
    print("\n" + "=" * 60)
    print("测试 2: QwenVLEmbeddingOutput 字段")
    print("=" * 60)

    try:
        from vllm_omni.worker.gpu_qwen_vl_model_runner import QwenVLEmbeddingOutput
        from dataclasses import fields

        field_names = [f.name for f in fields(QwenVLEmbeddingOutput)]
        print(f"QwenVLEmbeddingOutput 字段: {field_names}")

        if 'input_images' in field_names:
            print("✓ input_images 字段已添加")
            return True
        else:
            print("✗ input_images 字段缺失")
            return False

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_method_signatures():
    """测试方法签名是否正确"""
    print("\n" + "=" * 60)
    print("测试 3: 方法签名")
    print("=" * 60)

    try:
        from vllm_omni.worker.gpu_qwenimage_model_runner import QwenImageModelRunner
        import inspect

        # 检查 generate 方法签名
        sig = inspect.signature(QwenImageModelRunner.generate)
        params = list(sig.parameters.keys())

        print(f"QwenImageModelRunner.generate 参数:")
        for i, param in enumerate(params, 1):
            print(f"  {i}. {param}")

        required_params = ['image', 'image_latents', 'prompt_embeds']
        missing_params = [p for p in required_params if p not in params]

        if not missing_params:
            print(f"\n✓ 所有必需参数已添加: {required_params}")
            return True
        else:
            print(f"\n✗ 缺失参数: {missing_params}")
            return False

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_worker_dual_path_support():
    """测试 Worker 是否支持双路径"""
    print("\n" + "=" * 60)
    print("测试 4: Worker 双路径支持")
    print("=" * 60)

    try:
        from vllm_omni.worker.gpu_qwenvl_qwenimage_worker import QwenVLToQwenImageWorker
        import inspect

        # 检查类文档字符串
        docstring = QwenVLToQwenImageWorker.__doc__

        if docstring and "Dual-Path" in docstring:
            print("✓ Worker 文档包含双路径说明")
        else:
            print("✗ Worker 文档缺少双路径说明")

        # 检查 generate_images 方法
        source = inspect.getsource(QwenVLToQwenImageWorker.generate_images)

        if "input_images" in source:
            print("✓ generate_images 方法使用 input_images")
        else:
            print("✗ generate_images 方法未使用 input_images")

        if "image=" in source:
            print("✓ generate_images 传递 image 参数")
            return True
        else:
            print("✗ generate_images 未传递 image 参数")
            return False

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_vae_encoding_methods():
    """测试 VAE 编码方法是否存在"""
    print("\n" + "=" * 60)
    print("测试 5: VAE 编码方法")
    print("=" * 60)

    try:
        from vllm_omni.worker.gpu_qwenimage_model_runner import QwenImageModelRunner

        required_methods = ['_encode_vae_image', '_retrieve_latents']

        for method_name in required_methods:
            if hasattr(QwenImageModelRunner, method_name):
                print(f"✓ {method_name} 方法存在")
            else:
                print(f"✗ {method_name} 方法缺失")
                return False

        print("\n✓ 所有 VAE 编码方法已实现")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def print_summary(results):
    """打印测试总结"""
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    total = len(results)
    passed = sum(results.values())

    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！双路径架构实现成功！")
        print("\n下一步:")
        print("1. 查看 README_DUAL_PATH.md 了解详细架构")
        print("2. 运行 example_qwenvl_qwenimage_usage.py 查看使用示例")
        print("3. 使用真实模型测试双路径功能")
    else:
        print("\n⚠️ 部分测试失败，请检查实现")


if __name__ == "__main__":
    print("Qwen-Image 双路径架构实现验证")
    print("=" * 60)

    results = {
        "模块导入": test_imports(),
        "数据类字段": test_dataclass_fields(),
        "方法签名": test_method_signatures(),
        "Worker 双路径": test_worker_dual_path_support(),
        "VAE 编码": test_vae_encoding_methods(),
    }

    print_summary(results)
