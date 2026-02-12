import io as io_module
from typing_extensions import override

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import ComfyExtension, io
from google import genai
from google.genai import types


class Gemini3ImageNode(io.ComfyNode):
    """
    Gemini 3 图像生成节点

    使用 google-generativeai 库调用 Google Gemini 3 API 生成图像
    重要说明:
    1. google-genai 1.60.0 版本已支持 imageSize 参数!
    2. 参数名使用驼峰命名: aspectRatio, imageSize (不是 image_size)
    3. imageSize 支持的值: "1K", "2K", "4K" (必须大写K)
    4. 必须将图片包含在 contents 参数中: contents=[prompt, image]
    """

    @staticmethod
    def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
        """将 ComfyUI 的 IMAGE 格式 (torch.Tensor) 转换为 PIL Image"""
        # ComfyUI IMAGE 格式: [batch, height, width, channels] 或 [height, width, channels]
        # 值范围: 0.0 - 1.0
        if image_tensor.dim() == 4:
            # 如果是批次，取第一张
            image_tensor = image_tensor[0]

        # 转换为 numpy 数组
        image_np = image_tensor.cpu().numpy()

        # 确保值范围在 0-255
        if image_np.max() <= 1.0:
            image_np = (image_np * 255.0).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        # 转换为 PIL Image
        return Image.fromarray(image_np)

    @classmethod
    def define_schema(cls) -> io.Schema:
        """
        返回包含节点全部信息的 schema。
        常见类型："Model", "Vae", "Clip", "Conditioning", "Latent", "Image", "Int", "String", "Float", "Combo"。
        输出使用 "io.Model.Output"，输入使用 "io.Model.Input"。
        类型可以是 "Combo" - 表示下拉选项列表。
        """
        return io.Schema(
            node_id="Gemini3ImageNode",
            display_name="Gemini 3 Image (Google API)",
            category="LLM/Google",
            inputs=[
                io.String.Input("api_key", multiline=False),
                io.String.Input("prompt", multiline=True),
                io.Image.Input("input_image"),
                io.String.Input("model", default="gemini-3-pro-image-preview"),
                io.Combo.Input(
                    "aspect_ratio",
                    options=["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                    default="1:1",
                ),
                io.Combo.Input(
                    "image_size",
                    options=["1K", "2K", "4K"],
                    default="1K",
                ),
            ],
            outputs=[
                io.Image.Output("image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        api_key: str,
        prompt: str,
        input_image: torch.Tensor,
        model: str,
        aspect_ratio: str,
        image_size: str,
    ) -> io.NodeOutput:
        if not api_key or not api_key.strip():
            raise ValueError("api_key 不能为空")
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt 不能为空")

        try:
            # 初始化客户端
            client = genai.Client(
                vertexai=True,
                api_key=api_key,
                http_options=types.HttpOptions(api_version="v1"),
            )

            # 将输入图片转换为 PIL Image
            pil_image = cls._tensor_to_pil(input_image)

            # 调用生成内容 API
            response = client.models.generate_content(
                model=model,
                contents=[prompt, pil_image],  # 必须包含图片
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(
                        aspectRatio=aspect_ratio,  # 宽高比
                        imageSize=image_size,  # ✅ 分辨率参数(1.60.0版本支持)
                    ),
                ),
            )

            # 处理响应结果
            if not response.candidates:
                raise RuntimeError("API 返回了空响应，没有候选结果")

            # 从响应中提取图像数据
            image_bytes = None
            for candidate in response.candidates:
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        # 检查是否有图像数据
                        if hasattr(part, "inline_data") and part.inline_data:
                            image_bytes = part.inline_data.data
                            break
                        # 兼容其他可能的图像数据格式
                        if hasattr(part, "image") and part.image:
                            if hasattr(part.image, "data"):
                                image_bytes = part.image.data
                                break

            if image_bytes is None:
                raise RuntimeError(
                    "Google API 返回了空图像。"
                    "可能的原因：1) API key 无效 2) 模型名称错误 3) 提示词格式不正确。"
                    f"请检查模型名称: {model}"
                )

            # 将图像字节转换为 PIL Image
            generated_image = Image.open(io_module.BytesIO(image_bytes)).convert("RGB")

            # 转换为 ComfyUI 的 IMAGE 格式 (torch.Tensor)
            image_np = np.array(generated_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            return io.NodeOutput(image_tensor)

        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "NOT_FOUND" in error_msg:
                raise RuntimeError(
                    f"模型未找到: {model}。"
                    "请确保模型名称正确，例如: gemini-3-pro-image-preview"
                ) from e
            elif "401" in error_msg or "403" in error_msg or "UNAUTHENTICATED" in error_msg:
                raise RuntimeError("API key 无效或无权访问。请检查 API key 是否正确") from e
            else:
                raise RuntimeError(f"Google API 调用失败: {error_msg}") from e


class Gemini3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Gemini3ImageNode,
        ]


async def comfy_entrypoint() -> Gemini3Extension:  # ComfyUI 会调用该方法来加载扩展及其节点
    return Gemini3Extension()
