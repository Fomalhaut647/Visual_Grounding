import json
import os
import re

import torch
from PIL import Image

# 假设使用 Qwen2-VL 或类似的本地模型，如果没有则提供 Mock
try:
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class UIGroundingModel:
    def __init__(self, model_path="qwen-vl-max", mode="mock", api_key=None):
        self.mode = mode
        self.model_path = model_path
        self.api_key = api_key
        
        if mode == "local":
            if HAS_TRANSFORMERS:
                print(f"Loading local model from {model_path}...")
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype="auto", device_map="auto"
                )
                self.processor = AutoProcessor.from_pretrained(model_path)
            else:
                print("Error: transformers or qwen_vl_utils not installed. Falling back to mock.")
                self.mode = "mock"
        elif mode == "api":
            import dashscope
            self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
            dashscope.api_key = self.api_key
            print(f"Using Qwen API Mode (Model: {model_path})")
        else:
            print("Running in MOCK mode (no model loaded)")

    def _build_prompt(self, instruction):
        """
        推理增强：通过精心设计的 System Prompt 和 CoT 模板。
        """
        prompt = f"""你是一个专业的 UI 界面分析专家。
请根据提供的界面截图和指令，准确定位目标元素。

### 推理步骤：
1. **分析布局**：观察截图中的各个组件及其相对位置。
2. **寻找目标**：根据指令 '{instruction}'，识别最匹配的 UI 元素（如按钮、图标、输入框等）。
3. **确定坐标**：计算该元素的归一化矩形边框 [xmin, ymin, xmax, ymax]，其中所有值在 [0, 1] 之间。

### 输出格式：
Thought: <你的思考过程>
BBox: [xmin, ymin, xmax, ymax]

注意：请严格遵守输出格式。"""
        return prompt

    def predict(self, image_path, instruction):
        if self.mode == "mock":
            # 模拟推理过程
            return self._mock_predict(image_path, instruction)
        elif self.mode == "api":
            # 调用 API
            return self._api_predict(image_path, instruction)
        
        # 真实本地模型推理代码
        prompt = self._build_prompt(instruction)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return self._parse_response(response)

    def _mock_predict(self, image_path, instruction):
        """
        模拟输出，用于演示项目结构。
        """
        # 简单的规则模拟，实际项目中应使用大模型
        with open("data/dataset.json", "r") as f:
            dataset = json.load(f)
        
        for item in dataset:
            if item["image_path"] == image_path:
                thought = f"在图片中找到了与指令 '{instruction}' 相关的元素 '{item['target']}'。"
                bbox = item["bbox"]
                return thought, bbox
        
        return "无法识别目标元素。", [0, 0, 0, 0]

    def _api_predict(self, image_path, instruction):
        """
        调用 DashScope MultiModalConversation API。
        """
        import os

        from dashscope import MultiModalConversation

        prompt = self._build_prompt(instruction)
        # DashScope 支持本地文件路径前缀 file://
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"file://{os.path.abspath(image_path)}"},
                    {"text": prompt}
                ]
            }
        ]
        
        try:
            response = MultiModalConversation.call(model=self.model_path, messages=messages)
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                # 如果返回的是列表（某些版本 SDK），提取文本
                if isinstance(content, list):
                    content = next((item['text'] for item in content if 'text' in item), "")
                return self._parse_response(content)
            else:
                error_msg = f"API Error: {response.code} - {response.message}"
                print(error_msg)
                return error_msg, [0, 0, 0, 0]
        except Exception as e:
            error_msg = f"API Exception: {str(e)}"
            print(error_msg)
            return error_msg, [0, 0, 0, 0]

    def _parse_response(self, response):
        """
        从模型返回的文本中解析出 bbox。
        """
        try:
            thought = re.search(r"Thought: (.*)", response, re.S).group(1).split("BBox:")[0].strip()
            bbox_str = re.search(r"BBox: \[(.*)\]", response).group(1)
            bbox = [float(x.strip()) for x in bbox_str.split(",")]
            return thought, bbox
        except Exception as e:
            print(f"解析失败: {e}, 原始输出: {response}")
            return "解析失败", [0, 0, 0, 0]

