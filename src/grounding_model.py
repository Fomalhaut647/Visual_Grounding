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
            self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or 'sk-377fabbff2084817b5ac0cd6328b8760'
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
        # 获取图像尺寸以便后续可能的强制归一化
        with Image.open(image_path) as img:
            width, height = img.size

        if self.mode == "mock":
            # 模拟推理过程
            return self._mock_predict(image_path, instruction)
        elif self.mode == "api":
            # 调用 API
            return self._api_predict(image_path, instruction, width, height)
        
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
        
        return self._parse_response(response, width, height)

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

    def _api_predict(self, image_path, instruction, width, height):
        """
        调用 DashScope MultiModalConversation API。
        """
        import os

        from dashscope import MultiModalConversation

        prompt = self._build_prompt(instruction)
        # DashScope 支持本地 file:// 协议，需使用绝对路径
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
                # 如果返回的是列表，提取文本内容
                if isinstance(content, list):
                    content = next((item['text'] for item in content if 'text' in item), "")
                return self._parse_response(content, width, height)
            else:
                error_msg = f"API Error: {response.code} - {response.message}"
                print(error_msg)
                return error_msg, [0, 0, 0, 0]
        except Exception as e:
            error_msg = f"API Exception: {str(e)}"
            print(error_msg)
            return error_msg, [0, 0, 0, 0]

    def _parse_response(self, response, width, height):
        """
        从模型返回的文本中解析出 bbox。
        如果模型输出的是像素坐标，则根据 width 和 height 进行自动归一化。
        """
        FLOAT_RE = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"

        def _normalize_bbox_if_needed(bbox):
            # 自动纠正：如果任意坐标值大于 1.1，判定为像素坐标并强制归一化
            if any(v > 1.1 for v in bbox):
                bbox = [
                    bbox[0] / width,
                    bbox[1] / height,
                    bbox[2] / width,
                    bbox[3] / height,
                ]
            # 裁剪到合法范围 [0, 1]
            bbox = [max(0.0, min(1.0, float(v))) for v in bbox]
            # 统一保留 3 位小数（便于可读、也与现有逻辑一致）
            return [round(v, 3) for v in bbox]

        def _is_valid_bbox(bbox):
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return False
            x1, y1, x2, y2 = bbox
            if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
                return False
            # 必须是正面积矩形
            return (x2 > x1) and (y2 > y1)

        def _extract_candidate_bboxes(text):
            """
            返回候选 bbox 列表（按出现顺序）。之所以保留多个候选，是因为模型有时会先给一个尝试，
            后面再自我修正输出第二个/第三个 bbox；取最后一个合法 bbox 更稳健。
            """
            candidates = []

            # 1) 优先解析显式的 BBox: [...]（支持中文冒号）
            for m in re.finditer(r"BBox\s*[:：]\s*\[([^\]]+)\]", text, flags=re.I):
                nums = re.findall(FLOAT_RE, m.group(1))
                if len(nums) >= 4:
                    try:
                        bbox = [float(n) for n in nums[:4]]
                        candidates.append(bbox)
                    except Exception:
                        pass

            # 2) 若没找到 BBox 标记，再尝试全文中所有 [...]，取其中恰好 4 个数字的
            if not candidates:
                for m in re.finditer(r"\[([^\]]+)\]", text):
                    nums = re.findall(FLOAT_RE, m.group(1))
                    if len(nums) == 4:
                        try:
                            bbox = [float(n) for n in nums]
                            candidates.append(bbox)
                        except Exception:
                            pass

            return candidates

        try:
            # 提取 Thought
            thought_match = re.search(r"Thought\s*[:：]\s*(.*)", response, re.S | re.I)
            thought = thought_match.group(1).split("BBox:")[0].strip() if thought_match else ""
            
            # 提取并选择 bbox（取最后一个合法候选，避免“先错后改”被误解析）
            candidates = _extract_candidate_bboxes(response)
            for raw_bbox in reversed(candidates):
                bbox = _normalize_bbox_if_needed(raw_bbox)
                if _is_valid_bbox(bbox):
                    return thought, bbox

            return thought, [0, 0, 0, 0]
        except Exception as e:
            print(f"解析失败: {e}, 原始输出: {response}")
            return "解析失败", [0, 0, 0, 0]

