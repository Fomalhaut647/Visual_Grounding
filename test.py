import dashscope
from dashscope import MultiModalConversation
import json
import cv2
import numpy as np
import re
import os
from canny import detect_ui_with_canny

# 设置你的 API KEY
dashscope.api_key = 'sk-377fabbff2084817b5ac0cd6328b8760'
def parse_element_id(model_text):# 使用正则表达式提取数字
    match = re.search(r'\b(\d+)\b', model_text)
    if match:
        return int(match.group(1))
    return None

def call_qwen_vl_grounding(image_path, element_description):
    """
    image_path: 本地图片路径或 URL
    element_description: 用自然语言描述，如 "I want to submit the form"
    """
    raw_path = image_path
    if raw_path.startswith("file://"):
        raw_path = raw_path[7:]
    
    # canny 预处理，获取边界框
    boxes, output_img = detect_ui_with_canny(raw_path)
    if not boxes:
        print("未检测到任何有效元素")
        return None
    cv2.imwrite("canny_processed.png", output_img)
    api_image_full_path = "file://" + os.path.abspath("canny_processed.png")
    # 构造 Prompt
    # 技巧：明确告诉模型你需要 Detect（检测）或者找出 Bounding Box
    prompt = (
        f"I have detected UI elements and marked them with green boxes and ID numbers on this image. "
        f"Please follow the instruction described as '{element_description}'. "
        f"Output ONLY the ID number of that element."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"image": api_image_full_path}, # 支持本地路径 file://... 或 http:// URL
                {"text": prompt}
            ]
        }
    ]

    response = MultiModalConversation.call(
        model='qwen2.5-vl-7b-instruct',
        messages=messages
    )

    if response.status_code != 200:
        print(f"Error: {response.code} - {response.message}")
        return None
    # 解析响应
    model_text = response.output.choices[0].message['content']
    print("模型回复内容:", model_text)
    
    if isinstance(model_text, list):
        new_model_text = ""
        for item in model_text:
            if isinstance(item, dict) and "text" in item:
                new_model_text += item['text']
    else:
        new_model_text = str(model_text)
    print(new_model_text)
    
    target_id = parse_element_id(new_model_text)
    if target_id is None:
        print("未能从模型回复中提取到元素 ID")
        return None
    index = target_id - 1  # ID 从 1 开始，列表索引从 0 开始
    if index < 0 or index >= len(boxes):
        print("提取到的元素 ID 超出范围")
        return None
    target_box = boxes[index]
    x_min, y_min = target_box[0], target_box[1]
    x_max, y_max = x_min + target_box[2], y_min + target_box[3]
    return target_id, (x_min, y_min, x_max, y_max)
# 使用示例
result = call_qwen_vl_grounding("test2.png", "I want to find my favorite photos.")
print(result)