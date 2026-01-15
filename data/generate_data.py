import json
import os

from PIL import Image, ImageDraw, ImageFont


def generate_mock_ui():
    os.makedirs("data/images", exist_ok=True)
    
    # 模拟数据集定义
    dataset = [
        {
            "id": "1",
            "image_path": "data/images/1.png",
            "instruction": "点击底部的 'Confirm' 按钮",
            "target": "Confirm 按钮",
            "bbox": [0.35, 0.85, 0.65, 0.95] # xmin, ymin, xmax, ymax
        },
        {
            "id": "2",
            "image_path": "data/images/2.png",
            "instruction": "找到顶部的搜索输入框",
            "target": "搜索输入框",
            "bbox": [0.1, 0.05, 0.9, 0.15]
        },
        {
            "id": "3",
            "image_path": "data/images/3.png",
            "instruction": "点击右上角的 'Settings' 图标",
            "target": "Settings 图标",
            "bbox": [0.8, 0.02, 0.98, 0.12]
        }
    ]

    # 生成模拟图片
    for item in dataset:
        width, height = 500, 800
        img = Image.new('RGB', (width, height), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # 绘制背景装饰
        draw.rectangle([0, 0, width, 50], fill=(100, 100, 255)) # 状态栏
        
        # 根据 bbox 绘制目标元素
        xmin, ymin, xmax, ymax = item['bbox']
        real_bbox = [xmin * width, ymin * height, xmax * width, ymax * height]
        
        if "按钮" in item['target']:
            draw.rectangle(real_bbox, fill=(255, 100, 100), outline=(0, 0, 0))
            draw.text((real_bbox[0]+10, real_bbox[1]+10), "Confirm", fill=(255, 255, 255))
        elif "输入框" in item['target']:
            draw.rectangle(real_bbox, fill=(255, 255, 255), outline=(0, 0, 0))
            draw.text((real_bbox[0]+10, real_bbox[1]+10), "Search...", fill=(150, 150, 150))
        elif "图标" in item['target']:
            draw.ellipse(real_bbox, fill=(100, 255, 100), outline=(0, 0, 0))
            draw.text((real_bbox[0]+5, real_bbox[1]+5), "⚙", fill=(0, 0, 0))

        img.save(item['image_path'])
        print(f"Generated {item['image_path']}")

    with open("data/dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    generate_mock_ui()


