import json
import os

from PIL import Image

from src.grounding_model import UIGroundingModel
from src.utils import add_visual_grid, draw_bbox


def run_evaluation():
    # 初始化
    os.makedirs("output", exist_ok=True)
    
    # 获取 API Key (优先从环境变量读取)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    # 初始化模型：使用 API 模式
    # 可选模型：'qwen-vl-max', 'qwen-vl-plus'
    model = UIGroundingModel(
        mode="api", 
        model_path="qwen-vl-max", 
        api_key=api_key
    )
    
    if not api_key:
        print("警告: 未检测到 DASHSCOPE_API_KEY 环境变量，请确保已设置或在代码中手动填入。")
        # 如果需要演示，可以切回 mock 模式
        # model.mode = "mock"
    
    with open("data/dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"开始评估，共 {len(dataset)} 个任务...")
    
    results = []
    for item in dataset:
        img_id = item["id"]
        img_path = item["image_path"]
        instruction = item["instruction"]
        gt_bbox = item["bbox"]

        print(f"\n任务 {img_id}: {instruction}")
        
        # 1. 图像预处理 (推理增强：添加视觉网格)
        raw_img = Image.open(img_path)
        # grid_img = add_visual_grid(raw_img.copy())
        # grid_img.save(f"output/{img_id}_grid.png")

        # 2. 模型推理
        thought, pred_bbox = model.predict(img_path, instruction)
        
        print(f"思考过程: {thought}")
        print(f"预测 BBox: {pred_bbox}")
        print(f"真实 BBox: {gt_bbox}")

        # 3. 结果可视化
        result_img = draw_bbox(raw_img.copy(), pred_bbox, label="Pred")
        result_img = draw_bbox(result_img, gt_bbox, label="GT") # 红色为预测，绿色(这里代码简化都用红色了)为真实
        
        save_path = f"output/result_{img_id}.png"
        result_img.save(save_path)
        
        results.append({
            "id": img_id,
            "instruction": instruction,
            "pred_bbox": pred_bbox,
            "gt_bbox": gt_bbox,
            "thought": thought
        })

    # 保存结果
    with open("output/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print("\n评估完成，结果已保存至 output/ 目录。")

if __name__ == "__main__":
    run_evaluation()

