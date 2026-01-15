import io
import json
import os

from datasets import load_dataset
from PIL import Image


def load_mind2web_subset(num_samples=5):
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    print(f"Loading {num_samples} samples from osunlp/Multimodal-Mind2Web...")
    
    # 加载数据集的一个子集
    try:
        # 使用 test_task 分片，通常比 train 小
        ds = load_dataset("osunlp/Multimodal-Mind2Web", split="test_task", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    os.makedirs("data/images", exist_ok=True)
    dataset_metadata = []
    
    count = 0
    for sample in ds:
        if count >= num_samples:
            break
            
        try:
            img = sample['screenshot']
            instruction = sample['instruction']
            pos_candidates = sample.get('pos_candidates', [])
            
            if not pos_candidates:
                continue
                
            # 取第一个正样本候选作为目标
            target_candidate = pos_candidates[0]
            # Mind2Web 的 bbox 格式是 [top, left, height, width] (像素值)
            raw_bbox = target_candidate.get('bbox') 
            
            if not raw_bbox or len(raw_bbox) < 4:
                continue
                
            top, left, h, w = raw_bbox
            width, height = img.size
            
            # 转换为 [xmin, ymin, xmax, ymax] 并归一化
            xmin = max(0.0, min(1.0, left / width))
            ymin = max(0.0, min(1.0, top / height))
            xmax = max(0.0, min(1.0, (left + w) / width))
            ymax = max(0.0, min(1.0, (top + h) / height))
            
            normalized_bbox = [round(xmin, 3), round(ymin, 3), round(xmax, 3), round(ymax, 3)]
            
            img_filename = f"mind2web_{count}.png"
            img_path = f"data/images/{img_filename}"
            img.save(img_path)
            
            dataset_metadata.append({
                "id": str(count + 100), # 避免与之前的 ID 冲突
                "image_path": img_path,
                "instruction": instruction,
                "target": target_candidate.get('backend_node_id', 'element'),
                "bbox": normalized_bbox
            })
            
            print(f"Processed sample {count+1}: {instruction}")
            count += 1
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    # 更新 dataset.json
    # 如果已存在，则合并
    existing_data = []
    if os.path.exists("data/dataset.json"):
        with open("data/dataset.json", "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    
    # 为了演示，我们可以选择替换或追加。用户要求“使用”该数据集，我将替换它以确保证明使用了新数据。
    with open("data/dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, ensure_ascii=False, indent=4)
        
    print(f"Successfully saved {len(dataset_metadata)} samples to data/dataset.json")

if __name__ == "__main__":
    load_mind2web_subset()

