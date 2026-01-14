import json
import os

import requests


def download_images():
    dataset = [
        {
            "id": "1",
            "url": "https://github.com/google-research-datasets/screen2words/raw/master/images/100.png",
            "instruction": "点击底部的'Home'图标",
            "bbox": [0.0, 0.9, 0.2, 1.0]
        },
        {
            "id": "2",
            "url": "https://github.com/google-research-datasets/screen2words/raw/master/images/101.png",
            "instruction": "查找屏幕顶部的搜索框",
            "bbox": [0.1, 0.05, 0.9, 0.15]
        },
        {
            "id": "3",
            "url": "https://github.com/google-research-datasets/screen2words/raw/master/images/102.png",
            "instruction": "点击右上角的设置按钮",
            "bbox": [0.8, 0.0, 1.0, 0.1]
        }
    ]

    os.makedirs("data/images", exist_ok=True)
    
    for item in dataset:
        img_path = f"data/images/{item['id']}.png"
        print(f"Downloading {item['url']} to {img_path}...")
        try:
            response = requests.get(item['url'], timeout=10)
            if response.status_code == 200:
                with open(img_path, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {item['id']}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error downloading {item['id']}: {e}")

    with open("data/dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    download_images()

