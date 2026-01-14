# UI Grounding 基于多模态大模型的界面元素定位

本项目实现了一个基于多模态大模型（如 Qwen2-VL）的零训练 UI 界面元素定位方案。

## 核心特性
- **零训练 (Zero-shot)**: 直接利用预训练多模态大模型的能力。
- **推理增强 (Inference Enhancement)**:
  - **CoT (Chain-of-Thought)**: 通过系统提示词引导模型进行思考，分析布局后再输出坐标。
  - **视觉网格辅助 (Visual Grid)**: (可选) 在图片上叠加辅助网格，增强模型对空间位置的感知。
  - **格式约束**: 严格要求输出归一化坐标 `[xmin, ymin, xmax, ymax]`。

## 项目结构
- `data/`: 包含模拟数据集（图片和标注）。
- `src/`: 核心逻辑代码。
  - `grounding_model.py`: 模型推理封装。
  - `utils.py`: 图像处理和可视化工具。
- `main.py`: 主运行脚本。
- `output/`: 存放推理结果和可视化图片。

## 运行方法
1. 安装依赖：
   ```bash
   pip install pillow torch transformers qwen_vl_utils
   ```
2. 生成数据：
   ```bash
   python data/generate_data.py
   ```
3. 运行推理：
   ```bash
   python main.py
   ```

## 效果示例
程序会输出模型的思考过程并生成带有 BBox 标注的可视化结果。
