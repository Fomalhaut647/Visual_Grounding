# UI Grounding 基于多模态大模型的界面元素定位

本项目实现了一个基于多模态大模型（如 Qwen2-VL）的零训练 UI 界面元素定位方案。通过提供界面截图和自然语言指令，模型能够自动识别并定位目标 UI 元素的坐标。

## 核心特性
- **零训练 (Zero-shot)**: 直接利用预训练多模态大模型的能力，无需针对特定 UI 进行微调。
- **多模式支持**:
  - **API 模式**: 通过阿里云 DashScope 调用 Qwen-VL 系列大模型（推荐，速度快且无需高性能显卡）。
  - **本地模式**: 支持本地加载 Qwen2-VL 模型权重（需要较多显存）。
  - **Mock 模式**: 用于在没有网络或显卡的情况下演示项目结构和流程。
- **推理增强 (Inference Enhancement)**:
  - **CoT (Chain-of-Thought)**: 通过系统提示词引导模型进行思考，分析布局后再输出坐标。
  - **格式约束**: 严格要求输出归一化坐标 `[xmin, ymin, xmax, ymax]`，并能自动处理像素坐标与归一化坐标的转换。

## 快速开始

### 1. 环境准备

安装所需的 Python 依赖：
```bash
pip install pillow torch transformers qwen_vl_utils dashscope datasets
```

### 2. 配置 API Key (仅 API 模式)

如果您打算使用 API 模式（默认），需要获取 [DashScope API Key](https://dashscope.console.aliyun.com/apiKey)。

建议将其设置为环境变量：
```bash
export DASHSCOPE_API_KEY="您的_API_KEY_在这里"
```

### 3. 准备数据 (可选)

项目内置了简单的模拟数据集。您也可以获取更真实的 Mind2Web 数据：
```bash
python3 data/load_mind2web.py
```
*注意：Mind2Web 包含真实的网页截图和操作指令，非常适合测试模型在复杂场景下的表现。*

### 4. 运行推理

运行主评估脚本：
```bash
python3 main.py
```

## 项目结构
- `data/`: 包含数据集生成和加载脚本。
  - `images/`: 存放测试用例的截图。
  - `dataset.json`: 记录测试用例的指令和真实 BBox。
- `src/`: 核心逻辑代码。
  - `grounding_model.py`: 模型推理封装，包含 API 调用和本地推理。
  - `utils.py`: 图像处理（如绘制 BBox）和可视化工具。
- `main.py`: 主运行入口，负责加载数据、调用模型并保存结果。
- `output/`: 存放模型推理的 JSON 结果和标注后的可视化图片。

## 效果示例
程序运行结束后，可以在 `output/` 目录下查看结果。
- **红色框 (Pred)**: 模型预测的目标位置。
- **绿色框 (GT)**: 数据集中标注的真实位置（Ground Truth）。

## 数据集说明
- **模拟数据集**: 通过 `data/generate_data.py` 生成，包含简单的 UI 元素定位任务。
- **Multimodal-Mind2Web**: 从 Hugging Face 获取的真实网页 UI 数据集。包含真实的网页截图、操作指令和元素标注。
