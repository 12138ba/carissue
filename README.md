# 车型分类 - 任务一

本项目对应任务书的“任务一：数据分析与视觉模型训练”。它训练 `50 个车型 + unknown` 共 51 类分类器，输出验证集 Macro-F1，并支持单图与批量推理。

## 目录约定

数据集根目录应包含以下内容（`train`、`val` 内的类别目录为四位 ID）：

```text
dataset_root/
├── class_info.json
├── train/0000/*.jpg
└── val/0000/*.jpg
```

`classes.txt` 是本组固定的 50 类清单，按升序排列，`unknown` 不写入其中。首次运行前请核对其中 ID 均存在于教师提供的 `class_info.json`；若要重新选类，修改该文件并保持恰好 50 行、四位数字且升序。

## 安装与执行

```bash
python -m pip install -r requirements.txt

# 从未入选车型中为 train/val 各采样，生成 unknown 目录（只需运行一次）
python prepare_unknown.py --data-root /path/to/dataset_root --per-source-class 20

# 训练并以验证集 Macro-F1 保存最佳权重
python train.py --data-root /path/to/dataset_root --epochs 15 --batch-size 32

# 单张图像推理
python predict.py single --checkpoint checkpoints/best.pt --image sample.jpg

# 对教师指定测试目录生成提交文件
python predict.py batch --checkpoint checkpoints/best.pt --input-dir /path/to/test --output predictions.csv
```

训练模型和 checkpoint 不应提交。`predictions.csv` 使用 `image_id,predicted_label,confidence` 表头，标签为四位 ID 或小写 `unknown`。

## unknown 构造规则

`prepare_unknown.py` 仅从不在 `classes.txt` 内的类别采样；对 `train` 和 `val` 分别处理，故不会跨划分重复，也不会把选中 50 类标为 unknown。默认每个来源类别最多 20 张，来源均衡。请在报告记录实际来源类别数、样本数和该参数。

## 输出

训练过程会保存 `checkpoints/last.pt` 和按验证集 Macro-F1 选择的 `checkpoints/best.pt`；checkpoint 内包含类别顺序、预处理尺寸、模型名和训练配置，以确保 Web/批量推理使用同一映射。
