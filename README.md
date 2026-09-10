# 车型分类 - 任务一

本项目对应任务书的“任务一：数据分析与视觉模型训练”。它训练 50 个车型 + unknown 共 51 类分类器，输出验证集 Macro-F1，并支持单图与批量推理。

## 目录约定

数据集根目录应包含 class_info.json、train/ 和 val/；train、val 下的类别目录为四位 ID。

classes.txt 是本组固定的 50 类清单，按升序排列；unknown 不写入其中。

## 安装与执行

```bash
python -m pip install -r requirements.txt

# 从未入选车型中均衡采样 unknown；总量约等于每个已选车型的平均样本数
# --reset 仅删除 train/unknown 与 val/unknown，请确认它们是旧脚本生成的文件
python prepare_unknown.py --data-root "D:\project\数据集" --unknown-ratio 1.0 --reset

# 训练并以验证集 Macro-F1 保存最佳权重
python train.py --data-root "D:\project\数据集" --epochs 15 --batch-size 32

# 单张图像推理（--image 必须是一张实际图片，不能写 *.jpg）
python predict.py single --checkpoint checkpoints/best.pt --image "D:\project\数据集\val\0000\0000_0006.jpg"

# 对教师指定测试目录生成提交文件
python predict.py batch --checkpoint checkpoints/best.pt --input-dir /path/to/test --output predictions.csv
```

训练模型和 checkpoint 不应提交。predictions.csv 使用 image_id,predicted_label,confidence 表头，标签为四位 ID 或小写 unknown。

## unknown 构造规则

prepare_unknown.py 仅从不在 classes.txt 内的类别采样；对 train 和 val 分别处理，故不会跨划分重复，也不会把选中 50 类标成 unknown。默认的 unknown 总数等于每个已选车型的平均样本数，且从多个来源类别轮换抽样，避免 unknown 成为压倒性的多数类。可用 --unknown-count 指定总数，或用 --unknown-ratio 调整比例。请在报告记录实际来源类别数、样本数和参数。

## 输出

训练过程会保存 checkpoints/last.pt 和按验证集 Macro-F1 选择的 checkpoints/best.pt；checkpoint 内包含类别顺序、预处理尺寸、模型名和训练配置，以确保 Web/批量推理使用同一映射。
