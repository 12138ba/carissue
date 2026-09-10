from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from car_data import IMAGE_SUFFIXES, UNKNOWN
from modeling import build_model


def load_predictor(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = checkpoint["labels"]
    if len(labels) != 51 or labels[-1] != UNKNOWN:
        raise ValueError("checkpoint 的类别映射不符合 50 + unknown 要求")
    model = build_model(len(labels), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    size = checkpoint["image_size"]
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return model, labels, transform


def infer(model, labels, transform, image_path: Path):
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax())
    return labels[index], float(probabilities[index])


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("single")
    one.add_argument("--checkpoint", type=Path, required=True)
    one.add_argument("--image", type=Path, required=True)
    batch = sub.add_parser("batch")
    batch.add_argument("--checkpoint", type=Path, required=True)
    batch.add_argument("--input-dir", type=Path, required=True)
    batch.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()
    model, labels, transform = load_predictor(args.checkpoint)
    if args.command == "single":
        label, confidence = infer(model, labels, transform, args.image)
        print({"predicted_label": label, "confidence": confidence})
        return
    images = sorted(p for p in args.input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError("指定测试目录中未找到图像")
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "predicted_label", "confidence"])
        for image in images:
            label, confidence = infer(model, labels, transform, image)
            writer.writerow([image.name, label, f"{confidence:.6f}"])
    print(f"已生成 {args.output}，共 {len(images)} 条预测")


if __name__ == "__main__":
    main()
