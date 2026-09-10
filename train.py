from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from car_data import CarFolderDataset, labels_from_project
from modeling import build_model


def evaluate(model, loader, device):
    model.eval()
    truth, predicted = [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            predicted.extend(logits.argmax(1).cpu().tolist())
            truth.extend(targets.tolist())
    return f1_score(truth, predicted, labels=list(range(51)), average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    labels = labels_from_project(Path(__file__).parent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15, 0.15, 0.1), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    eval_tf = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_set = CarFolderDataset(args.data_root, "train", labels, train_tf)
    val_set = CarFolderDataset(args.data_root, "val", labels, eval_tf)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    model = build_model(51, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    config = vars(args) | {"data_root": str(args.data_root), "checkpoint_dir": str(args.checkpoint_dir)}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
        macro_f1 = evaluate(model, val_loader, device)
        scheduler.step()
        payload = {"model_state": model.state_dict(), "labels": labels, "image_size": args.image_size, "model_name": "resnet18", "epoch": epoch, "val_macro_f1": macro_f1, "config": config}
        torch.save(payload, args.checkpoint_dir / "last.pt")
        if macro_f1 > best:
            best = macro_f1
            torch.save(payload, args.checkpoint_dir / "best.pt")
        print(json.dumps({"epoch": epoch, "train_loss": total_loss / len(train_set), "val_macro_f1": macro_f1, "best": best}, ensure_ascii=False))


if __name__ == "__main__":
    main()
