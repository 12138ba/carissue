"""Dataset, class-list and metadata utilities for the task-one classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UNKNOWN = "unknown"


def read_classes(path: str | Path) -> list[str]:
    labels = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 50 or len(set(labels)) != 50:
        raise ValueError("classes.txt 必须恰好包含 50 个不重复的类别 ID")
    if labels != sorted(labels) or any(not (len(x) == 4 and x.isdigit()) for x in labels):
        raise ValueError("classes.txt 必须是升序的四位数字 ID，每行一个")
    return labels


def labels_from_project(project_dir: str | Path) -> list[str]:
    return read_classes(Path(project_dir) / "classes.txt") + [UNKNOWN]


def read_class_info(data_root: str | Path) -> dict[str, str]:
    path = Path(data_root) / "class_info.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.values() if isinstance(raw, dict) else raw
    result: dict[str, str] = {}
    for item in records:
        if isinstance(item, dict) and "id" in item:
            result[str(item["id"]).zfill(4)] = str(item.get("name_from_new", item["id"]))
    return result


def image_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


class CarFolderDataset(Dataset):
    def __init__(self, data_root: str | Path, split: str, labels: Iterable[str], transform=None):
        self.root = Path(data_root) / split
        self.labels = list(labels)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        missing = []
        for index, label in enumerate(self.labels):
            directory = self.root / label
            files = image_files(directory) if directory.is_dir() else []
            if not files:
                missing.append(str(directory))
            self.samples.extend((file, index) for file in files)
        if missing:
            raise FileNotFoundError("下列类别目录不存在或没有图片：\n" + "\n".join(missing))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image) if self.transform else image, target
