"""Create balanced unknown folders from classes outside classes.txt."""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from car_data import UNKNOWN, image_files, read_classes, read_class_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--classes", type=Path, default=Path(__file__).with_name("classes.txt"))
    parser.add_argument("--per-source-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true", help="允许 unknown 目录已存在；同名文件会跳过")
    args = parser.parse_args()
    selected = set(read_classes(args.classes))
    class_info = read_class_info(args.data_root)
    available = set(class_info)
    if not selected <= available:
        raise ValueError(f"classes.txt 中存在 class_info.json 未登记的 ID：{sorted(selected - available)}")
    rng = random.Random(args.seed)
    for split in ("train", "val"):
        source_root = args.data_root / split
        target = source_root / UNKNOWN
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"{target} 已存在；确认后使用 --overwrite")
        target.mkdir(parents=True, exist_ok=True)
        copied = 0
        sources = 0
        for source_id in sorted(available - selected):
            files = image_files(source_root / source_id)
            if not files:
                continue
            rng.shuffle(files)
            sources += 1
            for source in files[: args.per_source_class]:
                destination = target / f"{source_id}_{source.name}"
                if not destination.exists():
                    shutil.copy2(source, destination)
                    copied += 1
        print(f"{split}: {sources} 个来源类别，新增 {copied} 张 unknown 图片 -> {target}")


if __name__ == "__main__":
    main()
