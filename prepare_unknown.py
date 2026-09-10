"""Create a class-balanced unknown folder from classes outside classes.txt."""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from car_data import UNKNOWN, image_files, read_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--classes", type=Path, default=Path(__file__).with_name("classes.txt"))
    parser.add_argument("--unknown-count", type=int, default=None, help="每个划分中 unknown 的总图片数；未指定时取已选车型的平均样本数。")
    parser.add_argument("--unknown-ratio", type=float, default=1.0, help="unknown 总数相对每个已选车型平均样本数的比例（默认 1.0）。")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reset", action="store_true", help="删除并重建已有 unknown 目录；仅在确认该目录是本脚本生成时使用。")
    args = parser.parse_args()
    selected = set(read_classes(args.classes))
    if args.unknown_ratio <= 0:
        raise ValueError("--unknown-ratio 必须大于 0")
    if args.unknown_count is not None and args.unknown_count <= 0:
        raise ValueError("--unknown-count 必须大于 0")
    rng = random.Random(args.seed)
    for split in ("train", "val"):
        source_root = args.data_root / split
        if not source_root.is_dir():
            raise FileNotFoundError(f"找不到数据划分目录：{source_root}")
        target = source_root / UNKNOWN
        if target.exists() and args.reset:
            shutil.rmtree(target)
        elif target.exists() and image_files(target):
            raise FileExistsError(f"{target} 已有图片；确认后使用 --reset 重建")
        target.mkdir(parents=True, exist_ok=True)
        selected_counts = [len(image_files(source_root / label)) for label in sorted(selected)]
        if not all(selected_counts):
            missing = [label for label, count in zip(sorted(selected), selected_counts) if count == 0]
            raise FileNotFoundError(f"{split} 缺少已选类别图片：{missing}")
        average_known = sum(selected_counts) / len(selected_counts)
        target_count = args.unknown_count or max(1, round(average_known * args.unknown_ratio))
        pools = {
            directory.name: image_files(directory)
            for directory in source_root.iterdir()
            if directory.is_dir() and directory.name not in selected and directory.name != UNKNOWN
        }
        pools = {source_id: files for source_id, files in pools.items() if files}
        if not pools:
            raise ValueError(f"{split} 中没有可用于 unknown 的组外类别")
        for files in pools.values():
            rng.shuffle(files)
        copied = 0
        used_sources: set[str] = set()
        while copied < target_count and pools:
            source_ids = list(pools)
            rng.shuffle(source_ids)
            for source_id in source_ids:
                source = pools[source_id].pop()
                destination = target / f"{source_id}_{source.name}"
                shutil.copy2(source, destination)
                copied += 1
                used_sources.add(source_id)
                if not pools[source_id]:
                    del pools[source_id]
                if copied == target_count:
                    break
        print(f"{split}: unknown {copied} 张，来自 {len(used_sources)} 个组外类别；已选车型平均 {average_known:.1f} 张 -> {target}")


if __name__ == "__main__":
    main()
