from __future__ import annotations
import argparse
from pathlib import Path
import shutil

def parse_args():
    p = argparse.ArgumentParser(description="Prepare Tiny-ImageNet val split into class folders.")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Path to tiny-imagenet-200 (folder containing train/ and val/)")
    p.add_argument("--dest-name", type=str, default="val_by_class",
                   help="Name of the new class-split val folder to create")
    p.add_argument("--copy", action="store_true",
                   help="Copy files (default). If omitted, files are moved.")
    return p.parse_args()

def main():
    args = parse_args()
    root = args.data_dir
    val_dir = root / "val"
    images_dir = val_dir / "images"
    anno_file = val_dir / "val_annotations.txt"
    out_dir = root / args.dest_name

    if not images_dir.exists() or not anno_file.exists():
        raise SystemExit(f"Expected {images_dir} and {anno_file} — check --data-dir")

    if out_dir.exists():
        print(f"[info] Destination exists: {out_dir}. New files may overwrite existing ones.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read mapping: filename, class_id, x1, y1, x2, y2 (we only need filename->class_id)
    mapping = {}
    with anno_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts or len(parts) < 2:  # defensive
                parts = line.strip().split()  # some mirrors are space-separated
            if len(parts) >= 2:
                fname, cls = parts[0], parts[1]
                mapping[fname] = cls

    # Create class dirs
    classes = sorted(set(mapping.values()))
    for cls in classes:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    # Move or copy each image into its class folder
    move_fn = shutil.copy2 if args.copy or not hasattr(shutil, "move") else shutil.move
    n = 0
    for fname, cls in mapping.items():
        src = images_dir / fname
        dst = out_dir / cls / fname
        if not src.exists():
            print(f"[warn] missing file: {src}")
            continue
        move_fn(src, dst)
        n += 1

    print(f"[done] placed {n} files into {out_dir} ({'copied' if args.copy else 'moved'})")
    print("Update your config split.val to:", out_dir.name)

if __name__ == "__main__":
    main()
