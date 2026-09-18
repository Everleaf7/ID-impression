#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STYLE_TOKEN = "idavatar doodle style"
STYLE_SUFFIX = (
    "square avatar icon, rough adult amateur doodle made quickly in a basic paint app, "
    "shaky uneven mouse-drawn outlines, simple flat shapes, sparse centered composition, "
    "large empty background and blank upper margin, minimal accent colors, unpolished, "
    "no letters, no words, no signature, no watermark"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_color(value: str) -> tuple[int, int, int]:
    raw = value.removeprefix("0x")
    if len(raw) != 6:
        raise ValueError(f"invalid background color: {value}")
    return tuple(int(raw[offset:offset + 2], 16) for offset in (0, 2, 4))


def foreground_bbox(image: Image.Image, background: tuple[int, int, int]) -> tuple[int, int, int, int]:
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    bg = np.asarray(background, dtype=np.int16)
    difference = np.max(np.abs(pixels - bg), axis=2)
    ys, xs = np.nonzero(difference > 18)
    if len(xs) == 0:
        raise ValueError("no foreground pixels detected")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    margin = max(4, round(max(x1 - x0, y1 - y0) * 0.025))
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(image.width, x1 + margin),
        min(image.height, y1 + margin),
    )


def make_avatar(source: Path, target: Path, background: tuple[int, int, int], size: int) -> dict[str, object]:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    bbox = foreground_bbox(image, background)
    content = image.crop(bbox)
    max_width = round(size * 0.80)
    max_height = round(size * 0.64)
    scale = min(max_width / content.width, max_height / content.height)
    resized_size = (
        max(1, round(content.width * scale)),
        max(1, round(content.height * scale)),
    )
    content = content.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), background)
    x = (size - content.width) // 2
    y = round(size * 0.22 + (max_height - content.height) / 2)
    canvas.paste(content, (x, y))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=True)
    return {
        "source_bbox": list(bbox),
        "placed_bbox": [x, y, x + content.width, y + content.height],
        "scale": scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create centered square avatar LoRA data from cleaned v1 images")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/training/sdxl_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/training/sdxl_avatar_v2")
    parser.add_argument("--captions", type=Path, default=ROOT / "configs/training/captions_en.json")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    content_captions = json.loads(args.captions.read_text(encoding="utf-8"))
    if manifest.get("version") != "sdxl_v1":
        raise SystemExit("input dataset is not the expected sdxl_v1 dataset")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        old_manifest = output_dir / "manifest.json"
        if not args.replace:
            raise SystemExit(f"output already exists; pass --replace: {output_dir}")
        if not old_manifest.is_file() or json.loads(old_manifest.read_text()).get("version") != "sdxl_avatar_v2":
            raise SystemExit(f"refusing to replace unknown directory: {output_dir}")

    staging = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    staging.mkdir(parents=True)
    converted: list[dict[str, object]] = []
    try:
        for subset in ("train", "validation"):
            (staging / subset).mkdir()
        for record in manifest["records"]:
            source = input_dir / record["image_path"]
            target = staging / record["image_path"]
            background = parse_color(record["background_color"])
            placement = make_avatar(source, target, background, args.size)
            caption = f"{STYLE_TOKEN}, {content_captions[record['id']]}, {STYLE_SUFFIX}"
            caption_path = target.with_suffix(".txt")
            caption_path.write_text(caption + "\n", encoding="utf-8")
            converted.append({
                "id": record["id"],
                "subset": record["subset"],
                "source_path": str(source.relative_to(ROOT)),
                "image_path": str(target.relative_to(staging)),
                "caption_path": str(caption_path.relative_to(staging)),
                "background_color": record["background_color"],
                "width": args.size,
                "height": args.size,
                "training_sha256": sha256(target),
                "caption": caption,
                **placement,
            })
        output_manifest = {
            "version": "sdxl_avatar_v2",
            "source_version": manifest["version"],
            "seed": manifest["seed"],
            "style_token": STYLE_TOKEN,
            "record_count": len(converted),
            "train_count": sum(r["subset"] == "train" for r in converted),
            "validation_count": sum(r["subset"] == "validation" for r in converted),
            "records": converted,
        }
        (staging / "manifest.json").write_text(
            json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            backup = output_dir.parent / f".{output_dir.name}.backup-{os.getpid()}"
            output_dir.rename(backup)
            try:
                staging.rename(output_dir)
            except BaseException:
                backup.rename(output_dir)
                raise
            shutil.rmtree(backup)
        else:
            staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Prepared square avatar dataset: {output_dir}")
    print(f"train={output_manifest['train_count']}, validation={output_manifest['validation_count']}")


if __name__ == "__main__":
    main()
