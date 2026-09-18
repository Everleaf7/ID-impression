#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STYLE_PREFIX = "idimpression doodle style"
STYLE_SUFFIX = (
    "rough digital doodle by an adult drawing quickly in a basic paint app, "
    "shaky black mouse-drawn outlines, sparse composition, large empty space, "
    "minimal flat accent colors, imperfect shapes, unpolished, no text"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("dataset is empty")
    return records


def validate_inputs(
    records: list[dict[str, Any]],
    captions: dict[str, str],
    masks: dict[str, list[list[int]]],
    validation_ids: list[str],
) -> None:
    ids = [record["id"] for record in records]
    filenames = [Path(record["image_path"]).name for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset IDs must be unique")
    if set(captions) != set(ids):
        missing = sorted(set(ids) - set(captions))
        extra = sorted(set(captions) - set(ids))
        raise ValueError(f"caption coverage mismatch; missing={missing}, extra={extra}")
    if set(masks) != set(filenames):
        missing = sorted(set(filenames) - set(masks))
        extra = sorted(set(masks) - set(filenames))
        raise ValueError(f"mask coverage mismatch; missing={missing}, extra={extra}")
    if len(validation_ids) != len(set(validation_ids)):
        raise ValueError("validation IDs must be unique")
    unknown = sorted(set(validation_ids) - set(ids))
    if unknown:
        raise ValueError(f"validation IDs not found in dataset: {unknown}")
    if not 0 < len(validation_ids) < len(ids):
        raise ValueError("validation split must leave at least one train record")
    for record in records:
        if record.get("review_status") != "annotated_v1":
            raise ValueError(f"record is not confirmed: {record['id']}")
        image = ROOT / record["image_path"]
        if not image.is_file():
            raise ValueError(f"image not found: {image}")
        if not captions[record["id"]].strip():
            raise ValueError(f"empty caption: {record['id']}")
        for rect in masks[image.name]:
            if len(rect) != 4 or any(not isinstance(value, int) for value in rect):
                raise ValueError(f"invalid mask rectangle for {image.name}: {rect}")
            if rect[0] < 0 or rect[1] < 0 or rect[2] <= 0 or rect[3] <= 0:
                raise ValueError(f"invalid mask rectangle for {image.name}: {rect}")


def probe_size(image: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(image),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def sample_background(image: Path, width: int, height: int) -> str:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(image),
            "-vf", f"crop=1:1:{width - 1}:{height - 1},format=rgb24",
            "-frames:v", "1", "-f", "rawvideo", "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    if len(result.stdout) != 3:
        raise RuntimeError(f"could not sample background from {image}")
    return "0x" + result.stdout.hex()


def clean_header(source: Path, target: Path, rects: list[list[int]]) -> tuple[int, int, str]:
    width, height = probe_size(source)
    for x, y, rect_width, rect_height in rects:
        if x + rect_width > width or y + rect_height > height:
            raise ValueError(f"mask rectangle exceeds image bounds: {source.name}")
    background = sample_background(source, width, height)
    filters = [
        f"drawbox=x={x}:y={y}:w={rect_width}:h={rect_height}:color={background}:t=fill"
        for x, y, rect_width, rect_height in rects
    ]
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(source),
            "-vf", ",".join(filters), "-map_metadata", "-1", str(target),
        ],
        check=True,
    )
    return width, height, background


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_caption(content: str) -> str:
    return f"{STYLE_PREFIX}, {content}, {STYLE_SUFFIX}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean SDXL LoRA image-caption pairs")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/dataset.jsonl")
    parser.add_argument("--captions", type=Path, default=ROOT / "configs/training/captions_en.json")
    parser.add_argument("--masks", type=Path, default=ROOT / "configs/training/header_masks.json")
    parser.add_argument("--split", type=Path, default=ROOT / "configs/training/split_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/training/sdxl_v1")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace an existing dataset previously generated by this script",
    )
    args = parser.parse_args()

    records = load_records(args.dataset)
    captions = load_json(args.captions)
    masks = load_json(args.masks)
    split = load_json(args.split)
    validation_ids = split["validation_ids"]
    validate_inputs(records, captions, masks, validation_ids)
    if args.check_only:
        print(
            f"Training inputs valid: records={len(records)}, "
            f"train={len(records) - len(validation_ids)}, validation={len(validation_ids)}"
        )
        return

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        manifest_path = output_dir / "manifest.json"
        if not args.replace:
            raise SystemExit(f"output already exists; pass --replace or choose another directory: {output_dir}")
        if not manifest_path.is_file():
            raise SystemExit(f"refusing to replace directory without generated manifest: {output_dir}")
        current_manifest = load_json(manifest_path)
        if current_manifest.get("version") != "sdxl_v1":
            raise SystemExit(f"refusing to replace unknown generated dataset: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    staging.mkdir()
    manifest_records: list[dict[str, Any]] = []
    try:
        for subset in ("train", "validation"):
            (staging / subset).mkdir()
        validation_set = set(validation_ids)
        for record in records:
            source = (ROOT / record["image_path"]).resolve()
            subset = "validation" if record["id"] in validation_set else "train"
            image_target = staging / subset / source.name
            width, height, background = clean_header(source, image_target, masks[source.name])
            caption = build_caption(captions[record["id"]])
            caption_target = image_target.with_suffix(".txt")
            caption_target.write_text(caption + "\n", encoding="utf-8")
            manifest_records.append(
                {
                    "id": record["id"],
                    "subset": subset,
                    "source_path": str(source.relative_to(ROOT)),
                    "image_path": str(image_target.relative_to(staging)),
                    "caption_path": str(caption_target.relative_to(staging)),
                    "source_sha256": sha256(source),
                    "training_sha256": sha256(image_target),
                    "width": width,
                    "height": height,
                    "header_masks": masks[source.name],
                    "background_color": background,
                    "caption": caption,
                }
            )
        manifest = {
            "version": "sdxl_v1",
            "seed": split["seed"],
            "style_token": STYLE_PREFIX,
            "record_count": len(records),
            "train_count": len(records) - len(validation_ids),
            "validation_count": len(validation_ids),
            "records": manifest_records,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
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
    print(f"Prepared SDXL LoRA dataset: {output_dir}")
    print(f"train={len(records) - len(validation_ids)}, validation={len(validation_ids)}")


if __name__ == "__main__":
    main()
