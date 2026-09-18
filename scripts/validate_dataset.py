#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.semantic.schemas import validate_dataset_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset JSONL records")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/dataset.example.jsonl")
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    count = 0
    for line_number, line in enumerate(
        args.dataset.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        count += 1
        try:
            raw = json.loads(line)
            validate_dataset_record(raw)
            image_path = raw.get("image_path")
            if args.require_images:
                if not image_path:
                    raise ValueError("image_path is required before training")
                if not (ROOT / image_path).resolve().is_file():
                    raise ValueError(f"image_path does not exist: {image_path}")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"line {line_number}: {exc}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {count} record(s): {args.dataset}")


if __name__ == "__main__":
    main()
