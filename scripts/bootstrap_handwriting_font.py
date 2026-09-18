#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    lock = json.loads((ROOT / "configs/training/backend.lock.json").read_text(encoding="utf-8"))
    spec = lock["handwriting_font"]
    target = ROOT / ".tools/fonts/MaokenYingBiKaiShuJ"
    font = target / spec["file"]
    if target.exists():
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=target, check=True, capture_output=True, text=True
        ).stdout.strip()
        if revision != spec["commit"] or not font.is_file():
            raise SystemExit(f"existing font checkout does not match lock: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", spec["repository"], str(target)], check=True)
        subprocess.run(["git", "checkout", spec["commit"]], cwd=target, check=True)
    print(font)


if __name__ == "__main__":
    main()
