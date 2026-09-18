#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TOOL_ROOT = ROOT / ".tools/avif"
LOCAL_AVIFDEC = LOCAL_TOOL_ROOT / "usr/bin/avifdec"
LOCAL_LIBRARY_DIR = LOCAL_TOOL_ROOT / "usr/lib/x86_64-linux-gnu"


def find_decoder() -> tuple[str, dict[str, str]]:
    system_decoder = shutil.which("avifdec")
    if system_decoder:
        return system_decoder, os.environ.copy()
    if LOCAL_AVIFDEC.is_file():
        environment = os.environ.copy()
        current = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = (
            f"{LOCAL_LIBRARY_DIR}:{current}" if current else str(LOCAL_LIBRARY_DIR)
        )
        return str(LOCAL_AVIFDEC), environment
    raise SystemExit(
        "avifdec not found. Install libavif-bin or unpack it under .tools/avif."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Losslessly decode AVIF files to PNG")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/images")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/images")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    inputs = sorted(args.input_dir.glob("*.avif"))
    if not inputs:
        raise SystemExit(f"No AVIF files found in {args.input_dir}")
    decoder, environment = find_decoder()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for source in inputs:
        target = args.output_dir / f"{source.stem}.png"
        if target.exists() and not args.force:
            raise SystemExit(f"Target exists; use --force to replace it: {target}")
        subprocess.run(
            [decoder, str(source), str(target)],
            env=environment,
            check=True,
        )
        converted += 1
        print(f"{source.name} -> {target.name}")
    print(f"Converted {converted} AVIF file(s)")


if __name__ == "__main__":
    main()
