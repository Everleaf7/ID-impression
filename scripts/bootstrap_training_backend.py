#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def git_output(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone the pinned sd-scripts backend")
    parser.add_argument("--target", type=Path, default=ROOT / ".tools/sd-scripts")
    args = parser.parse_args()
    lock = json.loads(
        (ROOT / "configs/training/backend.lock.json").read_text(encoding="utf-8")
    )["trainer"]
    target = args.target.resolve()
    if target.exists():
        if not (target / ".git").is_dir():
            raise SystemExit(f"target exists and is not a Git checkout: {target}")
        current = git_output("rev-parse", "HEAD", cwd=target)
        dirty = git_output("status", "--porcelain", cwd=target)
        if dirty:
            raise SystemExit(f"refusing to alter dirty checkout: {target}")
        if current != lock["commit"]:
            raise SystemExit(f"checkout is {current}, expected {lock['commit']}")
        print(f"Pinned backend already present: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", lock["repository"], str(target)],
        check=True,
    )
    subprocess.run(["git", "checkout", "--detach", lock["commit"]], cwd=target, check=True)
    print(f"Cloned {lock['name']} at {lock['commit']}: {target}")


if __name__ == "__main__":
    main()
