#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

CRITERIA = (
    "id_relevance", "creative_association", "style_similarity", "simplicity",
    "imperfection", "composition_sparsity", "humor_surprise",
    "over_generation_penalty",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual evaluation for one output")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"image not found: {args.input}")

    scores: dict[str, int] = {}
    print("每项输入 1～5 分；over_generation_penalty 输入 0～5。")
    for criterion in CRITERIA:
        lower = 0 if criterion == "over_generation_penalty" else 1
        while True:
            try:
                score = int(input(f"{criterion} [{lower}-5]: "))
                if lower <= score <= 5:
                    scores[criterion] = score
                    break
            except ValueError:
                pass
            print("输入无效，请重试。")
    payload = {"id": args.id, "image": str(args.input), "scores": scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved evaluation: {args.output}")


if __name__ == "__main__":
    main()
