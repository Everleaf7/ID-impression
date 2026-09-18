#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.pipeline import render_placeholder
from src.postprocess.add_to_text import add_to_text
from src.prompting.prompt_builder import build_prompt
from src.semantic.concept_generator import generate_concept


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an ID impression baseline")
    parser.add_argument("--id", required=True, help="ID / 昵称")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = config["seed"] if args.seed is None else args.seed
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    concept = generate_concept(args.id)
    positive, negative = build_prompt(concept)
    concept.image_prompt = positive

    (output_dir / "concept.json").write_text(
        json.dumps(concept.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "prompt.txt").write_text(
        f"POSITIVE\n{positive}\n\nNEGATIVE\n{negative}\n", encoding="utf-8"
    )
    canvas = config["canvas"]
    image_path = output_dir / "image.png"
    render_placeholder(
        concept, image_path, seed,
        width=canvas["width"], height=canvas["height"],
    )
    post = config["postprocess"]
    add_to_text(
        image_path, output_dir / "final_with_to.png", args.id,
        post["font_path"], post["font_size"], seed,
    )
    metadata = {
        "seed": seed,
        "backend": config["generation"]["backend"],
        "model_version": config["generation"]["model_version"],
        "config": str(args.config.resolve()),
        "style_reference": None,
        "note": "Prototype renderer only; no diffusion model or training was used.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated baseline outputs in {output_dir}")


if __name__ == "__main__":
    main()
