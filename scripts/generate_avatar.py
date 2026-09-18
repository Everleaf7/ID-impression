#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.postprocess.add_handwritten_id import add_handwritten_id
from src.semantic.concept_generator import generate_concept


NEGATIVE = (
    "text, letters, words, handwriting, signature, watermark, logo, label, typography, "
    "photorealistic, realistic, 3d render, cinematic lighting, professional polished illustration, "
    "complex detailed background, perfect anatomy, glossy shading"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a square ID-impression avatar and add exact handwritten ID")
    parser.add_argument("--id", required=True, help="ID / 昵称")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--model", type=Path, default=ROOT / ".tools/models/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors")
    parser.add_argument("--lora", type=Path, default=ROOT / "outputs/training/sdxl_avatar_v2/id_impression_avatar_sdxl_v2.safetensors")
    parser.add_argument("--font", type=Path, default=ROOT / ".tools/fonts/MaokenYingBiKaiShuJ/MaokenYingBiKaiShuJ_0.09.ttf")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/generated/latest")
    args = parser.parse_args()

    for path, label in ((args.model, "base model"), (args.lora, "LoRA"), (args.font, "font")):
        if not path.resolve().is_file():
            raise SystemExit(f"{label} not found: {path.resolve()}")
    concept = generate_concept(args.id)
    prompt = (
        f"idavatar doodle style, {concept.image_prompt}, centered in the lower two thirds, "
        "wide blank space across the top, plain white background, rough amateur mouse doodle, "
        "shaky uneven black lines, simple flat shapes, sparse, unpolished"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    python = ROOT / ".venv-training/bin/python"
    generator = ROOT / ".tools/sd-scripts/sdxl_gen_img.py"
    with tempfile.TemporaryDirectory(prefix="id-avatar-") as temp_name:
        temp = Path(temp_name)
        command = [
            str(python), str(generator),
            "--ckpt", str(args.model.resolve()),
            "--tokenizer_cache_dir", str(ROOT / ".tools/tokenizers"),
            "--network_module", "networks.lora",
            "--network_weights", str(args.lora.resolve()),
            "--network_mul", str(args.strength),
            "--prompt", f"{prompt} --n {NEGATIVE}",
            "--W", "1024", "--H", "1024", "--steps", "30",
            "--sampler", "euler_a", "--scale", "7.0", "--seed", str(args.seed),
            "--images_per_prompt", "1", "--batch_size", "1",
            "--outdir", str(temp), "--sequential_file_name", "--bf16", "--sdpa",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        subprocess.run(command, cwd=ROOT / ".tools/sd-scripts", env=env, check=True)
        images = sorted(temp.glob("*.png"))
        if len(images) != 1:
            raise RuntimeError(f"expected one generated image, found {len(images)}")
        clean_path = output_dir / "avatar_clean.png"
        shutil.copy2(images[0], clean_path)
    final_path = output_dir / "avatar_with_id.png"
    add_handwritten_id(clean_path, final_path, args.id, args.font.resolve(), args.seed)
    (output_dir / "concept.json").write_text(
        json.dumps(concept.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "prompt.txt").write_text(
        f"POSITIVE\n{prompt}\n\nNEGATIVE\n{NEGATIVE}\n", encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps({
            "id": args.id, "seed": args.seed, "device": "cuda",
            "base_model": args.model.name, "lora": args.lora.name,
            "lora_strength": args.strength, "font": args.font.name,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(final_path)


if __name__ == "__main__":
    main()
