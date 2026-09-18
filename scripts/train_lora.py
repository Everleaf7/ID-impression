#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validated wrapper for the pinned SDXL LoRA trainer")
    parser.add_argument("--model", type=Path, required=True, help="Local SDXL Base 1.0 directory or checkpoint")
    parser.add_argument("--sd-scripts-dir", type=Path, default=ROOT / ".tools/sd-scripts")
    parser.add_argument("--gpu", default="0", help="Single CUDA device index exposed to the trainer")
    parser.add_argument("--execute", action="store_true", help="Run training; otherwise print a dry-run command")
    args = parser.parse_args()

    backend = args.sd_scripts_dir.resolve()
    trainer = backend / "sdxl_train_network.py"
    model = args.model.resolve()
    dataset = ROOT / "data/training/sdxl_v1"
    dataset_config = ROOT / "configs/training/sdxl_lora_dataset.toml"
    lock = json.loads((ROOT / "configs/training/backend.lock.json").read_text(encoding="utf-8"))
    if not trainer.is_file():
        raise SystemExit(f"pinned trainer not found: {trainer}")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=backend, check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != lock["trainer"]["commit"]:
        raise SystemExit(f"trainer revision {revision} does not match lock {lock['trainer']['commit']}")
    if args.execute and not model.exists():
        raise SystemExit(f"local base model not found: {model}")
    if not (dataset / "manifest.json").is_file():
        raise SystemExit("prepared dataset missing; run scripts/prepare_training_data.py first")

    accelerate = Path(sys.executable).with_name("accelerate")
    if args.execute and not accelerate.is_file():
        raise SystemExit(
            f"accelerate not found next to the active Python: {accelerate}; "
            "run this wrapper with .venv-training/bin/python"
        )

    command = [
        str(accelerate), "launch", "--num_cpu_threads_per_process", "2", str(trainer),
        "--pretrained_model_name_or_path", str(model),
        "--tokenizer_cache_dir", str(ROOT / ".tools/tokenizers"),
        "--dataset_config", str(dataset_config),
        "--output_dir", str(ROOT / "outputs/training/sdxl_lora_v1"),
        "--logging_dir", str(ROOT / "outputs/training/sdxl_lora_v1/logs"),
        "--output_name", "id_impression_doodle_sdxl_v1",
        "--save_model_as", "safetensors",
        "--network_module", "networks.lora",
        "--network_dim", "16", "--network_alpha", "8",
        "--learning_rate", "1e-4", "--unet_lr", "1e-4",
        "--network_train_unet_only",
        "--optimizer_type", "AdamW", "--lr_scheduler", "constant",
        "--max_train_steps", "450", "--save_every_n_steps", "75",
        "--mixed_precision", "bf16", "--save_precision", "bf16",
        "--gradient_checkpointing", "--cache_latents", "--cache_latents_to_disk",
        "--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk",
        "--max_data_loader_n_workers", "4", "--persistent_data_loader_workers",
        "--seed", "42", "--log_with", "tensorboard",
        "--sample_prompts", str(ROOT / "configs/training/sample_prompts.txt"),
        "--sample_every_n_steps", "75", "--sample_sampler", "euler_a",
    ]
    print("Pinned trainer:", revision)
    if not model.exists():
        print("Dry-run model path does not exist yet:", model)
    print("Command:", f"CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)}", shlex.join(command))
    if not args.execute:
        print("Dry run only. Add --execute to start GPU training.")
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    subprocess.run(command, cwd=backend, env=env, check=True)


if __name__ == "__main__":
    main()
