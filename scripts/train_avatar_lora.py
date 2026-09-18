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
    parser = argparse.ArgumentParser(description="Train the square-avatar SDXL LoRA v2")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpu", default="0", help="Single CUDA device index")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    backend = ROOT / ".tools/sd-scripts"
    trainer = backend / "sdxl_train_network.py"
    model = args.model.resolve()
    dataset = ROOT / "data/training/sdxl_avatar_v2/manifest.json"
    lock = json.loads((ROOT / "configs/training/backend.lock.json").read_text(encoding="utf-8"))
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=backend, check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != lock["trainer"]["commit"]:
        raise SystemExit(f"trainer revision mismatch: {revision}")
    if not trainer.is_file() or not dataset.is_file():
        raise SystemExit("trainer or square avatar dataset is missing")
    if args.execute and not model.is_file():
        raise SystemExit(f"base checkpoint not found: {model}")
    accelerate = Path(sys.executable).with_name("accelerate")
    output = ROOT / "outputs/training/sdxl_avatar_v2"
    if args.execute and output.exists() and any(output.glob("*.safetensors")):
        raise SystemExit(f"refusing to overwrite existing checkpoints: {output}")

    command = [
        str(accelerate), "launch", "--num_processes", "1", "--num_machines", "1",
        "--mixed_precision", "bf16", "--dynamo_backend", "no",
        "--num_cpu_threads_per_process", "2", str(trainer),
        "--pretrained_model_name_or_path", str(model),
        "--tokenizer_cache_dir", str(ROOT / ".tools/tokenizers"),
        "--dataset_config", str(ROOT / "configs/training/sdxl_avatar_v2_dataset.toml"),
        "--output_dir", str(output),
        "--logging_dir", str(output / "logs"),
        "--output_name", "id_impression_avatar_sdxl_v2",
        "--save_model_as", "safetensors",
        "--network_module", "networks.lora",
        "--network_dim", "16", "--network_alpha", "8",
        "--learning_rate", "7e-5", "--unet_lr", "7e-5",
        "--network_train_unet_only",
        "--optimizer_type", "AdamW", "--lr_scheduler", "constant",
        "--max_train_steps", "300", "--save_every_n_steps", "50",
        "--min_snr_gamma", "5",
        "--mixed_precision", "bf16", "--save_precision", "bf16",
        "--gradient_checkpointing", "--sdpa",
        "--cache_latents", "--cache_latents_to_disk",
        "--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk",
        "--max_data_loader_n_workers", "4", "--persistent_data_loader_workers",
        "--seed", "42", "--log_with", "tensorboard",
        "--sample_prompts", str(ROOT / "configs/training/avatar_v2_sample_prompts.txt"),
        "--sample_every_n_steps", "50", "--sample_sampler", "euler_a",
    ]
    print("Pinned trainer:", revision, flush=True)
    print("Command:", f"CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)}", shlex.join(command), flush=True)
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
