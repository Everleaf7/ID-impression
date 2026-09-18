#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.web_demo.app import (
    AvatarGenerator,
    GeneratorConfig,
    PublicError,
    RateLimiter,
    ServerContext,
    serve,
    validate_exposure,
    validate_proxy_mode,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ID impression avatar web service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument(
        "--public-behind-cloudflare",
        action="store_true",
        help="trust Cloudflare client IP headers while remaining bound to loopback",
    )
    parser.add_argument("--backend", choices=("sdxl", "placeholder"), default="sdxl")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/web_demo")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retention-hours", type=int, default=24)
    args = parser.parse_args()

    token = os.environ.get("ID_AVATAR_DEMO_TOKEN", "")
    try:
        validate_exposure(args.host, token, args.allow_remote)
        validate_proxy_mode(args.host, args.public_behind_cloudflare)
    except PublicError as exc:
        raise SystemExit(str(exc)) from exc
    if not 1 <= args.port <= 65535:
        raise SystemExit("port must be in [1, 65535]")
    if args.timeout < 30:
        raise SystemExit("timeout must be at least 30 seconds")
    if args.retention_hours < 1:
        raise SystemExit("retention-hours must be at least 1")

    output_root = args.output_root.resolve()
    allowed_output_root = (ROOT / "outputs").resolve()
    if output_root != allowed_output_root and allowed_output_root not in output_root.parents:
        raise SystemExit("output-root must be inside the project outputs directory")

    model = args.model or ROOT / ".tools/models/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"
    lora = args.lora or ROOT / "outputs/training/sdxl_avatar_v2/id_impression_avatar_sdxl_v2.safetensors"
    font = args.font or ROOT / ".tools/fonts/MaokenYingBiKaiShuJ/MaokenYingBiKaiShuJ_0.09.ttf"
    if args.backend == "sdxl":
        for path, label in ((model, "base model"), (lora, "LoRA"), (font, "font")):
            if not path.resolve().is_file():
                raise SystemExit(f"{label} not found; pass its path explicitly")

    config = GeneratorConfig(
        backend=args.backend,
        gpu=args.gpu,
        output_root=output_root,
        model=model.resolve() if args.backend == "sdxl" else None,
        lora=lora.resolve() if args.backend == "sdxl" else None,
        font=font.resolve() if args.backend == "sdxl" else None,
        timeout_seconds=args.timeout,
        retention_hours=args.retention_hours,
    )
    config.output_root.mkdir(parents=True, exist_ok=True)
    context = ServerContext(
        generator=AvatarGenerator(config),
        token=token,
        rate_limiter=RateLimiter(limit=3, window_seconds=900),
        global_rate_limiter=RateLimiter(limit=20, window_seconds=3600),
        trust_cloudflare=args.public_behind_cloudflare,
    )
    serve(args.host, args.port, context)


if __name__ == "__main__":
    main()
