#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, SpaceHardware


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy/huggingface"
DEFAULT_REPO_ID = "Everleaf7/ID-impression-avatar"


def space_files() -> dict[str, Path]:
    files = {
        "README.md": DEPLOY / "README.md",
        "app.py": DEPLOY / "app.py",
        "requirements.txt": DEPLOY / "requirements.txt",
        "THIRD_PARTY_NOTICES.md": DEPLOY / "THIRD_PARTY_NOTICES.md",
        "LICENSE": ROOT / "LICENSE",
        "web/assets/autumn-outfit-front.png": ROOT / "web/assets/autumn-outfit-front.png",
        "web/assets/autumn-outfit-pose.png": ROOT / "web/assets/autumn-outfit-pose.png",
        "weights/id_impression_avatar_sdxl_v2.safetensors": (
            ROOT / "outputs/training/sdxl_avatar_v2/id_impression_avatar_sdxl_v2.safetensors"
        ),
        "fonts/MaokenYingBiKaiShuJ_0.09.ttf": (
            ROOT / ".tools/fonts/MaokenYingBiKaiShuJ/MaokenYingBiKaiShuJ_0.09.ttf"
        ),
        "fonts/OFL.txt": ROOT / ".tools/fonts/MaokenYingBiKaiShuJ/OFL.txt",
    }
    for relative in (
        "src/postprocess/add_handwritten_id.py",
        "src/semantic/concept_generator.py",
        "src/semantic/id_parser.py",
        "src/semantic/schemas.py",
        "src/web_demo/__init__.py",
        "src/web_demo/app.py",
    ):
        files[relative] = ROOT / relative
    return files


def validate_files(files: dict[str, Path]) -> None:
    missing = [f"{target} <- {source}" for target, source in files.items() if not source.is_file()]
    if missing:
        raise SystemExit("Space publish inputs are missing:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the Gradio app to a Hugging Face ZeroGPU Space")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true", help="create a private Space instead of a public one")
    parser.add_argument("--dry-run", action="store_true", help="validate and list files without contacting the Hub")
    args = parser.parse_args()

    files = space_files()
    validate_files(files)
    print(f"Space: {args.repo_id}")
    print(f"Visibility: {'private' if args.private else 'public'}")
    for target, source in sorted(files.items()):
        print(f"  {target} <- {source.relative_to(ROOT) if source.is_relative_to(ROOT) else source.name}")
    if args.dry_run:
        return

    api = HfApi()
    account = api.whoami()
    print(f"Authenticated as: {account['name']}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware=SpaceHardware.ZERO_A10G,
        private=args.private,
        exist_ok=True,
    )
    operations = [
        CommitOperationAdd(path_in_repo=target, path_or_fileobj=source)
        for target, source in files.items()
    ]
    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="space",
        operations=operations,
        commit_message="Publish ID impression avatar ZeroGPU demo",
    )
    runtime = api.request_space_hardware(args.repo_id, SpaceHardware.ZERO_A10G)
    print(f"Commit: {commit.commit_url}")
    print(f"Runtime: {runtime.stage}; requested hardware: {runtime.requested_hardware}")
    print(f"Space: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
