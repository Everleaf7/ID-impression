from __future__ import annotations

import random
import subprocess
import tempfile
from pathlib import Path


def add_to_text(image_path: Path, output_path: Path, user_id: str,
                font_path: str, font_size: int, seed: int) -> None:
    if not user_id.strip():
        raise ValueError("ID / 昵称不能为空")
    font = _find_font(font_path)
    rng = random.Random(seed + 101)
    x, y = 42 + rng.randint(-3, 3), 38 + rng.randint(-3, 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt") as handle:
        handle.write(f"To: {user_id}")
        handle.flush()
        filter_value = (
            f"drawtext=fontfile={font}:textfile={handle.name}:"
            f"fontcolor=0x171717:fontsize={font_size}:x={x}:y={y}"
        )
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(image_path), "-vf", filter_value, "-frames:v", "1",
             "-update", "1", str(output_path)],
            check=True,
        )


def _find_font(preferred: str) -> str:
    for path in (
        preferred,
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return path
    raise FileNotFoundError("No font found for To: ID postprocessing")
