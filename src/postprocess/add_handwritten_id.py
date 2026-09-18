from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat


def _corner_background(image: Image.Image) -> tuple[int, int, int]:
    patch = max(8, image.width // 32)
    boxes = (
        (0, 0, patch, patch),
        (image.width - patch, 0, image.width, patch),
        (0, image.height - patch, patch, image.height),
        (image.width - patch, image.height - patch, image.width, image.height),
    )
    means = [ImageStat.Stat(image.crop(box)).mean[:3] for box in boxes]
    return tuple(round(sum(channel) / len(means)) for channel in zip(*means))


def add_handwritten_id(
    image_path: Path,
    output_path: Path,
    user_id: str,
    font_path: Path,
    seed: int,
) -> None:
    value = user_id.strip()
    if not value:
        raise ValueError("ID / 昵称不能为空")
    if not font_path.is_file():
        raise FileNotFoundError(f"handwriting font not found: {font_path}")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    # Normalize every generated image into the same layout as the collected
    # references. This guarantees a calm header area even when diffusion puts
    # detail too close to the top edge.
    background = _corner_background(image)
    canvas = Image.new("RGB", image.size, background)
    content = image.copy()
    content.thumbnail((round(image.width * 0.90), round(image.height * 0.82)), Image.Resampling.LANCZOS)
    content_x = (image.width - content.width) // 2
    content_y = round(image.height * 0.17)
    canvas.paste(content, (content_x, content_y))
    image = canvas
    text = f"To: {value}"
    max_width = round(image.width * 0.88)
    font_size = max(30, round(image.width * 0.070))
    while font_size > 30:
        font = ImageFont.truetype(str(font_path), font_size)
        bbox = ImageDraw.Draw(image).textbbox((0, 0), text, font=font, stroke_width=1)
        if bbox[2] - bbox[0] <= max_width:
            break
        font_size -= 2
    rng = random.Random(seed + 101)
    x = round(image.width * 0.055) + rng.randint(-4, 4)
    y = round(image.height * 0.045) + rng.randint(-3, 3)
    draw = ImageDraw.Draw(image)
    draw.text(
        (x, y), text, font=font, fill=(22, 22, 22),
        stroke_width=1, stroke_fill=(22, 22, 22),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
