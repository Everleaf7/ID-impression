from __future__ import annotations

from src.semantic.schemas import Concept


STYLE_PROMPT = (
    "rough digital doodle, adult amateur drawing made quickly in a basic paint app, "
    "simple black outlines, slightly shaky uneven lines, occasional open contours, "
    "white canvas, large empty space, minimal flat accent colors, simple shapes, "
    "imperfect anatomy and perspective, mouse-drawn feeling, unpolished, low detail, "
    "playful visual metaphor, sparse composition"
)
NEGATIVE_PROMPT = (
    "photorealistic, realistic, 3d render, cinematic lighting, highly detailed, "
    "professional illustration, perfect anatomy, complex background, highly polished, "
    "digital painting, smooth shading, realistic shadows, perfect symmetry, "
    "close-up portrait, intricate texture, readable text, watermark"
)


def build_prompt(concept: Concept) -> tuple[str, str]:
    concept.validate()
    subjects = ", ".join(concept.main_subjects)
    secondary = ", ".join(concept.secondary_elements) or "none"
    colors = ", ".join(concept.accent_colors) or "one or two saturated colors"
    mood = ", ".join(concept.mood) or "playful"
    positive = (
        f"Visual idea: {concept.main_concept}. "
        f"Main subjects: {subjects}. Secondary elements: {secondary}. "
        f"Composition: {concept.composition}. Mood: {mood}. "
        f"Accent colors: {colors}. Expression type: {concept.expression_type}. "
        f"{STYLE_PROMPT}. No letters or words in the generated artwork."
    )
    return positive, NEGATIVE_PROMPT
