from __future__ import annotations

import os
import secrets
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "GRADIO_TEMP_DIR",
    str(Path(tempfile.gettempdir()) / f"idavatar-gradio-{os.getuid()}"),
)

import spaces
import gradio as gr
import torch
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline
from PIL import Image

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT
if not (PROJECT_ROOT / "src").is_dir() and (APP_ROOT.parents[1] / "src").is_dir():
    PROJECT_ROOT = APP_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.postprocess.add_handwritten_id import add_handwritten_id
from src.semantic.concept_generator import generate_concept
from src.web_demo.app import validate_id, validate_seed


ASSET_DIR = PROJECT_ROOT / "web/assets"
LORA_PATH = (
    APP_ROOT / "weights/id_impression_avatar_sdxl_v2.safetensors"
    if (APP_ROOT / "weights").is_dir()
    else PROJECT_ROOT / "outputs/training/sdxl_avatar_v2/id_impression_avatar_sdxl_v2.safetensors"
)
FONT_PATH = (
    APP_ROOT / "fonts/MaokenYingBiKaiShuJ_0.09.ttf"
    if (APP_ROOT / "fonts").is_dir()
    else PROJECT_ROOT / ".tools/fonts/MaokenYingBiKaiShuJ/MaokenYingBiKaiShuJ_0.09.ttf"
)
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
MAX_SEED = 2**32 - 1

NEGATIVE_PROMPT = (
    "text, letters, words, handwriting, signature, watermark, logo, label, typography, "
    "photorealistic, realistic, 3d render, cinematic lighting, professional polished illustration, "
    "complex detailed background, perfect anatomy, glossy shading"
)


def build_prompt(user_id: str) -> tuple[str, str]:
    concept = generate_concept(user_id)
    prompt = (
        f"idavatar doodle style, {concept.image_prompt}, centered in the lower two thirds, "
        "wide blank space across the top, plain white background, rough amateur mouse doodle, "
        "shaky uneven black lines, simple flat shapes, sparse, unpolished"
    )
    return prompt, concept.main_concept


def normalize_inputs(user_id: str, seed: int | float, randomize: bool) -> tuple[str, int]:
    clean_id = validate_id(user_id)
    clean_seed = secrets.randbelow(MAX_SEED + 1) if randomize else validate_seed(seed)
    return clean_id, clean_seed


def load_pipeline() -> StableDiffusionXLPipeline | None:
    if os.environ.get("ID_AVATAR_TEST_MODE") == "1":
        return None
    if not LORA_PATH.is_file():
        raise FileNotFoundError(f"LoRA not found: {LORA_PATH}")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_lora_weights(
        LORA_PATH.parent,
        weight_name=LORA_PATH.name,
        adapter_name="idavatar",
    )
    pipeline.set_adapters("idavatar", adapter_weights=1.0)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to("cuda")
    return pipeline


pipe = load_pipeline()


@spaces.GPU(duration=50)
def generate_avatar(
    user_id: str,
    seed: int | float,
    randomize: bool,
) -> tuple[Image.Image, str, int]:
    clean_id, clean_seed = normalize_inputs(user_id, seed, randomize)
    prompt, concept = build_prompt(clean_id)
    if pipe is None:
        raise gr.Error("推理模型未加载")

    generator = torch.Generator(device="cuda").manual_seed(clean_seed)
    with torch.inference_mode():
        clean_image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=1024,
            height=1024,
            num_inference_steps=30,
            guidance_scale=7.0,
            generator=generator,
        ).images[0]

    with tempfile.TemporaryDirectory(prefix="idavatar-") as temp_name:
        temp = Path(temp_name)
        clean_path = temp / "avatar_clean.png"
        final_path = temp / "avatar_with_id.png"
        clean_image.save(clean_path, format="PNG")
        add_handwritten_id(clean_path, final_path, clean_id, FONT_PATH, clean_seed)
        with Image.open(final_path) as opened:
            final_image = opened.copy()
    return final_image, concept, clean_seed


CSS = """
:root {
  --night: #111611; --forest: #171e17; --cream: #eee9dc;
  --paper: #f5f0e4; --muted: #aaa798; --moss: #c5d35e;
}
.gradio-container {
  max-width: none !important; min-height: 100vh !important; color: var(--cream) !important;
  background: radial-gradient(circle at 50% 34%, rgba(197,211,94,.10), transparent 31rem),
              linear-gradient(145deg, var(--night), var(--forest) 52%, #0d120e) !important;
}
.main-wrap { position: relative; z-index: 2; max-width: 980px; margin: 0 auto; padding: 5vh 24px 8vh; }
.character-stage { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
.character-stage img { position: absolute; bottom: -3vh; height: min(94vh, 1020px); opacity: .48;
  filter: saturate(.8) brightness(.72); object-fit: contain; }
.character-stage .front { left: max(-10vw, -170px); }
.character-stage .pose { right: max(-12vw, -210px); }
.character-stage::after { content: ''; position: absolute; inset: 0;
  background: linear-gradient(90deg, transparent 0 17%, rgba(17,22,17,.72) 34% 66%, transparent 83%); }
.brand { color: var(--moss); font: 700 11px ui-monospace, monospace; letter-spacing: .17em; }
.hero h1 { color: var(--cream); font-family: Georgia, 'Noto Serif SC', serif; font-size: clamp(46px, 7vw, 76px);
  font-weight: 400; line-height: 1.08; letter-spacing: -.06em; margin: 16px 0; }
.hero p { max-width: 680px; color: #c6c2b6; line-height: 1.8; }
.panel { border: 1px solid rgba(238,233,220,.28) !important; background: rgba(23,30,23,.92) !important;
  box-shadow: 0 28px 90px rgba(0,0,0,.38); padding: 24px !important; }
.panel label, .panel span, .panel p { color: var(--cream) !important; }
.panel input, .panel textarea { color: var(--cream) !important; background: rgba(0,0,0,.15) !important; }
.primary { background: var(--moss) !important; color: #151a12 !important; border-color: var(--moss) !important; }
.result-image { background: var(--paper) !important; }
.footer-note { color: var(--muted); font: 10px ui-monospace, monospace; letter-spacing: .1em; margin-top: 18px; }
@media (max-width: 900px) {
  .character-stage img { opacity: .22; }
  .character-stage .front { left: -40vw; } .character-stage .pose { right: -45vw; }
}
"""


gr.set_static_paths(paths=[ASSET_DIR])

with gr.Blocks(title="ID 印象头像", css=CSS, fill_width=True) as demo:
    gr.HTML(
        """
        <div class="character-stage" aria-hidden="true">
          <img class="front" src="/gradio_api/file=web/assets/autumn-outfit-front.png" alt="">
          <img class="pose" src="/gradio_api/file=web/assets/autumn-outfit-pose.png" alt="">
        </div>
        """
    )
    with gr.Column(elem_classes="main-wrap"):
        gr.HTML(
            """
            <section class="hero">
              <div class="brand">✦ EVERLEAF · ID IMPRESSION</div>
              <h1>一个名字，<br>一张随手画。</h1>
              <p>输入你的 ID，让模型画出它带来的第一印象。画面保持简洁的鼠绘感，ID 会在最后准确写入。</p>
            </section>
            """
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_classes="panel"):
                user_id = gr.Textbox(
                    label="ID / 昵称",
                    placeholder="在这里写下你的名字",
                    max_lines=1,
                    max_length=40,
                )
                seed = gr.Number(label="Seed", value=42, minimum=0, maximum=MAX_SEED, precision=0)
                randomize = gr.Checkbox(label="每次随机 Seed", value=True)
                submit = gr.Button("开始联想 ↗", variant="primary", elem_classes="primary")
                concept = gr.Textbox(label="这次的联想", interactive=False, lines=2)
                used_seed = gr.Number(label="实际 Seed", interactive=False, precision=0)
            with gr.Column(scale=6, elem_classes="panel"):
                output = gr.Image(
                    label="生成结果",
                    type="pil",
                    format="png",
                    height=560,
                    elem_classes="result-image",
                )
        gr.HTML('<div class="footer-note">1024 × 1024 · SDXL + LoRA · ZERO GPU</div>')

    submit.click(
        fn=generate_avatar,
        inputs=[user_id, seed, randomize],
        outputs=[output, concept, used_seed],
        concurrency_limit=1,
        api_name="generate",
    )
    user_id.submit(
        fn=generate_avatar,
        inputs=[user_id, seed, randomize],
        outputs=[output, concept, used_seed],
        concurrency_limit=1,
        api_name=False,
    )

demo.queue(max_size=24, default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch()
