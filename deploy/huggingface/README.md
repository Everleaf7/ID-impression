---
title: ID Impression Avatar
emoji: 🎨
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.41.0
python_version: 3.10.13
app_file: app.py
pinned: false
license: other
models:
  - stabilityai/stable-diffusion-xl-base-1.0
tags:
  - text-to-image
  - lora
  - sdxl
  - zerogpu
---

# ID Impression Avatar

输入 ID 或昵称，生成一张带有准确手写 ID 的主观联想头像。

- 输出分辨率：1024 × 1024 PNG
- 推理：SDXL Base 1.0 + 项目 LoRA
- 页面角色立绘已获得权利方授权，可作为本项目的一部分公开展示与再分发

## License

Application code is available under the repository's MIT license. The LoRA weights are derived from SDXL Base 1.0 and remain subject to the CreativeML Open RAIL++-M license. The bundled handwriting font is licensed under SIL Open Font License 1.1. Character artwork is excluded from the MIT license and is redistributed only under the permission stated above.
