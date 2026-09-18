# ID Impression Avatar

输入一个 ID / 昵称，先生成可审计的主观联想，再生成简洁涂鸦头像，最后以确定性后处理添加准确的手写 ID。

```text
ID -> Concept JSON -> text-free doodle -> exact handwritten ID
```

文字不交给扩散模型生成，因此中文、数字和符号不会变成伪文字。

## 功能

- 规则 baseline：将任意 ID 转换为结构化 Concept JSON
- SDXL LoRA 数据准备、训练和推理入口
- 固定 seed、prompt、negative prompt 和生成元数据
- 自动预留顶部空间并添加 `To: ID`
- AVIF 到 PNG 的无损解码流程
- 数据校验和单元测试

## Demo

下面是使用虚构 ID `秋日花信` 生成的真实 LoRA 输出。扩散模型先生成无文字画面，随后由确定性后处理写入准确 ID。

![ID 印象头像生成示例](docs/demo/id-impression-example.png)

仓库不包含训练数据、基础模型或 LoRA 权重。规则占位版本可用于本地检查语义与文件流，但不代表训练模型的最终画质。

## 隐私约定

公开仓库只包含虚构示例。真实昵称、标注、图片、训练副本、模型权重、日志和生成结果均由 `.gitignore` 排除。请勿把真实数据复制到示例文件中。

需要本地训练时，先从示例创建私有配置：

```bash
cp data/dataset.example.jsonl data/dataset.jsonl
cp configs/training/captions_en.example.json configs/training/captions_en.json
cp configs/training/header_masks.example.json configs/training/header_masks.json
cp configs/training/split_v1.example.json configs/training/split_v1.json
```

随后替换为本地数据；这些目标文件不会被 Git 跟踪。

## 快速 baseline

规则占位版本用于快速检查语义和文件流：

```bash
python3 scripts/generate.py --id "示例昵称" --seed 42
```

输出 Concept JSON、prompt、占位涂鸦、带手写 ID 的 PNG 和元数据。

## 数据准备

AVIF 源文件可转换为 PNG：

```bash
python3 scripts/convert_avif.py \
  --input-dir data/raw/batch_local \
  --output-dir data/images
```

校验私有数据并生成训练副本：

```bash
python3 scripts/validate_dataset.py --dataset data/dataset.jsonl --require-images
python3 scripts/prepare_training_data.py --check-only
python3 scripts/prepare_training_data.py
python3 scripts/prepare_avatar_training_data.py
```

原图不会被原地修改。去标题训练副本、方形头像副本和 manifest 均写入被忽略的 `data/training/`。

## 训练

训练后端和基础模型 revision 固定在 `configs/training/backend.lock.json`：

```bash
python3 scripts/bootstrap_training_backend.py
python3 scripts/bootstrap_handwriting_font.py
bash scripts/setup_training_env.sh
```

训练包装器默认只打印命令。确认数据、模型和目标设备后，再显式添加 `--execute`：

```bash
.venv-training/bin/python scripts/train_avatar_lora.py \
  --model /path/to/sdxl-base.safetensors \
  --gpu 0
```

## 正式生成

准备本地基础模型、LoRA 和字体后运行：

```bash
.venv-training/bin/python scripts/generate_avatar.py \
  --id "你的ID" \
  --model /path/to/sdxl-base.safetensors \
  --lora /path/to/avatar-lora.safetensors \
  --gpu 0 \
  --output-dir outputs/generated/example
```

主要产物为 `avatar_with_id.png`；同目录还会保存无文字图、Concept JSON、prompt 和经过脱敏的生成元数据。

## 验证

```bash
python3 scripts/validate_dataset.py --dataset data/dataset.example.jsonl
python3 -m unittest discover -s tests -v
```

## 当前边界

规则联想层只提供最小可运行 baseline；高质量的主观联想仍取决于经授权的数据和人工评审。增加训练步数不能替代数据多样性，也不保证消除机器、界面等主体中的伪文字。

## 许可证

代码以 [MIT License](LICENSE) 发布。`web/assets/` 中的角色立绘已获权利持有人授权在本项目中公开再分发，但不属于 MIT License 的授权范围；未经权利持有人另行许可，不得独立提取、修改或再利用这些立绘。
