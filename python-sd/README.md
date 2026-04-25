# Python 方式使用 Stable Diffusion（Mac）

这个目录提供一个脚本版最小方案：不用 UI，也能直接通过 Python 传入 prompt 生成图片。

## 1) 激活环境

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /Users/tfwang/LLM/sd/sd-learn
```

## 2) 安装依赖（已装可跳过）

```bash
python -m pip install -U diffusers transformers accelerate safetensors
```

## 3) 运行示例

```bash
cd /Users/tfwang/LLM/sd/python-sd
python generate_image.py \
  --model runwayml/stable-diffusion-v1-5 \
  --prompt "a cinematic portrait of a young engineer, soft lighting, high detail" \
  --negative-prompt "lowres, blurry, bad anatomy, extra fingers, watermark, text, logo" \
  --width 512 \
  --height 512 \
  --steps 20 \
  --cfg 7 \
  --seed 42
```

输出会保存在：
- `./outputs/`

## 4) 常见可调参数

- `--steps`：步数，越高通常越清晰但更慢
- `--cfg`：提示词约束强度，常用 6~8
- `--seed`：固定随机种子，便于复现
- `--width --height`：分辨率，Mac 16GB 先从 512 起步

## 5) 注意事项

- 首次运行会下载模型文件，时间较长
- 默认优先使用 `mps`（Apple Silicon）
- 如果你已经有本地模型目录，也可将 `--model` 指向本地路径
