#!/usr/bin/env python3
"""
第一课：最小文生图（txt2img）
"""

import time
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

CONFIG = {
    "model": "runwayml/stable-diffusion-v1-5",
    # 人像：全身、偏时尚/妩媚气质（非露骨描写）；仍受 prompt_max_chars / tokenizer 截断约束。
    "prompt": "年轻亚洲女性，性感礼服全身照，影楼柔光妩媚",
    "negative_prompt": "身体变形，糊脸，畸形手指，低质",
    "width": 384,
    "height": 512,
    "steps": 24,
    "cfg": 6.5,
    "seed": 42,
    "outdir": "./outputs/lesson01",
    "prompt_max_chars": 40,
    "negative_prompt_max_chars": 24,
    "max_tokens": 77,
}


def align_to_multiple_of_8(x: int) -> int:
    return max(64, (x // 8) * 8)


def clamp_text(text: str, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    print(f"[WARN] {label} too long, truncating to {max_chars} chars.")
    return text[:max_chars]


def clamp_text_by_tokens(tokenizer, text: str, max_tokens: int, label: str) -> str:
    full_tokens = tokenizer.tokenize(text)
    if len(full_tokens) <= max_tokens:
        return text
    enc = tokenizer(text, truncation=True, max_length=max_tokens)
    trimmed = tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
    print(f"[WARN] {label} token length {len(full_tokens)}>{max_tokens}, truncated by tokenizer.")
    return trimmed


def main() -> None:
    outdir = Path(CONFIG["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # 稳定优先：MPS 下使用 float32，避免部分机器出现黑图/NaN。
    dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(CONFIG["model"], torch_dtype=dtype, use_safetensors=True).to(device)
    pipe.safety_checker = None
    # 降低内存占用，减少 MPS 分配失败概率。
    pipe.enable_attention_slicing()
    if device == "mps":
        torch.mps.empty_cache()
    generator = torch.Generator(device=device).manual_seed(CONFIG["seed"])
    width = align_to_multiple_of_8(CONFIG["width"])
    height = align_to_multiple_of_8(CONFIG["height"])
    prompt = clamp_text(CONFIG["prompt"], CONFIG["prompt_max_chars"], "prompt")
    negative_prompt = clamp_text(CONFIG["negative_prompt"], CONFIG["negative_prompt_max_chars"], "negative_prompt")
    prompt = clamp_text_by_tokens(pipe.tokenizer, prompt, CONFIG["max_tokens"], "prompt")
    negative_prompt = clamp_text_by_tokens(pipe.tokenizer, negative_prompt, CONFIG["max_tokens"], "negative_prompt")
    print(f"[INFO] Image size: {width}x{height}")

    start = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=CONFIG["steps"],
        guidance_scale=CONFIG["cfg"],
        generator=generator,
        output_type="np",
    )
    elapsed = time.time() - start

    img_arr = result.images[0]
    # 兜底：若出现 NaN/Inf，先替换后再保存，避免整张黑图。
    img_arr = np.nan_to_num(img_arr, nan=0.0, posinf=1.0, neginf=0.0)
    img_arr = np.clip(img_arr, 0.0, 1.0)
    image = Image.fromarray((img_arr * 255).astype(np.uint8))

    out = outdir / f"lesson01_seed{CONFIG['seed']}.png"
    image.save(out)
    print(f"[OK] Saved: {out}")
    print(f"[OK] Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
