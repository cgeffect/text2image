#!/usr/bin/env python3
"""
第三课：图生图（img2img）
使用前请准备一张输入图。
"""

from pathlib import Path

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

CONFIG = {
    "model": "runwayml/stable-diffusion-v1-5",
    "input_image": "./outputs/lesson01/lesson01_seed42.png",
    "prompt": "同一人物人像，头肩构图，电影感侧光，肤质自然，面部清晰",
    "negative_prompt": "畸形，脸部变形，眼睛不对称，多余手指，手部畸形，模糊，水印，文字",
    "strength": 0.35,
    "steps": 20,
    "cfg": 6.5,
    "seed": 42,
    "outdir": "./outputs/lesson03",
    "prompt_max_chars": 40,
    "negative_prompt_max_chars": 60,
    "max_tokens": 77,
}


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

    input_path = Path(CONFIG["input_image"])
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    init_image = Image.open(input_path).convert("RGB").resize((512, 512))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        CONFIG["model"], torch_dtype=dtype, use_safetensors=True
    ).to(device)
    pipe.safety_checker = None
    prompt = clamp_text(CONFIG["prompt"], CONFIG["prompt_max_chars"], "prompt")
    negative_prompt = clamp_text(CONFIG["negative_prompt"], CONFIG["negative_prompt_max_chars"], "negative_prompt")
    prompt = clamp_text_by_tokens(pipe.tokenizer, prompt, CONFIG["max_tokens"], "prompt")
    negative_prompt = clamp_text_by_tokens(pipe.tokenizer, negative_prompt, CONFIG["max_tokens"], "negative_prompt")

    gen = torch.Generator(device=device).manual_seed(CONFIG["seed"])
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=CONFIG["strength"],
        num_inference_steps=CONFIG["steps"],
        guidance_scale=CONFIG["cfg"],
        generator=gen,
    ).images[0]

    out = outdir / f"img2img_strength{CONFIG['strength']}_seed{CONFIG['seed']}.png"
    image.save(out)
    print(f"[OK] Saved: {out}")


if __name__ == "__main__":
    main()
