#!/usr/bin/env python3
"""
第二课：参数对比实验（固定 seed，比较 steps/cfg）
"""

from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

CONFIG = {
    "model": "runwayml/stable-diffusion-v1-5",
    "prompt": "职场女性人像，头肩构图，人物居中，皮肤自然，清晰对焦，室内窗边光",
    "negative_prompt": "畸形，脸部变形，眼睛不对称，多余手指，手部畸形，模糊，低清晰度，水印，文字",
    "width": 384,
    "height": 512,
    "seed": 42,
    "steps_list": [20, 26],
    "cfg_list": [6.0, 6.8],
    "outdir": "./outputs/lesson02",
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

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(CONFIG["model"], torch_dtype=dtype, use_safetensors=True).to(device)
    pipe.safety_checker = None
    prompt = clamp_text(CONFIG["prompt"], CONFIG["prompt_max_chars"], "prompt")
    negative_prompt = clamp_text(CONFIG["negative_prompt"], CONFIG["negative_prompt_max_chars"], "negative_prompt")
    prompt = clamp_text_by_tokens(pipe.tokenizer, prompt, CONFIG["max_tokens"], "prompt")
    negative_prompt = clamp_text_by_tokens(pipe.tokenizer, negative_prompt, CONFIG["max_tokens"], "negative_prompt")

    for steps in CONFIG["steps_list"]:
        for cfg in CONFIG["cfg_list"]:
            gen = torch.Generator(device=device).manual_seed(CONFIG["seed"])
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=CONFIG["width"],
                height=CONFIG["height"],
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=gen,
            ).images[0]
            out = outdir / f"s{steps}_cfg{cfg}_seed{CONFIG['seed']}.png"
            image.save(out)
            print(f"[OK] Saved: {out}")


if __name__ == "__main__":
    main()
