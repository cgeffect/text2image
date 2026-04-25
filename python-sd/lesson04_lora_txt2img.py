#!/usr/bin/env python3
"""
第四课：LoRA 文生图
提示：先下载 LoRA 权重到本地，然后填写 lora_path。
"""

import gc
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

PROFILE = "safe"  # 可选: safe / standard / quality
PROFILES = {
    "safe": {"width": 320, "height": 384, "steps": 14, "cfg": 6.2, "dtype_mode": "fp32", "lora_scales": [0.5, 0.7]},
    "standard": {"width": 384, "height": 512, "steps": 20, "cfg": 6.5, "dtype_mode": "fp32", "lora_scales": [0.5, 0.7, 0.9]},
    "quality": {"width": 512, "height": 640, "steps": 26, "cfg": 7.0, "dtype_mode": "fp16_if_possible", "lora_scales": [0.6, 0.8, 1.0]},
}

CONFIG = {
    "model": "runwayml/stable-diffusion-v1-5",
    "lora_path": "./LoRA.safetensors",  # 例如: "/path/to/your_lora.safetensors"
    "lora_scales": [0.5, 0.7],  # 会被 PROFILE 覆盖
    # 人像防变形提示词（英文更贴近 SD1.5 训练语料，通常更稳定）
    "prompt": "cute sweet East Asian girl, close-up portrait, straight-on front view, facing camera, centered composition, full face visible, big bright eyes, gentle smile, pastel makeup, colorful, vibrant colors, soft daylight, sharp focus, photorealistic",
    "negative_prompt": "monochrome, black and white, grayscale, desaturated, profile, side view, off-center subject, cropped at edge, face cut off, partial face, asymmetrical face, deformed, blurry, low quality, watermark, text",
    "steps": 14,  # 会被 PROFILE 覆盖
    "seed": 42,
    "width": 320,  # 会被 PROFILE 覆盖
    "height": 320,  # 会被 PROFILE 覆盖
    "outdir": "./outputs/lesson04",
    "prompt_max_chars": 40,
    "negative_prompt_max_chars": 60,
    "max_tokens": 77,
}


def cleanup_pipe(pipe: StableDiffusionPipeline | None) -> None:
    if pipe is not None:
        del pipe
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


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
    if PROFILE not in PROFILES:
        raise ValueError(f"Invalid PROFILE={PROFILE}, choose from: {list(PROFILES.keys())}")
    p = PROFILES[PROFILE]
    CONFIG["width"] = p["width"]
    CONFIG["height"] = p["height"]
    CONFIG["steps"] = p["steps"]
    CONFIG["cfg"] = p["cfg"]
    CONFIG["lora_scales"] = p["lora_scales"]

    outdir = Path(CONFIG["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # safe/standard 默认稳定优先；quality 在 MPS 下尝试 float16。
    dtype_mode = p["dtype_mode"]
    if dtype_mode == "fp32":
        dtype = torch.float32
    else:
        dtype = torch.float16 if device == "mps" else torch.float32
    print(f"[INFO] PROFILE={PROFILE}, size={CONFIG['width']}x{CONFIG['height']}, steps={CONFIG['steps']}, cfg={CONFIG['cfg']}")
    prompt = clamp_text(CONFIG["prompt"], CONFIG["prompt_max_chars"], "prompt")
    negative_prompt = clamp_text(CONFIG["negative_prompt"], CONFIG["negative_prompt_max_chars"], "negative_prompt")

    if not CONFIG["lora_path"]:
        print("[WARN] lora_path is empty, this run uses base model only.")
        pipe = StableDiffusionPipeline.from_pretrained(CONFIG["model"], torch_dtype=dtype, use_safetensors=True).to(device)
        pipe.safety_checker = None
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        prompt_run = clamp_text_by_tokens(pipe.tokenizer, prompt, CONFIG["max_tokens"], "prompt")
        negative_run = clamp_text_by_tokens(pipe.tokenizer, negative_prompt, CONFIG["max_tokens"], "negative_prompt")
        gen = torch.Generator(device=device).manual_seed(CONFIG["seed"])
        image = pipe(
            prompt=prompt_run,
            negative_prompt=negative_run,
            width=CONFIG["width"],
            height=CONFIG["height"],
            num_inference_steps=CONFIG["steps"],
            guidance_scale=CONFIG["cfg"],
            generator=gen,
        ).images[0]
        out = outdir / f"base_seed{CONFIG['seed']}.png"
        image.save(out)
        print(f"[OK] Saved: {out}")
        cleanup_pipe(pipe)
        return

    try:
        import peft  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "LoRA 需要 PEFT 后端。请先安装: python -m pip install -U peft"
        ) from exc

    for scale in CONFIG["lora_scales"]:
        # 为了保证每个 scale 的结果独立可比，每轮都重新加载一次 pipeline。
        pipe = StableDiffusionPipeline.from_pretrained(CONFIG["model"], torch_dtype=dtype, use_safetensors=True).to(device)
        pipe.safety_checker = None
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        prompt_run = clamp_text_by_tokens(pipe.tokenizer, prompt, CONFIG["max_tokens"], "prompt")
        negative_run = clamp_text_by_tokens(pipe.tokenizer, negative_prompt, CONFIG["max_tokens"], "negative_prompt")
        pipe.load_lora_weights(CONFIG["lora_path"])
        pipe.fuse_lora(lora_scale=scale)
        print(f"[INFO] Loaded LoRA: {CONFIG['lora_path']} (scale={scale})")

        gen = torch.Generator(device=device).manual_seed(CONFIG["seed"])
        image = pipe(
            prompt=prompt_run,
            negative_prompt=negative_run,
            width=CONFIG["width"],
            height=CONFIG["height"],
            num_inference_steps=CONFIG["steps"],
            guidance_scale=CONFIG["cfg"],
            generator=gen,
        ).images[0]

        out = outdir / f"lora_scale{scale}_seed{CONFIG['seed']}.png"
        image.save(out)
        print(f"[OK] Saved: {out}")
        cleanup_pipe(pipe)


if __name__ == "__main__":
    main()
