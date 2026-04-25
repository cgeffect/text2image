#!/usr/bin/env python3
"""
第五课：ControlNet（Canny）控制构图
需要额外依赖: opencv-python
"""

import gc
from pathlib import Path

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image

try:
    import cv2
except ImportError as exc:
    raise RuntimeError("Please install opencv-python: python -m pip install opencv-python") from exc

PROFILE = "safe"  # 可选: safe / standard / quality
PROFILES = {
    "safe": {"width": 320, "height": 320, "steps": 14, "cfg": 6.5, "controlnet_scale": 0.7, "dtype_mode": "fp32"},
    "standard": {"width": 384, "height": 384, "steps": 18, "cfg": 7.0, "controlnet_scale": 0.8, "dtype_mode": "fp32"},
    "quality": {"width": 512, "height": 512, "steps": 24, "cfg": 7.5, "controlnet_scale": 0.9, "dtype_mode": "fp16_if_possible"},
}

CONFIG = {
    "base_model": "runwayml/stable-diffusion-v1-5",
    "controlnet_model": "lllyasviel/sd-controlnet-canny",
    "input_image": "./outputs/lesson01/lesson01_seed42.png",
    "prompt": "时尚人物肖像，头肩构图，保持原始构图，面部清晰，细节自然",
    "negative_prompt": "畸形，脸部变形，眼睛不对称，多余手指，手部畸形，模糊，低清晰度，水印，文字",
    "controlnet_scale": 0.7,  # 会被 PROFILE 覆盖
    "steps": 14,  # 会被 PROFILE 覆盖
    "cfg": 6.5,  # 会被 PROFILE 覆盖
    "seed": 42,
    "width": 320,  # 会被 PROFILE 覆盖
    "height": 320,  # 会被 PROFILE 覆盖
    "outdir": "./outputs/lesson05",
    "prompt_max_chars": 40,
    "negative_prompt_max_chars": 60,
    "max_tokens": 77,
}


def canny_image(path: Path) -> Image.Image:
    src = cv2.imread(str(path))
    if src is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    src = cv2.resize(src, (CONFIG["width"], CONFIG["height"]))
    edges = cv2.Canny(src, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)


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
    CONFIG["controlnet_scale"] = p["controlnet_scale"]

    outdir = Path(CONFIG["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    input_path = Path(CONFIG["input_image"])
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    control_image = canny_image(input_path)
    control_image.save(outdir / "control_canny.png")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype_mode = p["dtype_mode"]
    dtype = torch.float32 if dtype_mode == "fp32" else (torch.float16 if device == "mps" else torch.float32)
    prompt = clamp_text(CONFIG["prompt"], CONFIG["prompt_max_chars"], "prompt")
    negative_prompt = clamp_text(CONFIG["negative_prompt"], CONFIG["negative_prompt_max_chars"], "negative_prompt")
    print(f"[INFO] PROFILE={PROFILE}, size={CONFIG['width']}x{CONFIG['height']}, steps={CONFIG['steps']}, cfg={CONFIG['cfg']}")

    controlnet = ControlNetModel.from_pretrained(CONFIG["controlnet_model"], torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        CONFIG["base_model"], controlnet=controlnet, torch_dtype=dtype, use_safetensors=True
    ).to(device)
    pipe.safety_checker = None
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    prompt = clamp_text_by_tokens(pipe.tokenizer, prompt, CONFIG["max_tokens"], "prompt")
    negative_prompt = clamp_text_by_tokens(pipe.tokenizer, negative_prompt, CONFIG["max_tokens"], "negative_prompt")

    gen = torch.Generator(device=device).manual_seed(CONFIG["seed"])
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        controlnet_conditioning_scale=CONFIG["controlnet_scale"],
        num_inference_steps=CONFIG["steps"],
        guidance_scale=CONFIG["cfg"],
        generator=gen,
    ).images[0]

    out = outdir / f"controlnet_seed{CONFIG['seed']}.png"
    image.save(out)
    print(f"[OK] Saved: {out}")
    del pipe
    del controlnet
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
