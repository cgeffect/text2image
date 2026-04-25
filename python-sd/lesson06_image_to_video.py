#!/usr/bin/env python3
"""
第六课：图生视频（Stable Video Diffusion）
需要额外依赖: imageio[ffmpeg]
"""

import gc
from pathlib import Path

import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image

try:
    import imageio.v3 as iio
except ImportError as exc:
    raise RuntimeError("Please install imageio[ffmpeg]: python -m pip install imageio[ffmpeg]") from exc

PROFILE = "safe"  # 可选: safe / standard / quality
PROFILES = {
    "safe": {
        "width": 320, "height": 320, "num_frames": 8, "fps": 8,
        "motion_bucket_id": 48, "decode_chunk_size": 1, "dtype_mode": "fp32",
    },
    "standard": {
        "width": 384, "height": 384, "num_frames": 12, "fps": 8,
        "motion_bucket_id": 64, "decode_chunk_size": 2, "dtype_mode": "fp32",
    },
    "quality": {
        "width": 512, "height": 512, "num_frames": 16, "fps": 8,
        "motion_bucket_id": 80, "decode_chunk_size": 2, "dtype_mode": "fp16_if_possible",
    },
}

CONFIG = {
    "model": "stabilityai/stable-video-diffusion-img2vid-xt",
    "input_image": "./outputs/lesson01/lesson01_seed42.png",
    "num_frames": 8,  # 会被 PROFILE 覆盖
    "fps": 8,  # 会被 PROFILE 覆盖
    "motion_bucket_id": 48,  # 会被 PROFILE 覆盖
    "noise_aug_strength": 0.02,
    "decode_chunk_size": 1,  # 会被 PROFILE 覆盖
    "seed": 42,
    "width": 320,  # 会被 PROFILE 覆盖
    "height": 320,  # 会被 PROFILE 覆盖
    "outdir": "./outputs/lesson06",
}


def main() -> None:
    if PROFILE not in PROFILES:
        raise ValueError(f"Invalid PROFILE={PROFILE}, choose from: {list(PROFILES.keys())}")
    p = PROFILES[PROFILE]
    for k in ("width", "height", "num_frames", "fps", "motion_bucket_id", "decode_chunk_size"):
        CONFIG[k] = p[k]

    outdir = Path(CONFIG["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    input_path = Path(CONFIG["input_image"])
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if p["dtype_mode"] == "fp32" else (torch.float16 if device == "mps" else torch.float32)
    print(
        f"[INFO] PROFILE={PROFILE}, size={CONFIG['width']}x{CONFIG['height']}, "
        f"frames={CONFIG['num_frames']}, fps={CONFIG['fps']}"
    )

    image = Image.open(input_path).convert("RGB").resize((CONFIG["width"], CONFIG["height"]))
    pipe = StableVideoDiffusionPipeline.from_pretrained(CONFIG["model"], torch_dtype=dtype).to(device)
    pipe.enable_vae_slicing()

    gen = torch.Generator(device=device).manual_seed(CONFIG["seed"])
    frames = pipe(
        image=image,
        num_frames=CONFIG["num_frames"],
        motion_bucket_id=CONFIG["motion_bucket_id"],
        noise_aug_strength=CONFIG["noise_aug_strength"],
        decode_chunk_size=CONFIG["decode_chunk_size"],
        generator=gen,
    ).frames[0]

    out = outdir / f"svd_seed{CONFIG['seed']}.mp4"
    iio.imwrite(out, frames, fps=CONFIG["fps"])
    print(f"[OK] Saved video: {out}")
    del pipe
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
