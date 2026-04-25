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

PROFILE = "ultra_safe"  # 可选: ultra_safe / safe / standard / quality
PROFILES = {
    "ultra_safe": {
        "width": 192,
        "height": 192,
        "num_frames": 4,
        "fps": 4,
        "motion_bucket_id": 24,
        "decode_chunk_size": 1,
        "dtype_mode": "fp32",
        "device_mode": "cpu",
    },
    "safe": {
        "width": 256, "height": 256, "num_frames": 6, "fps": 6,
        "motion_bucket_id": 32, "decode_chunk_size": 1, "dtype_mode": "fp32", "device_mode": "cpu",
    },
    "standard": {
        "width": 384, "height": 384, "num_frames": 12, "fps": 8,
        "motion_bucket_id": 64, "decode_chunk_size": 2, "dtype_mode": "fp32", "device_mode": "mps_if_available",
    },
    "quality": {
        "width": 512, "height": 512, "num_frames": 16, "fps": 8,
        "motion_bucket_id": 80, "decode_chunk_size": 2, "dtype_mode": "fp16_if_possible", "device_mode": "mps_if_available",
    },
}

CONFIG = {
    "model": "stabilityai/stable-video-diffusion-img2vid-xt",
    "input_image": "./LOREAL.JPG",
    "num_frames": 8,  # 会被 PROFILE 覆盖
    "fps": 8,  # 会被 PROFILE 覆盖
    "target_seconds": 1,  # 广告时长（秒）
    "motion_bucket_id": 48,  # 会被 PROFILE 覆盖
    "noise_aug_strength": 0.02,
    "decode_chunk_size": 1,  # 会被 PROFILE 覆盖
    "seed": 42,
    "width": 320,  # 会被 PROFILE 覆盖
    "height": 320,  # 会被 PROFILE 覆盖
    "outdir": "./outputs/lesson06",
}


def apply_ad_camera_motion(frames, width: int, height: int):
    """为广告感增加轻微慢推近 + 摇镜（Ken Burns 风格）。"""
    out = []
    total = max(1, len(frames))
    max_zoom = 1.10
    pan_px = 8
    for i, frame in enumerate(frames):
        # 统一转 PIL，兼容 numpy / PIL 两种输入。
        img = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
        img = img.convert("RGB")

        t = i / max(1, total - 1)
        zoom = 1.0 + (max_zoom - 1.0) * t
        w_zoom = int(width / zoom)
        h_zoom = int(height / zoom)
        # 轻微横向摇镜：左 -> 右，幅度很小，避免突兀。
        dx = int((-pan_px) + (2 * pan_px) * t)
        cx = width // 2 + dx
        cy = height // 2
        left = max(0, min(width - w_zoom, cx - w_zoom // 2))
        top = max(0, min(height - h_zoom, cy - h_zoom // 2))
        cropped = img.crop((left, top, left + w_zoom, top + h_zoom))
        moved = cropped.resize((width, height), Image.Resampling.LANCZOS)
        out.append(moved)
    return out


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

    # 限制 CPU 线程，避免风扇拉满和系统过热。
    torch.set_num_threads(2)
    if p.get("device_mode") == "cpu":
        device = "cpu"
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if p["dtype_mode"] == "fp32" else (torch.float16 if device == "mps" else torch.float32)
    print(
        f"[INFO] PROFILE={PROFILE}, size={CONFIG['width']}x{CONFIG['height']}, "
        f"base_frames={CONFIG['num_frames']}, fps={CONFIG['fps']}, target_seconds={CONFIG['target_seconds']}"
    )

    image = Image.open(input_path).convert("RGB").resize((CONFIG["width"], CONFIG["height"]))
    pipe = StableVideoDiffusionPipeline.from_pretrained(CONFIG["model"], torch_dtype=dtype).to(device)
    if hasattr(pipe, "enable_vae_slicing"):
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

    # 目标广告时长：不足目标秒数时循环片段补齐（保持稳定、低资源）。
    target_total_frames = max(1, int(CONFIG["fps"] * CONFIG["target_seconds"]))
    if len(frames) < target_total_frames:
        tiled = []
        while len(tiled) < target_total_frames:
            tiled.extend(frames)
        frames = tiled[:target_total_frames]
    else:
        frames = frames[:target_total_frames]

    # 广告镜头感：在已生成帧上做轻微慢推近+摇镜，不增加模型推理开销。
    frames = apply_ad_camera_motion(frames, CONFIG["width"], CONFIG["height"])

    out = outdir / f"svd_seed{CONFIG['seed']}_{CONFIG['target_seconds']}s.mp4"
    iio.imwrite(out, frames, fps=CONFIG["fps"])
    print(f"[OK] Saved video: {out}")
    del pipe
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
