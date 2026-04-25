#!/usr/bin/env python3
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

# 第 1 步：在这里配置参数（直接在 PyCharm 运行即可，无需命令行参数）。
CONFIG = {
    "model": "runwayml/stable-diffusion-v1-5",  # 使用的模型 ID（也可以换成本地模型目录）
    # 正向/反向各自约 77 CLIP tokens。亚洲面孔 + 人物/面部居中 + 全脸勿贴边（避免左右被裁）。
    # 仍裁脸时：换 seed，或略加宽高（保持 8 倍数），或略降 cfg。
    "prompt": "anime style, cute East Asian schoolgirl, face close-up portrait, straight-on front view, looking at viewer, centered symmetrical composition, full face visible, long loose hair, closed-mouth gentle smile, blue sky and white clouds background, vibrant colors, clean line art, soft shading, high detail",
    "negative_prompt": "photorealistic, realistic skin texture, adult woman, mature face, sexy, cleavage, arms, hands, fingers, upper body, full body, profile, side view, off-center, cropped face, open mouth, teeth, monochrome, black and white, blurry, low quality, watermark, text",
    "width": 512,
    "height": 640,  # 更紧的脸部竖构图（须为 8 的倍数）
    "steps": 24,
    "cfg": 7.8,  # 二次元风格可略高一点，增强风格一致性
    "seed": 123,  # 随机种子（固定后可复现结果，换 seed 可生成不同构图）
    "outdir": "./outputs",  # 输出目录（生成结果保存位置）
}


def main() -> None:
    # 第 2 步：创建输出目录（不存在就自动创建）。
    outdir = Path(CONFIG["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    # 第 3 步：选择运行设备。
    # Apple Silicon 优先使用 MPS；如果不可用则退回 CPU。
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"[INFO] Loading model: {CONFIG['model']}")
    print(f"[INFO] Device: {device}, dtype: {dtype}")

    # 第 4 步：加载 Stable Diffusion 模型并移动到目标设备。
    pipe = StableDiffusionPipeline.from_pretrained(
        CONFIG["model"],
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe = pipe.to(device)
    # 演示学习场景下关闭 safety_checker，避免额外依赖影响运行。
    pipe.safety_checker = None

    # 第 5 步：设置随机种子，确保结果可复现。
    generator = torch.Generator(device=device).manual_seed(CONFIG["seed"])

    # 第 6 步：根据 prompt 和参数生成图片。
    start = time.time()
    result = pipe(
        prompt=CONFIG["prompt"],
        negative_prompt=CONFIG["negative_prompt"],
        width=CONFIG["width"],
        height=CONFIG["height"],
        num_inference_steps=CONFIG["steps"],
        guidance_scale=CONFIG["cfg"],
        generator=generator,
    )
    elapsed = time.time() - start

    # 第 7 步：保存图片，并在文件名中记录关键参数。
    ts = int(time.time())
    output_path = outdir / f"sd_{ts}_seed{CONFIG['seed']}_{CONFIG['width']}x{CONFIG['height']}.png"
    result.images[0].save(output_path)

    print(f"[OK] Saved: {output_path}")
    print(f"[OK] Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
