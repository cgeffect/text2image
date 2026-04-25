#!/usr/bin/env python3
"""
LoRA 三实验对比脚本

实验 1: baseline(不加 LoRA) vs with LoRA
实验 2: LoRA 强度扫描 (0.5 / 0.7 / 0.9)
实验 3: 同 LoRA 多场景一致性
"""

import gc
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

CONFIG = {
    "model": "runwayml/stable-diffusion-v1-5",
    "lora_path": "",  # 例如: "/Users/tfwang/LLM/sd/models/loras/your_lora.safetensors"
    "seed": 42,
    # 低资源学习模式（MacBook Air 推荐）
    "width": 320,
    "height": 320,
    "steps": 14,
    "cfg": 6.5,
    "negative_prompt": "模糊，低清晰度，结构错误，多余手指，水印，文字",
    "outdir": "./outputs/lesson04_experiments",
    "exp1_prompt": "男性肖像摄影，电影感打光，面部清晰",
    "exp2_prompt": "女性肖像摄影，胶片风格，面部清晰",
    "exp2_scales": [0.5, 0.7],  # 低资源模式先比较两个常用强度
    "exp3_scale": 0.7,
    "run_experiment_1": True,
    "run_experiment_2": True,
    "run_experiment_3": True,
    "exp3_prompts": [
        "咖啡馆人物肖像，暖色环境光",
        "城市夜景人物肖像，柔和霓虹光",
    ],
    "prompt_max_chars": 36,
    "negative_prompt_max_chars": 50,
    "max_tokens": 77,
}


def make_pipe(model: str, device: str, dtype: torch.dtype) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=dtype, use_safetensors=True).to(device)
    pipe.safety_checker = None
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe


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


def generate_one(
    pipe: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg: float,
):
    prompt = clamp_text_by_tokens(pipe.tokenizer, prompt, CONFIG["max_tokens"], "prompt")
    negative_prompt = clamp_text_by_tokens(
        pipe.tokenizer, negative_prompt, CONFIG["max_tokens"], "negative_prompt"
    )
    gen = torch.Generator(device=pipe.device.type).manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=gen,
    ).images[0]


def main() -> None:
    outdir = Path(CONFIG["outdir"])
    exp1_dir = outdir / "exp1_baseline_vs_lora"
    exp2_dir = outdir / "exp2_lora_scale_sweep"
    exp3_dir = outdir / "exp3_multi_scene_consistency"
    for d in (exp1_dir, exp2_dir, exp3_dir):
        d.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # 稳定优先，避免 MPS 黑图和卡死
    dtype = torch.float32
    print(f"[INFO] Device={device}, dtype={dtype}")

    lora_path = CONFIG["lora_path"].strip()
    has_lora = bool(lora_path)
    if not has_lora:
        print("[WARN] lora_path 为空。实验 1/2/3 将退化为仅基础模型对比。")

    exp1_prompt = clamp_text(CONFIG["exp1_prompt"], CONFIG["prompt_max_chars"], "exp1_prompt")
    exp2_prompt = clamp_text(CONFIG["exp2_prompt"], CONFIG["prompt_max_chars"], "exp2_prompt")
    exp3_prompts = [clamp_text(p, CONFIG["prompt_max_chars"], f"exp3_prompt_{i+1}") for i, p in enumerate(CONFIG["exp3_prompts"])]
    negative_prompt = clamp_text(
        CONFIG["negative_prompt"], CONFIG["negative_prompt_max_chars"], "negative_prompt"
    )

    if CONFIG["run_experiment_1"]:
        print("[RUN] Experiment 1: baseline vs lora")
        pipe_base = make_pipe(CONFIG["model"], device, dtype)
        base_img = generate_one(
            pipe_base,
            exp1_prompt,
            negative_prompt,
            CONFIG["seed"],
            CONFIG["width"],
            CONFIG["height"],
            CONFIG["steps"],
            CONFIG["cfg"],
        )
        base_out = exp1_dir / f"baseline_seed{CONFIG['seed']}.png"
        base_img.save(base_out)
        print(f"[OK] {base_out}")
        cleanup_pipe(pipe_base)

        if has_lora:
            scale = CONFIG["exp2_scales"][min(1, len(CONFIG["exp2_scales"]) - 1)]
            pipe_lora = make_pipe(CONFIG["model"], device, dtype)
            pipe_lora.load_lora_weights(lora_path)
            pipe_lora.fuse_lora(lora_scale=scale)
            lora_img = generate_one(
                pipe_lora,
                exp1_prompt,
                negative_prompt,
                CONFIG["seed"],
                CONFIG["width"],
                CONFIG["height"],
                CONFIG["steps"],
                CONFIG["cfg"],
            )
            lora_out = exp1_dir / f"with_lora_scale{scale}_seed{CONFIG['seed']}.png"
            lora_img.save(lora_out)
            print(f"[OK] {lora_out}")
            cleanup_pipe(pipe_lora)

    # 实验 2：强度扫描
    if CONFIG["run_experiment_2"]:
        print("[RUN] Experiment 2: lora scale sweep")
        if has_lora:
            for scale in CONFIG["exp2_scales"]:
                pipe = make_pipe(CONFIG["model"], device, dtype)
                pipe.load_lora_weights(lora_path)
                pipe.fuse_lora(lora_scale=scale)
                img = generate_one(
                    pipe,
                    exp2_prompt,
                    negative_prompt,
                    CONFIG["seed"],
                    CONFIG["width"],
                    CONFIG["height"],
                    CONFIG["steps"],
                    CONFIG["cfg"],
                )
                out = exp2_dir / f"scale{scale}_seed{CONFIG['seed']}.png"
                img.save(out)
                print(f"[OK] {out}")
                cleanup_pipe(pipe)
        else:
            print("[SKIP] lora_path 为空，跳过实验 2。")

    # 实验 3：多场景一致性
    if CONFIG["run_experiment_3"]:
        print("[RUN] Experiment 3: multi-scene consistency")
        if has_lora:
            pipe_scene = make_pipe(CONFIG["model"], device, dtype)
            pipe_scene.load_lora_weights(lora_path)
            pipe_scene.fuse_lora(lora_scale=CONFIG["exp3_scale"])
        else:
            pipe_scene = make_pipe(CONFIG["model"], device, dtype)

        for i, prompt in enumerate(exp3_prompts, start=1):
            img = generate_one(
                pipe_scene,
                prompt,
                negative_prompt,
                CONFIG["seed"],
                CONFIG["width"],
                CONFIG["height"],
                CONFIG["steps"],
                CONFIG["cfg"],
            )
            tag = "with_lora" if has_lora else "base"
            out = exp3_dir / f"scene{i}_{tag}_seed{CONFIG['seed']}.png"
            img.save(out)
            print(f"[OK] {out}")
        cleanup_pipe(pipe_scene)

    print("[DONE] All experiments finished.")


if __name__ == "__main__":
    main()
