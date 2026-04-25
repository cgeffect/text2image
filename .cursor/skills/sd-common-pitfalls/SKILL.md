---
name: sd-common-pitfalls
description: Prevent common Stable Diffusion mistakes for local Mac workflows. Use when editing or running SD scripts, tuning prompts/parameters, or debugging generation issues such as token truncation, width/height divisibility, MPS memory failures, black images, and invalid LoRA checkpoints.
---

# Stable Diffusion Common Pitfalls (Mac)

## When to apply

Apply this checklist before running or editing SD scripts that generate images/videos.

## Pre-run checklist

1. **Prompt length**
   - SD1.5 commonly truncates around ~77 tokens.
   - Keep prompt short; put key constraints first.
2. **Resolution validity**
   - `width` and `height` must be divisible by `8`.
   - Prefer low-resource defaults on Mac Air: `320x320`, `384x512`.
3. **Resource profile**
   - Use `safe` profile first (low steps, low resolution).
   - Scale quality only after stable outputs.
4. **MPS stability**
   - Prefer `float32` for stability when black images/crashes appear.
   - Enable slicing when available: attention/vae slicing.
5. **Memory cleanup**
   - After generation loops: delete pipeline refs, run `gc.collect()`, call `torch.mps.empty_cache()` on MPS.

## Prompt quality guardrails (portrait tasks)

- Use explicit portrait constraints: head-and-shoulders, centered composition, symmetrical face, sharp focus.
- Add deformation negatives: bad anatomy, malformed face/hands, extra fingers, blurry, watermark/text.
- If results drift, reduce CFG slightly before increasing complexity.

## LoRA-specific checks

1. Ensure `peft` is installed.
2. Confirm file is real LoRA (not full checkpoint).
3. Confirm LoRA base compatibility (e.g., SD1.5 LoRA with SD1.5 base).
4. Start with scales `0.5~0.7`; avoid high scale first.

## Error-to-fix quick map

- **Token truncated**
  - Shorten prompt, move critical words earlier.
- **`width/height ... divisible by 8`**
  - Adjust to valid multiples of 8.
- **MPS alloc failure / system freeze**
  - Lower resolution/steps, use safe profile, release memory.
- **Black image / NaN warning**
  - Switch to float32, clamp/clean output path if needed.
- **Invalid LoRA checkpoint**
  - Replace with proper LoRA `.safetensors` file.

## Default safe starting point (portrait)

- Resolution: `320x384` or `384x512`
- Steps: `14~20`
- CFG: `6.2~6.8`
- Seed: fixed for comparison
