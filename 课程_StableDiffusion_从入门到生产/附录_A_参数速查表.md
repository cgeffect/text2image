# 附录 A：参数速查表（常用）

## 文生图核心

- `model`：模型
- `prompt`：正向提示词
- `negative_prompt`：反向提示词
- `width/height`：分辨率
- `steps`：采样步数
- `cfg`：提示词遵循强度
- `seed`：随机种子（复现关键）

## 进阶控制

- `sampler`：采样器（影响风格、细节、速度）
- `scheduler`：调度策略
- `num_images_per_prompt`：单次出图数量
- `clip_skip`：部分模型常用

## img2img / inpainting

- `init_image`：输入图
- `strength`：重绘强度
- `mask_image`：局部重绘掩码

## LoRA

- `lora_path`：LoRA 权重路径
- `lora_scale`：LoRA 强度（通常 0.4~1.0）

## ControlNet

- `control_image`：控制图
- `controlnet_model`：控制模型
- `controlnet_scale`：控制强度

## 视频

- `num_frames`：帧数
- `fps`：帧率
- `motion_strength`：运动幅度
- `consistency`：跨帧一致性相关参数（因工作流不同而异）
