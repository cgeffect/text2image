# 课程代码索引（python-sd）

按课程顺序运行：

1. `lesson01_txt2img.py`：第一课，最小文生图  
2. `lesson02_param_sweep.py`：第二课，参数对比实验  
3. `lesson03_img2img.py`：第三课，图生图  
4. `lesson04_lora_txt2img.py`：第四课，LoRA 风格增强  
5. `lesson05_controlnet_canny.py`：第五课，ControlNet 控构图  
6. `lesson06_image_to_video.py`：第六课，图生视频（SVD）

## 运行前准备

- 激活环境：
  - `conda activate /Users/tfwang/LLM/sd/sd-learn`
- 已安装基础依赖：
  - `diffusers transformers accelerate safetensors`

## 额外依赖（第 5、6 课建议安装）

```bash
python -m pip install opencv-python imageio[ffmpeg]
```

## 通用说明

- 每个脚本顶部都有 `CONFIG`，直接改参数
- 默认输出目录在 `python-sd/outputs/`
- 首次下载模型会较慢，属于正常现象
