# 第五课：ControlNet 与构图控制

## 目标

- 学会“可控生成”，不再只靠运气
- 能基于姿态图/边缘图/深度图控制结果结构

## ControlNet 解决什么问题

普通文生图常见问题：
- 构图不稳定
- 姿态不可控
- 线稿还原不准

ControlNet 的作用是给模型“结构约束”。

## 常见控制类型

- Canny：按边缘控制构图
- OpenPose：按人体骨架控制姿态
- Depth：按深度结构控制空间关系
- Lineart：按线稿上色与风格化

## 常用参数

- `control_image`：控制输入图
- `controlnet_model`：对应控制模型
- `controlnet_scale`：控制强度（0~1+）
- `start/end`：控制在哪个采样阶段生效

## 练习路径

1. 从 Canny 开始（最容易看到效果）
2. 再做 OpenPose 人物姿态
3. 最后尝试 Depth 做空间层次

## 作业

- 用同一 prompt：
  - 不加 ControlNet 生成 2 张
  - 加 Canny 生成 2 张
  - 对比构图可控性

## 验收标准

- 你能稳定复现指定姿态/构图
