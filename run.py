"""
============================================================
主程序 —— 基于3D卷积的运动感知与边缘检测流水线
============================================================

本程序将边缘检测视为 **3D卷积核 (T, 3, 3) 对 3D矩阵 (frames, H, W) 的卷积** 操作。

核心设计：
  传统的"时间运动模糊 + 空间边缘检测"是分离的两步操作。
  本程序将时间平滑核与空间边缘检测核通过 **外积 (outer product)**
  组合为统一的 3D 卷积核 (T, 3, 3)，一次卷积同时完成时间平滑和空间边缘检测。

主要流程如下：

  1. GUI 选择阶段 —— 弹出窗口选择要处理的视频文件
  2. 处理阶段：
         a) 图像放大 —— 将每帧图像放大 SCALE_FACTOR 倍
         b) 3D卷积边缘检测 —— 使用 3D 卷积核 (T, 3, 3) 对视频做真正的3D卷积，
            同时完成时间平滑和空间方向边缘检测（8方向）
         c) 帧间差分 —— 相邻帧差分绝对值，反映运动强度
         d) 最大池化还原 —— 将图像大小还原到原始尺寸
         e) 输出灰度视频 —— 各方向边缘检测结果及帧间差分分别输出为独立灰度视频

配置方式：
  卷积核大小（时间维度T）、放大倍数等参数通过 .env 文件配置。
  视频文件通过 GUI 窗口选择。

依赖模块：
  - src/config.py           : 配置加载
  - src/video_selector.py   : GUI 视频选择
  - src/video_converter.py  : MP4 与 NumPy 数组转换
  - src/temporal_convolver.py : 图像放大
  - src/edge_process.py     : 3D卷积边缘检测核与流水线处理

作者：数字图像处理课程项目
============================================================
"""

import gc
import os
import shutil

import numpy as np

from src.config import get_config
from src.video_selector import select_video_simple
from src.video_converter import mp4_to_grayscale_array
from src.temporal_convolver import scale_and_convolve
from src.edge_process import process_motion_blurred_array

# ============================================================
# 加载配置
# ============================================================
config = get_config()
CONVOLUTION_SIZE = config['CONVOLUTION_SIZE']
SCALE_FACTOR = config['SCALE_FACTOR']
POOL_SIZE = config['POOL_SIZE']

# ============================================================
# 第一步：GUI 选择视频与参数
# ============================================================
print("\n正在打开视频选择窗口...")
selected_video_name, selected_video_path, gui_conv_min, gui_conv_max, \
    enable_scale, gui_scale_factor, \
    enable_diff = select_video_simple(
    "选择待处理的视频", default_conv_size=CONVOLUTION_SIZE,
    default_scale_factor=SCALE_FACTOR
)

if selected_video_name is None:
    print("\n用户取消了视频选择，程序退出。")
    exit(0)

# 确定实际使用的放大倍数和池化大小
if enable_scale:
    actual_scale_factor = gui_scale_factor
    actual_pool_size = gui_scale_factor  # 池化大小跟随放大倍数
else:
    actual_scale_factor = 1  # 不放大
    actual_pool_size = 1  # 不放大时不进行池化（pool_size=1 表示不做降采样）

print(f"\n已选择视频: {selected_video_name}")
print(f"视频路径: {selected_video_path}")

# 确定3D卷积核的时间维度大小 T（GUI 区间选择优先）
# 注意：这里的 convolution_size 现在代表 3D 卷积核的时间维度 T
temporal_sizes = list(range(gui_conv_min, gui_conv_max + 1))

print(f"\n已选择3D卷积核时间维度 T 区间: {gui_conv_min} ~ {gui_conv_max}")
print(f"将依次处理 T = {temporal_sizes}")
print(f"图像放大: {'启用' if enable_scale else '禁用'} (倍数: {actual_scale_factor}x)")
print(f"池化大小: {actual_pool_size}x{actual_pool_size}")
print(f"帧间差分: {'启用' if enable_diff else '禁用'}")

# 读取视频
arr = mp4_to_grayscale_array(selected_video_path)

# ========== 将原始视频复制到 output 目录 ==========
video_output_dir = f"output/{selected_video_name}"
os.makedirs(video_output_dir, exist_ok=True)
original_video_output_path = os.path.join(video_output_dir, f"{selected_video_name}.mp4")
if not os.path.exists(original_video_output_path):
    shutil.copy2(selected_video_path, original_video_output_path)
    print(f"\n原始视频已复制到: {original_video_output_path}")
else:
    print(f"\n原始视频已存在: {original_video_output_path}，跳过复制")

# 遍历3D卷积核的时间维度大小 T
for temporal_size in temporal_sizes:
    output_dir = f"output/{selected_video_name}/{temporal_size}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"3D卷积核时间维度 T={temporal_size} | 放大倍数: {actual_scale_factor}x | 池化大小: {actual_pool_size}x{actual_pool_size}")
    print(f"{'=' * 70}")

    # ========== 图像放大（如果需要） ==========
    if actual_scale_factor > 1:
        print(f"\n>>> 开始图像放大 ({actual_scale_factor}x)")
        # 放大时不需要时间卷积，只做放大
        # 创建一个单位时间核（长度为1，不做时间平滑，因为3D核会处理）
        identity_kernel = np.ones((1, 1, 1), dtype=np.float64)
        processed_arr = scale_and_convolve(arr, actual_scale_factor, identity_kernel)
    else:
        processed_arr = arr.copy()

    # ========== 3D卷积边缘检测处理 ==========
    # process_motion_blurred_array 内部使用 3D 卷积核 (T, 3, 3)
    # 对视频做真正的3D卷积，一次调用同时完成时间平滑和空间边缘检测
    print(f"\n>>> 开始3D卷积边缘检测处理 (T={temporal_size})")
    process_motion_blurred_array(
        processed_arr, output_dir, suffix="",
        pool_size=actual_pool_size,
        enable_diff=enable_diff,
        temporal_size=temporal_size
    )

    # 释放内存
    if actual_scale_factor > 1:
        del processed_arr
    gc.collect()

# ========== 处理完成 ==========
print(f"\n{'=' * 70}")
print("所有视频处理完成！")
print(f"{'=' * 70}")

# 自动打开 output 文件夹
output_path = os.path.abspath(f"output/{selected_video_name}")
print(f"\n正在打开输出文件夹: {output_path}")
os.startfile(output_path)
