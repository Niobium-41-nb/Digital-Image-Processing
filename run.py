"""
============================================================
主程序 —— 时间-空间联合卷积的运动感知与边缘检测流水线
============================================================

本程序实现了一套完整的视频运动分析与边缘检测流水线，主要流程如下：

  1. 生成阶段 —— 创建多种复杂运动模式的测试视频
  2. GUI 选择阶段 —— 弹出窗口选择要处理的视频文件
  3. 处理阶段：
        a) 图像放大 —— 将每帧图像放大 SCALE_FACTOR 倍
        b) 时间维度运动模糊 —— 三维卷积核沿时间轴平滑
        c) 空间边缘检测 —— 水平/垂直双方向 Prewitt 卷积核
        d) 帧间差分 —— 相邻帧差分绝对值，反映运动强度
        e) 最大池化还原 —— 将图像大小还原到原始尺寸
        f) RGB 三通道融合 —— R=水平边缘, G=垂直边缘, B=帧间差分
  4. 三个处理副本：
        - 副本A：原始时间-空间结构 (T, H, W)
        - 副本B：转置Y轴与时间轴 (H, T, W)
        - 副本C：转置X轴与时间轴 (W, H, T)

配置方式：
  卷积核大小、放大倍数等参数通过 .env 文件配置。
  视频文件通过 GUI 窗口选择。

依赖模块：
  - src/config.py           : 配置加载
  - src/video_generator.py  : 视频生成
  - src/video_selector.py   : GUI 视频选择
  - src/video_converter.py  : MP4 与 NumPy 数组转换
  - src/temporal_convolver.py : 三维卷积与图像放大
  - src/edge_process.py     : 方向边缘检测与流水线处理

作者：数字图像处理课程项目
============================================================
"""

import gc
import os
import shutil

import numpy as np

from src.config import get_config
from src.video_generator import generate_all_test_videos
from src.video_selector import select_video_simple
from src.video_converter import mp4_to_grayscale_array
from src.temporal_convolver import (
    create_temporal_motion_blur, scale_and_convolve, apply_temporal_convolution
)
from src.edge_process import process_motion_blurred_array

# ============================================================
# 加载配置
# ============================================================
config = get_config()
CONVOLUTION_SIZE = config['CONVOLUTION_SIZE']
SCALE_FACTOR = config['SCALE_FACTOR']
POOL_SIZE = config['POOL_SIZE']

# ============================================================
# 第一步：生成测试视频
# ============================================================
generate_all_test_videos()

# ============================================================
# 第二步：GUI 选择视频与参数
# ============================================================
print("\n正在打开视频选择窗口...")
selected_video_name, selected_video_path, gui_conv_min, gui_conv_max, \
    enable_transposed, enable_transposed_x, \
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

# 确定卷积核大小（GUI 区间选择优先）
convolution_sizes = list(range(gui_conv_min, gui_conv_max + 1))

print(f"\n已选择卷积核大小区间: {gui_conv_min} ~ {gui_conv_max}")
print(f"将依次处理: {convolution_sizes}")
print(f"图像放大: {'启用' if enable_scale else '禁用'} (倍数: {actual_scale_factor}x)")
print(f"池化大小: {actual_pool_size}x{actual_pool_size}")
print(f"副本B (Y-T转置): {'启用' if enable_transposed else '禁用'}")
print(f"副本C (X-T转置): {'启用' if enable_transposed_x else '禁用'}")
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

# 定义处理函数：根据是否放大选择不同路径
def process_with_or_without_scale(input_arr, scale_factor, convolution_kernel):
    """根据 scale_factor 决定是否放大后卷积，或直接卷积"""
    if scale_factor > 1:
        return scale_and_convolve(input_arr, scale_factor, convolution_kernel)
    else:
        return apply_temporal_convolution(input_arr, convolution_kernel)


# 遍历卷积核大小
for convolution_size in convolution_sizes:
    output_dir = f"output/{selected_video_name}/{convolution_size}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"卷积核大小: {convolution_size} | 放大倍数: {actual_scale_factor}x | 池化大小: {actual_pool_size}x{actual_pool_size}")
    print(f"{'=' * 70}")

    # 创建时间维度运动模糊卷积核
    convolution = create_temporal_motion_blur(convolution_size, convolution_size, convolution_size)

    # ========== 副本A：原始时间-空间结构 ==========
    print(f"\n>>> 处理副本A（未转置）- 原始时间-空间结构")
    processed_arr = process_with_or_without_scale(arr, actual_scale_factor, convolution)
    process_motion_blurred_array(
        processed_arr, output_dir, suffix="",
        pool_size=actual_pool_size,
        enable_diff=enable_diff
    )

    # 释放副本A内存（后续不再需要）
    del processed_arr
    gc.collect()

    # ========== 副本B：转置Y轴与时间轴（可选） ==========
    if enable_transposed:
        print(f"\n>>> 处理副本B（先转置Y轴与时间轴，再运动模糊）")
        # 原始形状: (T, H, W) → 转置后: (H, T, W)
        transposed_arr = np.transpose(arr, axes=(1, 0, 2))
        print(f"转置前形状: {arr.shape} → 转置后形状: {transposed_arr.shape}")

        transposed_processed = process_with_or_without_scale(transposed_arr, actual_scale_factor, convolution)
        process_motion_blurred_array(
            transposed_processed, output_dir, suffix="_transposed",
            pool_size=actual_pool_size,
            enable_diff=enable_diff
        )

        del transposed_arr, transposed_processed
        gc.collect()
    else:
        print(f"\n>>> 跳过副本B（用户未选择）")

    # ========== 副本C：转置X轴与时间轴（可选） ==========
    if enable_transposed_x:
        print(f"\n>>> 处理副本C（先转置X轴与时间轴，再运动模糊）")
        # 原始形状: (T, H, W) → 转置后: (W, H, T)
        transposed_x_arr = np.transpose(arr, axes=(2, 1, 0))
        print(f"转置前形状: {arr.shape} → 转置后形状: {transposed_x_arr.shape}")

        transposed_x_processed = process_with_or_without_scale(transposed_x_arr, actual_scale_factor, convolution)
        process_motion_blurred_array(
            transposed_x_processed, output_dir, suffix="_transposed_x",
            pool_size=actual_pool_size,
            enable_diff=enable_diff
        )

        del transposed_x_arr, transposed_x_processed
        gc.collect()
    else:
        print(f"\n>>> 跳过副本C（用户未选择）")

# ========== 处理完成 ==========
print(f"\n{'=' * 70}")
print("所有视频处理完成！")
print(f"{'=' * 70}")
