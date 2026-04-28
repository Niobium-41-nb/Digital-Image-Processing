"""
============================================================
边缘处理模块 —— 方向边缘检测核与流水线处理函数
============================================================

提供垂直/水平两个方向的 Prewitt 边缘检测卷积核定义，
以及完整的边缘检测→帧间差分→池化还原→RGB融合流水线函数。

使用分离卷积优化：将 3x3 Prewitt 核拆分为 (3,1) × (1,3)，
计算量从 O(9×H×W) 降至 O(3×H×W + 3×H×W) = O(6×H×W)。

依赖：
  - numpy, scipy
  - src/video_converter.py
============================================================
"""

import gc
import os
import sys

import numpy as np
from scipy.ndimage import convolve1d, maximum_filter

from src.progress_bar import update_progress, finish_progress
from src.video_converter import gray_array_to_mp4, three_gray_array_to_RGB_mp4


# ============================================================
# 双方向 Prewitt 边缘检测 —— 分离卷积核 (3,1) × (1,3)
# ============================================================
# Prewitt 3x3 核是可分离的，拆分为两个一维卷积核：
#
# 水平方向边缘检测（检测垂直边缘）：
#   完整核:  [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
#   分离为: 垂直平滑 [1, 1, 1]ᵀ × 水平差分 [-1, 0, 1]
#   先沿 axis=1（高度方向）用 [1,1,1] 平滑
#   再沿 axis=2（宽度方向）用 [-1,0,1] 差分
#
# 垂直方向边缘检测（检测水平边缘）：
#   完整核:  [[-1,-1,-1], [ 0, 0, 0], [ 1, 1, 1]]
#   分离为: 垂直差分 [-1, 0, 1]ᵀ × 水平平滑 [1, 1, 1]
#   先沿 axis=1（高度方向）用 [-1,0,1] 差分
#   再沿 axis=2（宽度方向）用 [1,1,1] 平滑

# 水平方向：垂直平滑核（沿高度轴）
KERNEL_H_SMOOTH = np.array([1.0, 1.0, 1.0])
# 水平方向：水平差分核（沿宽度轴）
KERNEL_H_DIFF = np.array([-1.0, 0.0, 1.0])

# 垂直方向：垂直差分核（沿高度轴）
KERNEL_V_DIFF = np.array([-1.0, 0.0, 1.0])
# 垂直方向：水平平滑核（沿宽度轴）
KERNEL_V_SMOOTH = np.array([1.0, 1.0, 1.0])


def process_motion_blurred_array(blurred_arr, output_dir, suffix="",
                                  pool_size=4, max_memory=5 * 1024 ** 3,
                                  enable_diff=True):
    """
    对运动模糊后的数组执行双方向边缘检测、帧间差分和RGB融合。
    流式流水线处理：边缘检测 → 池化 → 差分 合并为单次遍历，
    避免存储大尺寸中间结果。

    参数:
        blurred_arr: 运动模糊后的三维数组，形状 (frames, height, width)
        output_dir: 输出目录路径
        suffix: 输出文件名的后缀（用于区分转置/未转置的结果）
        pool_size: 最大池化核大小（pool_size=1 时不进行池化降采样）
        max_memory: 最大允许内存（字节）
        enable_diff: 是否计算帧间差分（False 时 B 通道置零）
    """
    num_frames = blurred_arr.shape[0]
    height, width = blurred_arr.shape[1], blurred_arr.shape[2]

    # 池化后的目标尺寸
    target_height = height // pool_size
    target_width = width // pool_size

    # 计算分块大小：每块需要保留上一帧用于差分，以及边缘检测临时内存
    # 每帧边缘检测需要：1 输入帧 + 2 个临时结果（水平/垂直各一次 convolve1d）
    bytes_per_frame = height * width * 4 * 4  # 输入帧 + 2临时 + 1备用
    chunk_size = max(1, max_memory // bytes_per_frame)
    chunk_size = min(chunk_size, 15)  # 大图下保守分块

    if pool_size > 1:
        print(f"开始流式处理{suffix}，共 {num_frames} 帧，"
              f"尺寸 {height}x{width} → 池化后 {target_height}x{target_width}，"
              f"分块 {chunk_size} 帧")
    else:
        print(f"开始边缘检测{suffix}，共 {num_frames} 帧，"
              f"尺寸 {height}x{width}（不进行池化降采样），"
              f"分块 {chunk_size} 帧")

    # 预分配结果数组（pool_size=1 时保持原尺寸）
    result_horizontal = np.zeros((num_frames, target_height, target_width), dtype=np.float32)
    result_vertical = np.zeros((num_frames, target_height, target_width), dtype=np.float32)
    if enable_diff:
        result_diff = np.zeros((num_frames, target_height, target_width), dtype=np.float32)

    # 流式处理：逐块完成 边缘检测 + 池化 + 差分
    # 为支持帧间差分，需要保留上一块的最后一帧
    prev_frame = None

    for start in range(0, num_frames, chunk_size):
        end = min(start + chunk_size, num_frames)

        # 处理当前块中的每一帧
        for i in range(start, end):
            frame = blurred_arr[i].astype(np.float32)

            # === 步骤1：双方向边缘检测（分离卷积） ===
            # 水平边缘：垂直平滑 → 水平差分
            temp_h = convolve1d(frame, KERNEL_H_SMOOTH, axis=0, mode='constant', cval=0.0)
            edge_h = convolve1d(temp_h, KERNEL_H_DIFF, axis=1, mode='constant', cval=0.0)

            # 垂直边缘：垂直差分 → 水平平滑
            temp_v = convolve1d(frame, KERNEL_V_DIFF, axis=0, mode='constant', cval=0.0)
            edge_v = convolve1d(temp_v, KERNEL_V_SMOOTH, axis=1, mode='constant', cval=0.0)

            # === 步骤2：池化（pool_size>1 时降采样，否则保持原尺寸） ===
            if pool_size > 1:
                result_horizontal[i] = maximum_filter(edge_h, size=pool_size, mode='constant', cval=0.0)[::pool_size, ::pool_size]
                result_vertical[i] = maximum_filter(edge_v, size=pool_size, mode='constant', cval=0.0)[::pool_size, ::pool_size]
            else:
                result_horizontal[i] = edge_h
                result_vertical[i] = edge_v

            # === 步骤3：帧间差分（流式，利用上一帧） ===
            if enable_diff:
                if i == 0:
                    # 第一帧：差分结果为 0
                    result_diff[i] = np.zeros((target_height, target_width), dtype=np.float32)
                else:
                    # 与上一帧做差分（上一帧可能是 prev_frame 或块内前一帧）
                    if i == start and prev_frame is not None:
                        prev = prev_frame
                    else:
                        prev = blurred_arr[i - 1].astype(np.float32)
                    diff_frame = np.abs(frame - prev)
                    if pool_size > 1:
                        result_diff[i] = maximum_filter(diff_frame, size=pool_size, mode='constant', cval=0.0)[::pool_size, ::pool_size]
                    else:
                        result_diff[i] = diff_frame

            # 释放临时大尺寸变量
            del temp_h, edge_h, temp_v, edge_v

            # 每帧更新进度条
            prefix = f"边缘检测+池化{suffix}" if pool_size > 1 else f"边缘检测{suffix}"
            update_progress(i + 1, num_frames, prefix=prefix)

        # 保存当前块最后一帧，供下一块差分使用
        if enable_diff:
            prev_frame = blurred_arr[end - 1].astype(np.float32)

        gc.collect()

    prefix = f"边缘检测+池化{suffix}" if pool_size > 1 else f"边缘检测{suffix}"
    finish_progress(prefix=prefix)
    if pool_size > 1:
        print(f"流式处理完成{suffix}: {height}x{width} → {target_height}x{target_width}")
    else:
        print(f"边缘检测完成{suffix}: 保持原尺寸 {height}x{width}")

    # ========== 保存各处理结果为独立灰度视频 ==========
    gray_array_to_mp4(blurred_arr, f"{output_dir}/output{suffix}.mp4")
    gray_array_to_mp4(result_horizontal, f"{output_dir}/output_edge_horizontal{suffix}.mp4")
    gray_array_to_mp4(result_vertical, f"{output_dir}/output_edge_vertical{suffix}.mp4")
    if enable_diff:
        gray_array_to_mp4(result_diff, f"{output_dir}/output_diff{suffix}.mp4")

    # ========== RGB 三通道融合 ==========
    # R 通道 ← 水平方向边缘检测（检测垂直边缘）
    # G 通道 ← 垂直方向边缘检测（检测水平边缘）
    # B 通道 ← 相邻帧间差分（反映运动强度，禁用时置零）
    if enable_diff:
        three_gray_array_to_RGB_mp4(
            result_horizontal, result_vertical, result_diff,
            f"{output_dir}/output_RGB{suffix}.mp4"
        )
    else:
        # 禁用差分时，B 通道用零数组
        result_diff = np.zeros_like(result_horizontal)
        three_gray_array_to_RGB_mp4(
            result_horizontal, result_vertical, result_diff,
            f"{output_dir}/output_RGB{suffix}.mp4"
        )

    # 释放内存
    del result_horizontal, result_vertical
    if enable_diff:
        del result_diff
    gc.collect()


def modulate_by_diff(edge_result: np.ndarray, diff_result: np.ndarray) -> np.ndarray:
    """
    使用帧间差分结果作为亮度调制信号，对边缘检测结果进行亮度调制。

    原理：
      帧间差分值反映了运动强度（绝对值越大 = 运动越剧烈）。
      将差分值归一化到 [0, 1] 范围后，作为乘性因子与边缘强度相乘，
      使得运动区域的边缘被保留并增强，静止区域的边缘被抑制（变暗）。

    参数:
        edge_result: 边缘检测结果数组，形状 (frames, H, W)，float64
        diff_result: 帧间差分结果数组，形状 (frames, H, W)，float64

    返回:
        亮度调制后的边缘数组，形状与输入相同
    """
    diff_abs = np.abs(diff_result)
    max_vals = diff_abs.max(axis=(1, 2), keepdims=True)
    max_vals = np.where(max_vals == 0, 1.0, max_vals)
    diff_norm = diff_abs / max_vals
    return edge_result * diff_norm


if __name__ == "__main__":
    # 直接运行时添加项目根目录到 sys.path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.video_converter import mp4_to_grayscale_array

    # 简单测试：读取视频并测试流式处理
    videos = [f for f in os.listdir("data") if f.endswith(".mp4")]
    if videos:
        test_video = os.path.join("data", videos[0])
        print(f"测试视频: {test_video}")
        arr = mp4_to_grayscale_array(test_video)
        print(f"数组形状: {arr.shape}")
        process_motion_blurred_array(arr, "output", suffix="_test", pool_size=4, enable_diff=True)
    else:
        print("data/ 目录下没有视频文件，跳过测试")
