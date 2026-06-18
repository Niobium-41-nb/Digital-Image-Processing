"""
时间卷积模块。

提供创建时间运动模糊卷积核和执行三维数组卷积的功能。
使用分离卷积优化：将三维卷积核拆分为一维时间轴卷积，
大幅减少计算量（从 O(T*H*W) 降至 O(T+H+W)）。
"""

import argparse
import gc
import os

import numpy as np
from scipy.ndimage import convolve, maximum_filter

from src.progress_bar import update_progress, finish_progress


def create_temporal_motion_blur(time_frames=3, height=1, width=1):
    """
    创建时间维度的运动模糊卷积核。

    卷积核形状为 (time_frames, height, width)，其中所有元素值相同（均为 1.0）。
    即 (n, n, n) 中全为相同的数，确保每帧权重相等，实现均匀的时间模糊。
    卷积核数值和 = n^3 ≥ 1，确保卷积后信号强度不衰减。

    参数:
        time_frames: 模糊的时间跨度（帧数）
        height, width: 空间维度大小（设为1表示无空间模糊）
    返回:
        卷积核数组，形状为 (time_frames, height, width)
        所有元素均为 1.0，权重完全均匀
    """
    # 所有元素均为 1.0，实现 (n, n, n) 全相同数值
    kernel = np.ones((time_frames, height, width), dtype=np.float64)

    # 归一化：核的总和为 1，确保卷积后信号强度不变
    kernel = kernel / (time_frames * height * width)

    return kernel


def apply_temporal_convolution(array: np.ndarray, convolution: np.ndarray,
                                verbose: bool = True) -> np.ndarray:
    """对三维数组执行分离式时间轴卷积。

    利用卷积核为 (T, 1, 1) 的可分离特性，将三维卷积拆分为
    一维时间轴卷积，计算量从 O(T*H*W) 降至 O(T+H+W)。

    输入:
        array: 形状为 (frames, height, width) 的三维数组
        convolution: 形状为 (k_frames, k_height, k_width) 的卷积核
        verbose: 是否显示进度条（被 scale_and_convolve 调用时设为 False）
    返回:
        经过卷积后的数组，输出形状与输入相同。
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("array must be a numpy ndarray")
    if not isinstance(convolution, np.ndarray):
        raise TypeError("convolution must be a numpy ndarray")
    if array.ndim != 3:
        raise ValueError("array must have 3 dimensions")
    if convolution.ndim != 3:
        raise ValueError("convolution kernel must have 3 dimensions")

    frames = array.shape[0]
    height, width = array.shape[1], array.shape[2]

    # 提取一维时间轴卷积核（kernel 形状为 (T, 1, 1)）
    # 利用分离卷积优化：仅沿时间轴做 1D 卷积
    kernel_1d = convolution[:, 0, 0].astype(np.float32)
    k_frames = len(kernel_1d)

    # 计算单帧内存开销
    single_frame_bytes = height * width * 4  # float32 = 4 bytes
    
    # 目标：额外空间开销 < 5GB = 5 * 1024^3 bytes
    max_memory_bytes = 5 * 1024 ** 3  # 5GB
    frames_per_chunk = max(1, max_memory_bytes // (single_frame_bytes * 2))
    chunk_size = min(frames_per_chunk, 60)  # 分离卷积内存更省，可增大块大小

    # 使用分块 + 1D 卷积处理
    result = np.zeros((frames, height, width), dtype=np.float32)
    overlap = k_frames - 1

    for start in range(0, frames, chunk_size):
        end = min(start + chunk_size, frames)
        # 扩展边界以处理边缘
        buf_start = max(0, start - overlap)
        buf_end = min(frames, end + overlap)

        # 取当前块数据
        chunk = array[buf_start:buf_end].astype(np.float32)

        # 沿时间轴（axis=0）做一维卷积
        # scipy.ndimage.convolve1d 自动沿指定轴做 1D 卷积
        from scipy.ndimage import convolve1d
        chunk_result = convolve1d(chunk, kernel_1d, axis=0, mode='constant', cval=0.0)

        # 只保留有效部分
        result_start = start - buf_start
        result_end = result_start + (end - start)
        result[start:end] = chunk_result[result_start:result_end]

        del chunk, chunk_result

        if verbose:
            # 块内每帧更新进度条
            for i in range(start, end):
                update_progress(i + 1, frames, prefix="时间卷积")

    if verbose:
        finish_progress(prefix="时间卷积")
    return result.astype(array.dtype)

def apply_frame_convolution(array: np.ndarray, convolution: np.ndarray) -> np.ndarray:
    """对每一帧图像执行卷积。并返回视频结合后的结果。
    输入:
        array: 形状为 (frames, height, width) 的三维数组
        convolution: 形状为 (k_height, k_width) 的卷积核
    返回:
        经过卷积后的数组，输出形状与输入相同。
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("array must be a numpy ndarray")
    if not isinstance(convolution, np.ndarray):
        raise TypeError("convolution must be a numpy ndarray")
    if array.ndim != 3:
        raise ValueError("array must have 3 dimensions")
    if convolution.ndim != 2:
        raise ValueError("convolution kernel must have 2 dimensions")

    # 对每一帧应用二维卷积
    frames, height, width = array.shape
    result = np.zeros_like(array, dtype=np.float32)

    print(f"开始逐帧卷积处理，共 {frames} 帧")
    for i in range(frames):
        result[i] = convolve(array[i].astype(np.float64), convolution, mode='constant', cval=0.0)

        update_progress(i + 1, frames, prefix="逐帧卷积")

    finish_progress(prefix="逐帧卷积")
    return result

def apply_diff(array:np.array) -> np.ndarray:
    """对视频做差分处理，将后一帧的灰度图减去前一帧的灰度图

    输入:
        array: 形状为 (frames, height, width) 的三维数组
    返回:
        差分数组，输出形状与输入相同。
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("array must be a numpy ndarray")
    if array.ndim != 3:
        raise ValueError("array must have 3 dimensions")
    
    frames, height, width = array.shape
    result = np.zeros_like(array, dtype=np.float64)

    print(f"开始差分处理，共 {frames} 帧")
    for i in range(1, frames):
        result[i] = array[i] - array[i - 1]
        update_progress(i, frames, prefix="帧间差分")

    result[0] = 0
    finish_progress(prefix="帧间差分")
    return result.astype(array.dtype)

def scale_and_convolve(arr, scale_factor, convolution, max_memory=5 * 1024 ** 3, max_chunk_frames=30):
    """
    对视频进行分块放大并立即执行时间维度卷积。
    
    将每帧图像放大 scale_factor 倍后，立即进行时间卷积，
    避免同时存储原始和放大后的完整视频。

    参数:
        arr: 原始视频数组，形状 (frames, height, width)
        scale_factor: 图像放大倍数
        convolution: 时间维度卷积核
        max_memory: 最大允许内存（字节）
        max_chunk_frames: 每块最大帧数

    返回:
        放大并卷积后的数组
    """
    original_height, original_width = arr.shape[1], arr.shape[2]
    scaled_height = original_height * scale_factor
    scaled_width = original_width * scale_factor

    print(f"\n开始图像放大，倍数: {scale_factor}x")
    print(f"原始尺寸: {original_height}x{original_width} → 放大后: {scaled_height}x{scaled_width}")

    # 计算合适的分块大小
    bytes_per_frame = scaled_height * scaled_width * 4 * 2  # 输入+输出
    chunk_size = max(1, max_memory // bytes_per_frame)
    chunk_size = min(chunk_size, max_chunk_frames)

    print(f"放大分块大小: {chunk_size} 帧/块")

    processed_frames = []
    for start in range(0, arr.shape[0], chunk_size):
        end = min(start + chunk_size, arr.shape[0])

        # 放大当前块
        chunk_scaled = np.zeros((end - start, scaled_height, scaled_width), dtype=arr.dtype)
        for i in range(start, end):
            chunk_scaled[i - start] = np.repeat(np.repeat(arr[i], scale_factor, axis=0), scale_factor, axis=1)
            # 放大每帧后立即更新进度条
            update_progress(i + 1, arr.shape[0], prefix="放大+卷积")

        # 时间维度卷积（静默模式，不显示内部进度条）
        chunk_result = apply_temporal_convolution(chunk_scaled, convolution, verbose=False)
        processed_frames.append(chunk_result)

        del chunk_scaled

    result = np.concatenate(processed_frames, axis=0)
    del processed_frames
    gc.collect()

    finish_progress(prefix="放大+卷积")
    print(f"图像放大完成: {original_height}x{original_width} → {scaled_height}x{scaled_width}")
    return result


def apply_peel_max(array: np.ndarray, size: int = 5) -> np.ndarray:
    """对每一帧图像执行最大池化。

    输入:
        array: 形状为 (frames, height, width) 的三维数组
        size: 池化核大小，即池化窗口为 (size, size)
    返回:
        经过最大池化后的数组，输出形状与输入相同。
    """
    if array.ndim != 3:
        raise ValueError("Array must be 3D with shape (frames, height, width)")
    
    frames, height, width = array.shape
    result = np.zeros_like(array, dtype=np.float64)

    print(f"开始逐帧最大池化处理，共 {frames} 帧")
    for f in range(frames):
        result[f] = maximum_filter(array[f].astype(np.float64), size=size, mode='constant', cval=0.0)

        update_progress(f + 1, frames, prefix="最大池化")

    finish_progress(prefix="最大池化")
    return result.astype(array.dtype)

def main():
    from .video_converter import mp4_to_grayscale_array, gray_array_to_mp4

    arr = mp4_to_grayscale_array("kinetic_boundary_waterfall.mp4")
    convolution = create_temporal_motion_blur(3, 1, 1)

    result = apply_temporal_convolution(array=arr, convolution=convolution)
    print("卷积完成，结果形状:", result.shape)


if __name__ == "__main__":
    main()