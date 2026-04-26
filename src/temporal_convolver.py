"""
时间卷积模块。

提供创建时间运动模糊卷积核和执行三维数组卷积的功能。
使用scipy.ndimage.convolve进行高效的三维卷积计算。
"""

import argparse
import os

import numpy as np
from scipy.ndimage import convolve, maximum_filter


def create_temporal_motion_blur(time_frames=3, height=1, width=1):
    """
    创建时间维度的运动模糊卷积核。

    注意：只使用当前帧及过去的帧，未来的帧权重设为0。
    这是一种因果卷积（causal convolution），确保输出只依赖于当前和过去的信息。

    参数:
        time_frames: 模糊的时间跨度（帧数）
        height, width: 空间维度大小（设为1表示无空间模糊）
    返回:
        卷积核数组，形状为 (time_frames, height, width)
        前半部分为有效权重，后半部分（未来帧）为0
    """
    kernel = np.zeros((time_frames, height, width))

    # 只对当前帧及过去的帧设置权重，未来的帧保持为0
    # 例如 time_frames=3 时：[过去, 当前, 未来] → [1, 1, 0]
    # 例如 time_frames=5 时：[t-2, t-1, 当前, t+1, t+2] → [1, 1, 1, 0, 0]
    effective_frames = (time_frames + 1) // 2  # 向上取整，确保至少包含当前帧
    
    for t in range(effective_frames):
        kernel[t, :, :] = 1.0

    # 归一化，使有效权重之和为1
    kernel = kernel / kernel.sum()
    return kernel


def apply_temporal_convolution(array: np.ndarray, convolution: np.ndarray) -> np.ndarray:
    """对三维数组执行卷积。

    输入:
        array: 形状为 (frames, height, width) 的三维数组
        convolution: 形状为 (k_frames, k_height, k_width) 的卷积核
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
    print(f"开始时间维度卷积处理，共 {frames} 帧")

    result = convolve(array.astype(np.float64), convolution, mode='constant', cval=0.0)

    print("时间维度卷积处理完成")
    return result

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
    result = np.zeros_like(array, dtype=np.float64)

    print(f"开始逐帧卷积处理，共 {frames} 帧")
    for i in range(frames):
        result[i] = convolve(array[i].astype(np.float64), convolution, mode='constant', cval=0.0)

        if (i + 1) % 30 == 0:
            print(f"已处理帧: {i + 1}/{frames}")

    print("逐帧卷积处理完成")
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
        if i % 30 == 0:
            print(f"已处理帧: {i}/{frames}")

    result[0] = 0
    print("差分处理完成")
    return result.astype(array.dtype)

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

        if (f + 1) % 30 == 0:
            print(f"已处理帧: {f + 1}/{frames}")

    print("逐帧最大池化处理完成")
    return result.astype(array.dtype)

def main():
    from .video_converter import mp4_to_grayscale_array, gray_array_to_mp4

    arr = mp4_to_grayscale_array("kinetic_boundary_waterfall.mp4")
    convolution = create_temporal_motion_blur(3, 1, 1)

    result = apply_temporal_convolution(array=arr, convolution=convolution)
    print("卷积完成，结果形状:", result.shape)


if __name__ == "__main__":
    main()