"""
卷积操作模块。

提供创建时间运动模糊卷积核和执行三维数组卷积的功能。
使用scipy.ndimage.convolve进行高效的三维卷积计算。
"""

import argparse
import os

import cv2
import numpy as np
from scipy.ndimage import convolve

def create_temporal_motion_blur(time_frames=3, height=1, width=1):
    """
    创建时间维度的运动模糊卷积核。
    
    参数:
        time_frames: 模糊的时间跨度（帧数）
        height, width: 空间维度大小（设为1表示无空间模糊）
    返回:
        卷积核数组，形状为 (time_frames, height, width)
    """
    kernel = np.zeros((time_frames, height, width))
    
    # 在时间轴上均匀分布权重
    for t in range(time_frames):
        kernel[t, :, :] = 1.0
    
    # 归一化
    kernel = kernel / kernel.sum()
    return kernel

def f(array: np.ndarray, convolution: np.ndarray) -> np.ndarray:
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

    # 使用scipy.ndimage.convolve进行高效的三维卷积
    result = convolve(array.astype(np.float64), convolution, mode='constant', cval=0.0)
    return result


def main():
    from mp4_to_gray_array import mp4_to_grayscale_array
    from gray_array_to_mp4 import gray_array_to_mp4

    arr = mp4_to_grayscale_array("kinetic_boundary_waterfall.mp4")
    convolution = create_temporal_motion_blur(3, 1, 1)  # 示例卷积核

    result = f(array=arr, convolution=convolution)
    print("卷积完成，结果形状:", result.shape)

if __name__ == "__main__":
    main()