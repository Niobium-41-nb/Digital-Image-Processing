"""
视频时间卷积模块。

提供创建时间运动模糊卷积核和执行三维数组卷积的功能。
使用循环方式进行卷积计算。
"""

import argparse
import os

import cv2
import numpy as np

def create_temporal_motion_blur(time_frames=3, height=1, width=1):
    """
    仅在时间维度产生模糊（帧间混合）
    
    参数:
        time_frames: 模糊的时间跨度（帧数）
        height, width: 空间维度大小（设为1表示无空间模糊）
    """
    kernel = np.zeros((time_frames, height, width))
    
    # 在时间轴上均匀分布权重
    for t in range(time_frames):
        kernel[t, :, :] = 1.0
    
    # 归一化
    kernel = kernel / kernel.sum()
    return kernel

def f(array: np.ndarray, Convolution: np.ndarray) -> np.ndarray:
    """对三维数组执行卷积。

    输入:
        array: 形状为 (frames, height, width) 的三维数组
        Convolution: 形状为 (k_frames, k_height, k_width) 的卷积核
    返回:
        经过卷积后的数组，输出形状与输入相同。
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("array must be a numpy ndarray")
    if not isinstance(Convolution, np.ndarray):
        raise TypeError("Convolution must be a numpy ndarray")
    if array.ndim != 3:
        raise ValueError("array must have 3 dimensions")
    if Convolution.ndim != 3:
        raise ValueError("Convolution kernel must have 3 dimensions")

    kernel = Convolution.astype(np.float64)
    kf, kh, kw = kernel.shape
    pad_f = kf // 2
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(array, ((pad_f, pad_f), (pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
    result = np.zeros_like(array, dtype=np.float64)

    for z in range(array.shape[0]):
        for y in range(array.shape[1]):
            for x in range(array.shape[2]):
                window = padded[z : z + kf, y : y + kh, x : x + kw]
                result[z, y, x] = np.sum(window * kernel)

    if np.issubdtype(array.dtype, np.integer):
        result = np.clip(result, np.iinfo(array.dtype).min, np.iinfo(array.dtype).max)
        return result.astype(array.dtype)
    return result


def main():
    print(create_temporal_motion_blur(5,5,5))

if __name__ == "__main__":
    main()