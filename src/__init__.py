"""
数字图像处理核心模块

提供视频处理功能。
"""

from .video_converter import mp4_to_grayscale_array, gray_array_to_mp4

__all__ = [
    'mp4_to_grayscale_array',
    'gray_array_to_mp4',
]