"""
视频生成器模块

提供各种视频生成功能。
"""

from .perfect_camouflage import generate_perfect_camouflage_video
from .scrolling_texture import generate_scrolling_texture_video
from .dual_scrolling import generate_dual_scrolling_texture_video

__all__ = [
    'generate_perfect_camouflage_video',
    'generate_scrolling_texture_video',
    'generate_dual_scrolling_texture_video',
]