"""
视频生成器模块

提供各种视频生成功能，包括基础纹理视频和复杂运动模式视频。
"""

from .perfect_camouflage import generate_perfect_camouflage_video
from .scrolling_texture import generate_scrolling_texture_video
from .dual_scrolling import generate_dual_scrolling_texture_video
from .complex_motion import (
    generate_diagonal_motion_video,
    generate_circular_motion_video,
    generate_spiral_motion_video,
    generate_pulsing_zoom_video,
    generate_figure_eight_motion_video,
)

__all__ = [
    'generate_perfect_camouflage_video',
    'generate_scrolling_texture_video',
    'generate_dual_scrolling_texture_video',
    'generate_diagonal_motion_video',
    'generate_circular_motion_video',
    'generate_spiral_motion_video',
    'generate_pulsing_zoom_video',
    'generate_figure_eight_motion_video',
]