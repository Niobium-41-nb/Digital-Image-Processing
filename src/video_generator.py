"""
============================================================
视频生成模块 —— 生成复杂运动模式测试视频
============================================================

封装了所有测试视频的生成逻辑，提供统一的生成入口。

依赖：
  - generators/complex_motion.py
============================================================
"""

import os

from generators.complex_motion import (
    generate_diagonal_motion_video,
    generate_circular_motion_video,
    generate_spiral_motion_video,
    generate_figure_eight_motion_video,
)

# 定义待生成的测试视频列表
# 键为运动模式名称，值为输出文件路径
TEST_VIDEOS = {
    "diagonal": "data/diagonal_motion.mp4",
    "circular": "data/circular_motion.mp4",
    "spiral": "data/spiral_motion.mp4",
    "figure_eight": "data/figure_eight_motion.mp4",
}

# 所有待处理的视频源列表
VIDEO_SOURCES = {
    "dual_scroll": "data/dual_scroll_background_right_foreground_down.mp4",
    "diagonal": "data/diagonal_motion.mp4",
    "circular": "data/circular_motion.mp4",
    "spiral": "data/spiral_motion.mp4",
    "figure_eight": "data/figure_eight_motion.mp4",
}


def generate_all_test_videos():
    """生成所有不存在的测试视频"""
    os.makedirs("data", exist_ok=True)

    print("=" * 70)
    print("开始生成复杂运动模式测试视频...")
    print("=" * 70)

    for name, path in TEST_VIDEOS.items():
        if not os.path.exists(path):
            print(f"\n生成 {name} 运动视频...")
            if name == "diagonal":
                generate_diagonal_motion_video(path, square_size=4)
            elif name == "circular":
                generate_circular_motion_video(path, square_size=4)
            elif name == "spiral":
                generate_spiral_motion_video(path, square_size=4)
            elif name == "figure_eight":
                generate_figure_eight_motion_video(path, square_size=4)
        else:
            print(f"{path} 已存在，跳过生成")


def show_video_list():
    """显示可选视频列表及其存在状态"""
    print("\n" + "=" * 50)
    print("可选视频列表：")
    print("=" * 50)
    for idx, (name, path) in enumerate(VIDEO_SOURCES.items(), 1):
        exists = "[Y]" if os.path.exists(path) else "[N]"
        print(f"  {idx}. {name:15} {exists}")
    print("=" * 50)


def get_video_path(choice):
    """根据用户选择获取视频名称和路径"""
    selected_name = list(VIDEO_SOURCES.keys())[choice - 1]
    selected_path = VIDEO_SOURCES[selected_name]
    return selected_name, selected_path
