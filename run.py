"""
主程序文件

对多种运动模式的视频进行时间-空间联合卷积处理，
包括基础双滚动纹理和新增的复杂运动模式（斜向、圆周、螺旋、缩放脉冲、8字形）。
"""

import os
import numpy as np
from src.video_converter import mp4_to_grayscale_array, gray_array_to_mp4, two_gray_array_to_GB_mp4, three_gray_array_to_RGB_mp4
from src.temporal_convolver import create_temporal_motion_blur, apply_temporal_convolution, apply_frame_convolution, apply_peel_max, apply_diff

# ============================================================
# 第一步：生成所有复杂运动模式的测试视频
# ============================================================
from generators.complex_motion import (
    generate_diagonal_motion_video,
    generate_circular_motion_video,
    generate_spiral_motion_video,
    generate_pulsing_zoom_video,
    generate_figure_eight_motion_video,
)

os.makedirs("data", exist_ok=True)

print("=" * 70)
print("开始生成复杂运动模式测试视频...")
print("=" * 70)

# 生成所有复杂运动视频（如果不存在）
test_videos = {
    "diagonal": "data/diagonal_motion.mp4",
    "circular": "data/circular_motion.mp4",
    "spiral": "data/spiral_motion.mp4",
    "pulsing": "data/pulsing_zoom.mp4",
    "figure_eight": "data/figure_eight_motion.mp4",
}

for name, path in test_videos.items():
    if not os.path.exists(path):
        print(f"\n生成 {name} 运动视频...")
        if name == "diagonal":
            generate_diagonal_motion_video(path, square_size=4)
        elif name == "circular":
            generate_circular_motion_video(path, square_size=4)
        elif name == "spiral":
            generate_spiral_motion_video(path, square_size=4)
        elif name == "pulsing":
            generate_pulsing_zoom_video(path, square_size=4)
        elif name == "figure_eight":
            generate_figure_eight_motion_video(path, square_size=4)
    else:
        print(f"{path} 已存在，跳过生成")

# ============================================================
# 第二步：对所有视频进行时间-空间联合卷积处理
# ============================================================

# 所有待处理的视频列表（包括原有的双滚动视频）
video_sources = {
    "dual_scroll": "data/dual_scroll_background_right_foreground_down.mp4",
    "diagonal": "data/diagonal_motion.mp4",
    "circular": "data/circular_motion.mp4",
    "spiral": "data/spiral_motion.mp4",
    "pulsing": "data/pulsing_zoom.mp4",
    "figure_eight": "data/figure_eight_motion.mp4",
}

# 卷积核定义
convolution_vertical = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

convolution_horizontal = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])

for video_name, video_path in video_sources.items():
    if not os.path.exists(video_path):
        print(f"\n[跳过] {video_path} 不存在")
        continue

    print(f"\n{'=' * 70}")
    print(f"处理视频: {video_name} ({video_path})")
    print(f"{'=' * 70}")

    # 读取视频
    arr = mp4_to_grayscale_array(video_path)

    for convolution_size in range(1, 7):
        output_dir = f"output/{video_name}/{convolution_size}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n--- 卷积核大小: {convolution_size} ---")

        # 时间维度运动模糊卷积核
        convolution = create_temporal_motion_blur(convolution_size, convolution_size, convolution_size)

        # 时间卷积（运动模糊）
        processed_arr = apply_temporal_convolution(arr, convolution)

        # 垂直与水平边缘检测
        result_vertical = apply_frame_convolution(processed_arr, convolution_vertical)
        result_horizontal = apply_frame_convolution(processed_arr, convolution_horizontal)

        # 帧间差分
        result_diff = apply_diff(arr)

        # 保存结果
        gray_array_to_mp4(processed_arr, f"{output_dir}/output.mp4")
        gray_array_to_mp4(result_vertical, f"{output_dir}/output_vertical.mp4")
        gray_array_to_mp4(result_horizontal, f"{output_dir}/output_horizontal.mp4")
        gray_array_to_mp4(result_diff, f"{output_dir}/output_diff.mp4")

        # RGB多通道融合（R=差分, G=垂直边缘, B=水平边缘）
        three_gray_array_to_RGB_mp4(
            result_diff, result_vertical, result_horizontal,
            f"{output_dir}/output_RGB.mp4"
        )

        # GB双通道融合（G=垂直边缘, B=水平边缘）
        two_gray_array_to_GB_mp4(
            result_vertical, result_horizontal,
            f"{output_dir}/output_GB.mp4",
            output_frames_folder=(convolution_size == 3)  # 仅在中间参数保存帧
        )

print(f"\n{'=' * 70}")
print("所有视频处理完成！")
print(f"{'=' * 70}")
