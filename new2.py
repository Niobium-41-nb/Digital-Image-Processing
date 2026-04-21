"""
生成滚动纹理视频模块。

创建具有固定背景和滚动前景纹理的视频。
前景纹理在固定区域内循环滚动，用于视觉效果演示。
"""

import cv2
import numpy as np


def generate_scrolling_texture_video(filename, width=1280, height=720, square_size=4, fps=30, duration=15):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    num_frames = fps * duration

    # 确保分辨率是对齐的
    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    # --- 1. 生成固定的静态背景 ---
    bg_cols = width // square_size
    bg_rows = height // square_size
    bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)
    bg_img = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)

    # --- 2. 生成前景内部纹理 (小网格状态) ---
    rect_w = (400 // square_size) * square_size
    rect_h = (400 // square_size) * square_size
    fg_cols = rect_w // square_size
    fg_rows = rect_h // square_size

    # 我们只生成小网格，在小网格层面进行滚动，这样效率最高且绝对严丝合缝
    fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)

    # --- 3. 固定的区域位置 ---
    # 区域本身不动，固定在画面中央
    rect_x = ((bg_cols - fg_cols) // 2) * square_size
    rect_y = ((bg_rows - fg_rows) // 2) * square_size

    print(f"开始渲染视频: {filename}")
    print(f"方块边长: {square_size}像素, 区域固定，内部纹理向下循环滚动")

    for i in range(num_frames):
        # 每一帧从干净背景开始
        frame_gray = bg_img.copy()

        # 将当前的小网格放大成实际像素块
        fg_block = np.repeat(np.repeat(fg_small, square_size, axis=0), square_size, axis=1)

        # 贴到固定的区域中
        frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block

        # 写入视频
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

        # --- 核心改动：向下滚动内部纹理 ---
        # np.roll(..., shift=1, axis=0) 表示沿垂直方向（Y轴）将矩阵向下滚动 1 格
        # 到底部的行会自动回到最顶部，完美实现“边界折返/循环”
        fg_small = np.roll(fg_small, shift=1, axis=0)

        if i % 30 == 0:
            print(f"已生成帧: {i}/{num_frames}")

    out.release()
    print(f"视频生成完成: {filename}")


if __name__ == "__main__":
    generate_scrolling_texture_video('kinetic_boundary_waterfall.mp4', square_size=4)