"""
生成双滚动纹理视频模块。

创建具有移动背景和滚动前景纹理的视频。
背景纹理与前景方块沿不同方向循环滚动，用于视觉效果演示。
"""

import cv2
import numpy as np


def generate_dual_scrolling_texture_video(filename, width=1280, height=720, square_size=4, fps=30, duration=15):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    num_frames = fps * duration

    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    bg_cols = width // square_size
    bg_rows = height // square_size
    bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)

    rect_w = (400 // square_size) * square_size
    rect_h = (400 // square_size) * square_size
    fg_cols = rect_w // square_size
    fg_rows = rect_h // square_size

    fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)

    rect_x = ((bg_cols - fg_cols) // 2) * square_size
    rect_y = ((bg_rows - fg_rows) // 2) * square_size

    print(f"开始渲染视频: {filename}")
    print(f"方块边长: {square_size}像素, 背景向右移动，前景方块内部向下移动")

    for i in range(num_frames):
        bg_frame = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)
        frame_gray = bg_frame.copy()

        fg_block = np.repeat(np.repeat(fg_small, square_size, axis=0), square_size, axis=1)

        frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block

        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

        bg_small = np.roll(bg_small, shift=1, axis=1)

        fg_small = np.roll(fg_small, shift=1, axis=0)

        if i % 30 == 0:
            print(f"已生成帧: {i}/{num_frames}")

    out.release()
    print(f"视频生成完成: {filename}")


if __name__ == "__main__":
    generate_dual_scrolling_texture_video('dual_scroll_background_right_foreground_down.mp4', square_size=4)