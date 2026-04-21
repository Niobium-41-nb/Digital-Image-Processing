"""
生成完美伪装视频模块。

创建具有随机黑白方块背景和移动前景块的视频，
用于测试或演示伪装效果。前景块以方块大小为步长移动。
"""

import cv2
import numpy as np


def generate_perfect_camouflage_video(filename, width=1280, height=720, square_size=4, fps=30, duration=15):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    num_frames = fps * duration

    # 确保视频的分辨率是方块大小的整数倍，避免边缘裁剪问题
    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    # --- 1. 生成固定的静态背景 ---
    bg_cols = width // square_size
    bg_rows = height // square_size
    bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)
    bg_img = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)

    # --- 2. 生成移动实体块 ---
    # 尺寸设定，同样确保是整数倍
    rect_w = (400 // square_size) * square_size
    rect_h = (400 // square_size) * square_size

    fg_cols = rect_w // square_size
    fg_rows = rect_h // square_size
    fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)
    fg_block = np.repeat(np.repeat(fg_small, square_size, axis=0), square_size, axis=1)

    # --- 3. 初始位置和速度 ---
    # 初始位置的 X 和 Y 必须也是方块的整数倍，保证从第一帧就完美对齐
    rect_x = ((bg_cols - fg_cols) // 2) * square_size
    rect_y = ((bg_rows - fg_rows) // 2) * square_size

    # 核心改动：步长严格等于正方形的边长
    speed_x = square_size

    print(f"开始渲染视频: {filename}")
    print(f"方块边长: {square_size}像素, 步长: {speed_x}像素/帧")

    for i in range(num_frames):
        # 每帧使用干净的背景
        frame_gray = bg_img.copy()

        # 将实体块贴上去
        frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block

        # 转换为 BGR 并写入
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

        # 更新位置：按整格移动
        rect_x += speed_x

        # 触壁反弹，同时确保修正后的坐标依然在网格上
        if rect_x <= 0 or rect_x + rect_w >= width:
            speed_x *= -1  # 翻转方向
            # 将溢出的位置拉回网格边界
            if rect_x < 0:
                rect_x = 0
            if rect_x + rect_w > width:
                rect_x = width - rect_w

        if i % 30 == 0:
            print(f"已生成帧: {i}/{num_frames}")

    out.release()
    print(f"视频生成完成: {filename}")


if __name__ == "__main__":
    generate_perfect_camouflage_video('perfect_camouflage.mp4', square_size=4)