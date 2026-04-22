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

    # 确保分辨率是对齐的
    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    # --- 1. 生成背景内部纹理 (小网格状态) ---
    bg_cols = width // square_size
    bg_rows = height // square_size
    # 背景也做成可滚动的纹理，不再只是一张静态图
    bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)

    # --- 2. 生成前景内部纹理 (小网格状态) ---
    rect_w = (400 // square_size) * square_size
    rect_h = (400 // square_size) * square_size
    fg_cols = rect_w // square_size
    fg_rows = rect_h // square_size

    fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)

    # --- 3. 固定的前景区域位置 ---
    # 前景区域本身位置固定不动，只有内部纹理滚动
    rect_x = ((bg_cols - fg_cols) // 2) * square_size
    rect_y = ((bg_rows - fg_rows) // 2) * square_size

    print(f"开始渲染视频: {filename}")
    print(f"方块边长: {square_size}像素, 背景向右移动，前景方块内部向下移动")

    for i in range(num_frames):
        # 1. 将当前背景小网格放大成实际像素块
        bg_frame = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)
        frame_gray = bg_frame.copy()

        # 2. 将当前前景小网格放大成实际像素块
        fg_block = np.repeat(np.repeat(fg_small, square_size, axis=0), square_size, axis=1)

        # 3. 将前景块覆盖到背景的固定区域中
        frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block

        # 4. 写入视频
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

        # --- 核心改动：背景与前景沿不同方向滚动 ---
        # 背景向右滚动 (shift=1, axis=1 表示沿水平方向移动)
        bg_small = np.roll(bg_small, shift=1, axis=1)
        
        # 前景向下滚动 (shift=1, axis=0 表示沿垂直方向移动)
        fg_small = np.roll(fg_small, shift=1, axis=0)

        if i % 30 == 0:
            print(f"已生成帧: {i}/{num_frames}")

    out.release()
    print(f"视频生成完成: {filename}")


if __name__ == "__main__":
    generate_dual_scrolling_texture_video('dual_scroll_background_right_foreground_down.mp4', square_size=4)