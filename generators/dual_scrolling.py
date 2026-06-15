"""
生成双滚动纹理视频模块。

创建具有移动背景和滚动前景纹理的视频。
背景纹理与前景方块沿不同方向循环滚动，用于视觉效果演示。
"""

import os
import cv2
import numpy as np


def generate_dual_scrolling_texture_video(filename, width=1280, height=720, square_size=4, fps=30, duration=15, output_frames_folder=False, bg_dx=1, bg_dy=0, fg_dx=0, fg_dy=1):
    """
    生成双滚动纹理视频。

    该函数创建一个视频，其中背景纹理和前景方块分别以指定速度分量循环滚动。
    dx 为水平速度分量（正=右，负=左），dy 为垂直速度分量（正=下，负=上）。

    参数:
        filename: 输出视频文件路径
        width: 视频宽度（默认1280）
        height: 视频高度（默认720）
        square_size: 方块大小，影响纹理颗粒度（默认4）
        fps: 帧率（默认30）
        duration: 视频时长，单位秒（默认15）
        output_frames_folder: 是否将所有帧输出到与视频文件同级文件夹中（默认False）
        bg_dx: 背景水平速度分量，正=右，负=左，0=不移动（默认1）
        bg_dy: 背景垂直速度分量，正=下，负=上，0=不移动（默认0）
        fg_dx: 前景水平速度分量，正=右，负=左，0=不移动（默认0）
        fg_dy: 前景垂直速度分量，正=下，负=上，0=不移动（默认1）

    返回:
        无返回值，生成视频文件保存到指定路径
    """
    # 创建视频写入器，使用mp4v编码格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    # 计算总帧数
    num_frames = fps * duration

    # 将宽高调整为方块大小的整数倍，确保像素完整性
    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    # 计算背景小图像的行列数（以方块为单位）
    bg_cols = width // square_size
    bg_rows = height // square_size
    # 生成随机黑白背景小图像（0或255）
    bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)

    # 计算前景方块区域的大小
    rect_w = (400 // square_size) * square_size
    rect_h = (400 // square_size) * square_size
    fg_cols = rect_w // square_size
    fg_rows = rect_h // square_size
    # 生成随机黑白前景小图像
    fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)

    # 计算前景方块在画面中的位置（居中）
    rect_x = ((bg_cols - fg_cols) // 2) * square_size
    rect_y = ((bg_rows - fg_rows) // 2) * square_size

    print(f"开始渲染视频: {filename}")
    print(f"方块边长: {square_size}像素, 背景速度(dx={bg_dx}, dy={bg_dy})，前景速度(dx={fg_dx}, dy={fg_dy})")

    # 逐帧生成视频
    for i in range(num_frames):
        # 将背景小图像放大为实际尺寸
        bg_frame = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)
        frame_gray = bg_frame.copy()

        # 将前景小图像放大为实际尺寸
        fg_block = np.repeat(np.repeat(fg_small, square_size, axis=0), square_size, axis=1)

        # 将前景方块放置到画面中央
        frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block

        # 将灰度图转换为BGR格式并写入视频
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

        # 背景按速度分量循环滚动
        bg_small = np.roll(bg_small, shift=(bg_dy, bg_dx), axis=(0, 1))

        # 前景纹理按速度分量循环滚动
        fg_small = np.roll(fg_small, shift=(fg_dy, fg_dx), axis=(0, 1))

        # 每30帧输出一次进度信息
        if i % 30 == 0:
            print(f"已生成帧: {i}/{num_frames}")

    # 释放视频写入器
    out.release()
    print(f"视频生成完成: {filename}")

    # 如果需要输出帧文件夹
    if output_frames_folder:
        frames_dir = filename.rsplit('.', 1)[0] + '_frames'
        os.makedirs(frames_dir, exist_ok=True)
        print(f"开始保存帧到: {frames_dir}")
        
        # 重新生成帧并保存
        bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)
        fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)
        
        for i in range(num_frames):
            bg_frame = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)
            frame_gray = bg_frame.copy()
            fg_block = np.repeat(np.repeat(fg_small, square_size, axis=0), square_size, axis=1)
            frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block
            
            # 保存帧
            frame_path = os.path.join(frames_dir, f'frame_{i:05d}.png')
            cv2.imwrite(frame_path, frame_gray)
            
            # 按速度分量滚动
            bg_small = np.roll(bg_small, shift=(bg_dy, bg_dx), axis=(0, 1))
            fg_small = np.roll(fg_small, shift=(fg_dy, fg_dx), axis=(0, 1))
        
        print(f"帧保存完成: {frames_dir}")


if __name__ == "__main__":
    import os
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    os.makedirs(DATA_DIR, exist_ok=True)

    # 设置背景和前景的移动速度分量 (dx, dy)
    # dx: 水平速度，正=右，负=左，0=不动
    # dy: 垂直速度，正=下，负=上，0=不动
    bg_dx, bg_dy = 1, 2   # 背景向右速度1
    fg_dx, fg_dy = 1, -1   # 前景向下速度1

    # 文件名体现速度设置
    filename = f'dual_scroll_background_dx{bg_dx}_dy{bg_dy}_foreground_dx{fg_dx}_dy{fg_dy}.mp4'
    generate_dual_scrolling_texture_video(
        os.path.join(DATA_DIR, filename),
        square_size=4, bg_dx=bg_dx, bg_dy=bg_dy, fg_dx=fg_dx, fg_dy=fg_dy)