"""
生成完美伪装视频模块。

创建具有随机黑白方块背景和移动前景块的视频，
用于测试或演示伪装效果。前景块以方块大小为步长移动。
"""

import cv2
import numpy as np


def generate_perfect_camouflage_video(filename, width=1280, height=720, square_size=4, fps=30, duration=15, output_frames_folder=False):
    """
    生成完美伪装效果视频。

    该函数创建一个具有随机黑白方块背景的视频，前景块在画面中水平往返移动。
    由于前景块也使用与背景相同的随机纹理模式，当前景移动时会产生视觉上的"消失"效果，
    即前景块与背景融为一体，难以分辨。此效果可用于演示伪装或视觉欺骗原理。

    参数:
        filename: 输出视频文件路径
        width: 视频宽度（默认1280）
        height: 视频高度（默认720）
        square_size: 方块大小，影响纹理颗粒度（默认4）
        fps: 帧率（默认30）
        duration: 视频时长，单位秒（默认15）
        output_frames_folder: 是否将所有帧输出到与视频文件同级文件夹中（默认False）

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
    # 将背景小图像放大为实际尺寸
    bg_img = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)

    # 计算前景方块区域的大小
    rect_w = (400 // square_size) * square_size
    rect_h = (400 // square_size) * square_size

    fg_cols = rect_w // square_size
    fg_rows = rect_h // square_size
    # 生成随机黑白前景小图像（与背景相同的随机模式）
    fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)
    # 将前景小图像放大为实际尺寸
    fg_block = np.repeat(np.repeat(fg_small, square_size, axis=0), square_size, axis=1)

    # 计算前景方块在画面中的初始位置（居中）
    rect_x = ((bg_cols - fg_cols) // 2) * square_size
    rect_y = ((bg_rows - fg_rows) // 2) * square_size

    # 设置前景移动速度（每帧移动一个方块宽度）
    speed_x = square_size

    print(f"开始渲染视频: {filename}")
    print(f"方块边长: {square_size}像素, 步长: {speed_x}像素/帧")

    # 逐帧生成视频
    for i in range(num_frames):
        # 复制背景图像
        frame_gray = bg_img.copy()

        # 将前景方块放置到当前位置
        frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block

        # 将灰度图转换为BGR格式并写入视频
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

        # 前景方块水平移动
        rect_x += speed_x

        # 边界检测：当前景块碰到边界时反弹
        if rect_x <= 0 or rect_x + rect_w >= width:
            speed_x *= -1  # 反转移动方向
            if rect_x < 0:
                rect_x = 0
            if rect_x + rect_w > width:
                rect_x = width - rect_w

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
        bg_img = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)
        fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)
        fg_block = np.repeat(np.repeat(fg_small, square_size, axis=0), square_size, axis=1)
        
        rect_x = ((bg_cols - fg_cols) // 2) * square_size
        rect_y = ((bg_rows - fg_rows) // 2) * square_size
        speed_x = square_size
        
        for i in range(num_frames):
            frame_gray = bg_img.copy()
            frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block
            
            # 保存帧
            frame_path = os.path.join(frames_dir, f'frame_{i:05d}.png')
            cv2.imwrite(frame_path, frame_gray)
            
            # 移动
            rect_x += speed_x
            if rect_x <= 0 or rect_x + rect_w >= width:
                speed_x *= -1
                if rect_x < 0:
                    rect_x = 0
                if rect_x + rect_w > width:
                    rect_x = width - rect_w
        
        print(f"帧保存完成: {frames_dir}")


if __name__ == "__main__":
    import os
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    os.makedirs(DATA_DIR, exist_ok=True)
    generate_perfect_camouflage_video(
        os.path.join(DATA_DIR, 'perfect_camouflage.mp4'),
        square_size=4)