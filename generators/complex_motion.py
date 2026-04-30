"""
复杂运动图形视频生成模块。

创建具有复杂图形效果和非垂直水平运动模式的视频。
支持斜向运动、圆周运动、螺旋运动、缩放脉冲运动等多种模式，
用于测试时间-空间联合卷积在复杂运动场景下的表现。
"""

import cv2
import numpy as np
import os
import math


def generate_diagonal_motion_video(filename, width=1280, height=720, square_size=4,
                                    fps=30, duration=15, output_frames_folder=False):
    """
    生成斜向运动纹理视频。

    背景纹理沿45度对角线方向循环滚动，前景方块沿-45度方向循环滚动。
    运动方向既非纯水平也非纯垂直，用于测试斜向运动下的边缘检测响应。

    参数:
        filename: 输出视频文件路径
        width: 视频宽度（默认1280）
        height: 视频高度（默认720）
        square_size: 方块大小，影响纹理颗粒度（默认4）
        fps: 帧率（默认30）
        duration: 视频时长，单位秒（默认15）
        output_frames_folder: 是否将所有帧输出到与视频文件同级文件夹中（默认False）
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    num_frames = fps * duration

    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    bg_cols = width // square_size
    bg_rows = height // square_size
    # 生成随机黑白背景小图像
    bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)

    # 前景方块区域
    rect_w = (400 // square_size) * square_size
    rect_h = (400 // square_size) * square_size
    fg_cols = rect_w // square_size
    fg_rows = rect_h // square_size
    fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)

    rect_x = ((bg_cols - fg_cols) // 2) * square_size
    rect_y = ((bg_rows - fg_rows) // 2) * square_size

    print(f"开始渲染斜向运动视频: {filename}")
    print(f"背景沿45°对角线方向滚动，前景沿-45°对角线方向滚动")

    for i in range(num_frames):
        bg_frame = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)
        frame_gray = bg_frame.copy()

        fg_block = np.repeat(np.repeat(fg_small, square_size, axis=0), square_size, axis=1)
        frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block

        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

        # 背景沿对角线方向滚动（同时向右和向下）
        bg_small = np.roll(bg_small, shift=1, axis=1)  # 向右
        bg_small = np.roll(bg_small, shift=1, axis=0)  # 向下

        # 前景沿反对角线方向滚动（同时向左和向上）
        fg_small = np.roll(fg_small, shift=-1, axis=1)  # 向左
        fg_small = np.roll(fg_small, shift=-1, axis=0)  # 向上

        if i % 30 == 0:
            print(f"已生成帧: {i}/{num_frames}")

    out.release()
    print(f"斜向运动视频生成完成: {filename}")

    if output_frames_folder:
        _save_frames_diagonal(filename, width, height, square_size, num_frames,
                              bg_rows, bg_cols, fg_rows, fg_cols, rect_x, rect_y)


def _save_frames_diagonal(filename, width, height, square_size, num_frames,
                           bg_rows, bg_cols, fg_rows, fg_cols, rect_x, rect_y):
    """辅助函数：保存斜向运动视频的帧"""
    frames_dir = filename.rsplit('.', 1)[0] + '_frames'
    os.makedirs(frames_dir, exist_ok=True)
    print(f"开始保存帧到: {frames_dir}")

    bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)
    fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)

    for i in range(num_frames):
        bg_frame = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)
        frame_gray = bg_frame.copy()
        fg_block = np.repeat(np.repeat(fg_small, square_size, axis=0), square_size, axis=1)
        frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block

        frame_path = os.path.join(frames_dir, f'frame_{i:05d}.png')
        cv2.imwrite(frame_path, frame_gray)

        bg_small = np.roll(bg_small, shift=1, axis=1)
        bg_small = np.roll(bg_small, shift=1, axis=0)
        fg_small = np.roll(fg_small, shift=-1, axis=1)
        fg_small = np.roll(fg_small, shift=-1, axis=0)

    print(f"帧保存完成: {frames_dir}")


def generate_circular_motion_video(filename, width=1280, height=720, square_size=4,
                                    fps=30, duration=15, output_frames_folder=False):
    """
    生成圆周运动纹理视频。

    前景方块沿圆形轨迹运动，背景纹理静止。
    圆周运动同时涉及水平和垂直方向的速度分量变化，
    用于测试曲线运动下的边缘检测和运动感知。

    参数:
        filename: 输出视频文件路径
        width: 视频宽度（默认1280）
        height: 视频高度（默认720）
        square_size: 方块大小，影响纹理颗粒度（默认4）
        fps: 帧率（默认30）
        duration: 视频时长，单位秒（默认15）
        output_frames_folder: 是否将所有帧输出到与视频文件同级文件夹中（默认False）
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    num_frames = fps * duration

    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    bg_cols = width // square_size
    bg_rows = height // square_size
    bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)
    bg_img = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)

    # 前景方块大小
    rect_w = (200 // square_size) * square_size
    rect_h = (200 // square_size) * square_size
    fg_cols = rect_w // square_size
    fg_rows = rect_h // square_size
    fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)
    fg_block = np.repeat(np.repeat(fg_small, square_size, axis=0), square_size, axis=1)

    # 圆周运动参数
    center_x = width // 2
    center_y = height // 2
    orbit_radius = min(width, height) // 4
    # 每帧角度增量（弧度），使前景在duration秒内完成约3圈
    angular_speed = (2 * math.pi * 3) / num_frames

    print(f"开始渲染圆周运动视频: {filename}")
    print(f"前景沿圆形轨迹运动，半径={orbit_radius}像素，约3圈")

    for i in range(num_frames):
        frame_gray = bg_img.copy()

        # 计算当前帧前景方块的位置（圆形轨迹）
        angle = angular_speed * i
        rect_x = int(center_x + orbit_radius * math.cos(angle) - rect_w // 2)
        rect_y = int(center_y + orbit_radius * math.sin(angle) - rect_h // 2)

        # 确保在画面内
        rect_x = max(0, min(rect_x, width - rect_w))
        rect_y = max(0, min(rect_y, height - rect_h))

        frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block

        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

        if i % 30 == 0:
            print(f"已生成帧: {i}/{num_frames}")

    out.release()
    print(f"圆周运动视频生成完成: {filename}")

    if output_frames_folder:
        _save_frames_circular(filename, width, height, square_size, num_frames,
                              bg_rows, bg_cols, fg_rows, fg_cols,
                              center_x, center_y, orbit_radius, angular_speed)


def _save_frames_circular(filename, width, height, square_size, num_frames,
                           bg_rows, bg_cols, fg_rows, fg_cols,
                           center_x, center_y, orbit_radius, angular_speed):
    """辅助函数：保存圆周运动视频的帧"""
    frames_dir = filename.rsplit('.', 1)[0] + '_frames'
    os.makedirs(frames_dir, exist_ok=True)
    print(f"开始保存帧到: {frames_dir}")

    bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)
    bg_img = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)
    fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)
    fg_block = np.repeat(np.repeat(fg_small, square_size, axis=0), square_size, axis=1)

    rect_w = fg_cols * square_size
    rect_h = fg_rows * square_size

    for i in range(num_frames):
        frame_gray = bg_img.copy()
        angle = angular_speed * i
        rect_x = int(center_x + orbit_radius * math.cos(angle) - rect_w // 2)
        rect_y = int(center_y + orbit_radius * math.sin(angle) - rect_h // 2)
        rect_x = max(0, min(rect_x, width - rect_w))
        rect_y = max(0, min(rect_y, height - rect_h))
        frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block

        frame_path = os.path.join(frames_dir, f'frame_{i:05d}.png')
        cv2.imwrite(frame_path, frame_gray)

    print(f"帧保存完成: {frames_dir}")


def generate_spiral_motion_video(filename, width=1280, height=720, square_size=4,
                                  fps=30, duration=15, output_frames_folder=False):
    """
    生成螺旋运动纹理视频。

    前景方块沿螺旋轨迹运动，同时自身纹理持续旋转滚动。
    螺旋运动结合了圆周运动和径向运动，运动方向持续变化，
    是测试时间-空间联合卷积在复杂非匀速运动场景下表现的理想测试数据。

    参数:
        filename: 输出视频文件路径
        width: 视频宽度（默认1280）
        height: 视频高度（默认720）
        square_size: 方块大小，影响纹理颗粒度（默认4）
        fps: 帧率（默认30）
        duration: 视频时长，单位秒（默认15）
        output_frames_folder: 是否将所有帧输出到与视频文件同级文件夹中（默认False）
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    num_frames = fps * duration

    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    bg_cols = width // square_size
    bg_rows = height // square_size
    bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)
    bg_img = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)

    # 前景方块
    rect_w = (160 // square_size) * square_size
    rect_h = (160 // square_size) * square_size
    fg_cols = rect_w // square_size
    fg_rows = rect_h // square_size
    fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)

    # 螺旋运动参数
    center_x = width // 2
    center_y = height // 2
    max_radius = min(width, height) // 3
    total_turns = 5  # 总圈数
    # 每帧角度增量
    angular_speed = (2 * math.pi * total_turns) / num_frames
    # 半径变化率（从内向外螺旋）
    radius_speed = max_radius / num_frames

    print(f"开始渲染螺旋运动视频: {filename}")
    print(f"前景沿螺旋轨迹运动，{total_turns}圈，最大半径={max_radius}像素")

    for i in range(num_frames):
        frame_gray = bg_img.copy()

        # 计算螺旋轨迹位置
        angle = angular_speed * i
        current_radius = radius_speed * i
        rect_x = int(center_x + current_radius * math.cos(angle) - rect_w // 2)
        rect_y = int(center_y + current_radius * math.sin(angle) - rect_h // 2)

        rect_x = max(0, min(rect_x, width - rect_w))
        rect_y = max(0, min(rect_y, height - rect_h))

        # 前景纹理自身也旋转滚动（沿两个轴）
        fg_rotated = np.roll(fg_small, shift=i, axis=1)
        fg_rotated = np.roll(fg_rotated, shift=i, axis=0)
        fg_block = np.repeat(np.repeat(fg_rotated, square_size, axis=0), square_size, axis=1)

        frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block

        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

        if i % 30 == 0:
            print(f"已生成帧: {i}/{num_frames}")

    out.release()
    print(f"螺旋运动视频生成完成: {filename}")

    if output_frames_folder:
        _save_frames_spiral(filename, width, height, square_size, num_frames,
                            bg_rows, bg_cols, fg_rows, fg_cols,
                            center_x, center_y, max_radius, total_turns)


def _save_frames_spiral(filename, width, height, square_size, num_frames,
                         bg_rows, bg_cols, fg_rows, fg_cols,
                         center_x, center_y, max_radius, total_turns):
    """辅助函数：保存螺旋运动视频的帧"""
    frames_dir = filename.rsplit('.', 1)[0] + '_frames'
    os.makedirs(frames_dir, exist_ok=True)
    print(f"开始保存帧到: {frames_dir}")

    bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)
    bg_img = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)
    fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)

    rect_w = fg_cols * square_size
    rect_h = fg_rows * square_size
    angular_speed = (2 * math.pi * total_turns) / num_frames
    radius_speed = max_radius / num_frames

    for i in range(num_frames):
        frame_gray = bg_img.copy()
        angle = angular_speed * i
        current_radius = radius_speed * i
        rect_x = int(center_x + current_radius * math.cos(angle) - rect_w // 2)
        rect_y = int(center_y + current_radius * math.sin(angle) - rect_h // 2)
        rect_x = max(0, min(rect_x, width - rect_w))
        rect_y = max(0, min(rect_y, height - rect_h))

        fg_rotated = np.roll(fg_small, shift=i, axis=1)
        fg_rotated = np.roll(fg_rotated, shift=i, axis=0)
        fg_block = np.repeat(np.repeat(fg_rotated, square_size, axis=0), square_size, axis=1)
        frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block

        frame_path = os.path.join(frames_dir, f'frame_{i:05d}.png')
        cv2.imwrite(frame_path, frame_gray)

    print(f"帧保存完成: {frames_dir}")



def generate_pulsing_zoom_video(filename, width=1280, height=720, square_size=4,
                                 fps=30, duration=15, output_frames_folder=False):
    """
    生成缩放脉冲运动视频。

    前景方块周期性放大和缩小（脉冲缩放），同时背景纹理沿径向向外扩散。
    缩放运动涉及空间尺度的变化，运动方向为径向（从中心向外或向内），
    用于测试尺度变化下的边缘检测响应。

    参数:
        filename: 输出视频文件路径
        width: 视频宽度（默认1280）
        height: 视频高度（默认720）
        square_size: 方块大小，影响纹理颗粒度（默认4）
        fps: 帧率（默认30）
        duration: 视频时长，单位秒（默认15）
        output_frames_folder: 是否将所有帧输出到与视频文件同级文件夹中（默认False）
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    num_frames = fps * duration

    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    bg_cols = width // square_size
    bg_rows = height // square_size
    bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)

    # 前景基础纹理（固定大小）
    base_fg_cols = 60
    base_fg_rows = 60
    fg_small = np.random.choice([0, 255], size=(base_fg_rows, base_fg_cols)).astype(np.uint8)

    # 脉冲参数
    min_scale = 0.3   # 最小缩放比例
    max_scale = 1.5   # 最大缩放比例
    pulse_cycles = 4  # 脉冲周期数

    print(f"开始渲染缩放脉冲运动视频: {filename}")
    print(f"前景脉冲缩放，{pulse_cycles}个周期，缩放范围[{min_scale}, {max_scale}]")

    for i in range(num_frames):
        # 背景径向滚动
        bg_rolled = np.roll(bg_small, shift=i, axis=1)
        bg_frame = np.repeat(np.repeat(bg_rolled, square_size, axis=0), square_size, axis=1)
        frame_gray = bg_frame.copy()

        # 计算当前缩放比例（正弦波脉冲）
        phase = (2 * math.pi * pulse_cycles * i) / num_frames
        scale = min_scale + (max_scale - min_scale) * (0.5 + 0.5 * math.sin(phase))

        # 根据缩放比例计算实际前景大小
        current_fg_cols = max(1, int(base_fg_cols * scale))
        current_fg_rows = max(1, int(base_fg_rows * scale))

        # 缩放前景纹理（使用最近邻插值保持方块效果）
        fg_scaled = cv2.resize(fg_small, (current_fg_cols, current_fg_rows),
                               interpolation=cv2.INTER_NEAREST)
        fg_block = np.repeat(np.repeat(fg_scaled, square_size, axis=0), square_size, axis=1)

        current_rect_w = current_fg_cols * square_size
        current_rect_h = current_fg_rows * square_size

        # 居中放置
        rect_x = (width - current_rect_w) // 2
        rect_y = (height - current_rect_h) // 2

        frame_gray[rect_y: rect_y + current_rect_h, rect_x: rect_x + current_rect_w] = fg_block

        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

        if i % 30 == 0:
            print(f"已生成帧: {i}/{num_frames}, 缩放比例: {scale:.2f}")

    out.release()
    print(f"缩放脉冲运动视频生成完成: {filename}")

    if output_frames_folder:
        _save_frames_pulsing(filename, width, height, square_size, num_frames,
                             bg_rows, bg_cols, base_fg_rows, base_fg_cols,
                             fg_small, min_scale, max_scale, pulse_cycles)


def _save_frames_pulsing(filename, width, height, square_size, num_frames,
                          bg_rows, bg_cols, base_fg_rows, base_fg_cols,
                          fg_small, min_scale, max_scale, pulse_cycles):
    """辅助函数：保存缩放脉冲视频的帧"""
    frames_dir = filename.rsplit('.', 1)[0] + '_frames'
    os.makedirs(frames_dir, exist_ok=True)
    print(f"开始保存帧到: {frames_dir}")

    bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)

    for i in range(num_frames):
        bg_rolled = np.roll(bg_small, shift=i, axis=1)
        bg_frame = np.repeat(np.repeat(bg_rolled, square_size, axis=0), square_size, axis=1)
        frame_gray = bg_frame.copy()

        phase = (2 * math.pi * pulse_cycles * i) / num_frames
        scale = min_scale + (max_scale - min_scale) * (0.5 + 0.5 * math.sin(phase))

        current_fg_cols = max(1, int(base_fg_cols * scale))
        current_fg_rows = max(1, int(base_fg_rows * scale))

        fg_scaled = cv2.resize(fg_small, (current_fg_cols, current_fg_rows),
                               interpolation=cv2.INTER_NEAREST)
        fg_block = np.repeat(np.repeat(fg_scaled, square_size, axis=0), square_size, axis=1)

        current_rect_w = current_fg_cols * square_size
        current_rect_h = current_fg_rows * square_size
        rect_x = (width - current_rect_w) // 2
        rect_y = (height - current_rect_h) // 2

        frame_gray[rect_y: rect_y + current_rect_h, rect_x: rect_x + current_rect_w] = fg_block

        frame_path = os.path.join(frames_dir, f'frame_{i:05d}.png')
        cv2.imwrite(frame_path, frame_gray)

    print(f"帧保存完成: {frames_dir}")


def generate_figure_eight_motion_video(filename, width=1280, height=720, square_size=4,
                                        fps=30, duration=15, output_frames_folder=False):
    """
    生成"8"字形（无穷符号）运动视频。

    前景方块沿"8"字形（Lissajous曲线）轨迹运动。
    该轨迹在两个正交方向上有不同的频率成分，
    运动方向持续平滑变化，是测试运动感知算法的理想复杂轨迹。

    参数:
        filename: 输出视频文件路径
        width: 视频宽度（默认1280）
        height: 视频高度（默认720）
        square_size: 方块大小，影响纹理颗粒度（默认4）
        fps: 帧率（默认30）
        duration: 视频时长，单位秒（默认15）
        output_frames_folder: 是否将所有帧输出到与视频文件同级文件夹中（默认False）
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    num_frames = fps * duration

    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    bg_cols = width // square_size
    bg_rows = height // square_size
    bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)
    bg_img = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)

    # 前景方块
    rect_w = (160 // square_size) * square_size
    rect_h = (160 // square_size) * square_size
    fg_cols = rect_w // square_size
    fg_rows = rect_h // square_size
    fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)
    fg_block = np.repeat(np.repeat(fg_small, square_size, axis=0), square_size, axis=1)

    # "8"字形轨迹参数（Lissajous曲线）
    center_x = width // 2
    center_y = height // 2
    amp_x = width // 4      # 水平振幅
    amp_y = height // 4     # 垂直振幅
    freq_x = 2              # 水平频率（完成2个周期）
    freq_y = 1              # 垂直频率（完成1个周期）→ 形成"8"字
    cycles = 3              # 总周期数

    print(f"开始渲染8字形运动视频: {filename}")
    print(f"前景沿8字形（Lissajous曲线）轨迹运动")

    for i in range(num_frames):
        frame_gray = bg_img.copy()

        # Lissajous曲线参数方程
        t = (2 * math.pi * cycles * i) / num_frames
        pos_x = center_x + amp_x * math.sin(freq_x * t)
        pos_y = center_y + amp_y * math.sin(freq_y * t + math.pi / 2)

        rect_x = int(pos_x - rect_w // 2)
        rect_y = int(pos_y - rect_h // 2)
        rect_x = max(0, min(rect_x, width - rect_w))
        rect_y = max(0, min(rect_y, height - rect_h))

        frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block

        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

        if i % 30 == 0:
            print(f"已生成帧: {i}/{num_frames}")

    out.release()
    print(f"8字形运动视频生成完成: {filename}")

    if output_frames_folder:
        _save_frames_figure_eight(filename, width, height, square_size, num_frames,
                                  bg_rows, bg_cols, fg_rows, fg_cols,
                                  center_x, center_y, amp_x, amp_y, freq_x, freq_y, cycles)


def _save_frames_figure_eight(filename, width, height, square_size, num_frames,
                               bg_rows, bg_cols, fg_rows, fg_cols,
                               center_x, center_y, amp_x, amp_y, freq_x, freq_y, cycles):
    """辅助函数：保存8字形运动视频的帧"""
    frames_dir = filename.rsplit('.', 1)[0] + '_frames'
    os.makedirs(frames_dir, exist_ok=True)
    print(f"开始保存帧到: {frames_dir}")

    bg_small = np.random.choice([0, 255], size=(bg_rows, bg_cols)).astype(np.uint8)
    bg_img = np.repeat(np.repeat(bg_small, square_size, axis=0), square_size, axis=1)
    fg_small = np.random.choice([0, 255], size=(fg_rows, fg_cols)).astype(np.uint8)
    fg_block = np.repeat(np.repeat(fg_small, square_size, axis=0), square_size, axis=1)

    rect_w = fg_cols * square_size
    rect_h = fg_rows * square_size

    for i in range(num_frames):
        frame_gray = bg_img.copy()
        t = (2 * math.pi * cycles * i) / num_frames
        pos_x = center_x + amp_x * math.sin(freq_x * t)
        pos_y = center_y + amp_y * math.sin(freq_y * t + math.pi / 2)
        rect_x = int(pos_x - rect_w // 2)
        rect_y = int(pos_y - rect_h // 2)
        rect_x = max(0, min(rect_x, width - rect_w))
        rect_y = max(0, min(rect_y, height - rect_h))
        frame_gray[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = fg_block

        frame_path = os.path.join(frames_dir, f'frame_{i:05d}.png')
        cv2.imwrite(frame_path, frame_gray)

    print(f"帧保存完成: {frames_dir}")


if __name__ == "__main__":
    # 始终把视频输出到"项目根目录/data/"，与 generators/ 同级，
    # 不受当前工作目录（在 generators/ 下还是项目根下运行）影响。
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"输出目录: {DATA_DIR}")

    print("=" * 60)
    print("生成斜向运动视频...")
    generate_diagonal_motion_video(os.path.join(DATA_DIR, 'diagonal_motion.mp4'), square_size=4)

    print("=" * 60)
    print("生成圆周运动视频...")
    generate_circular_motion_video(os.path.join(DATA_DIR, 'circular_motion.mp4'), square_size=4)

    print("=" * 60)
    print("生成螺旋运动视频...")
    generate_spiral_motion_video(os.path.join(DATA_DIR, 'spiral_motion.mp4'), square_size=4)

    print("=" * 60)
    print("生成缩放脉冲运动视频...")
    generate_pulsing_zoom_video(os.path.join(DATA_DIR, 'pulsing_zoom.mp4'), square_size=4)

    print("=" * 60)
    print("生成8字形运动视频...")
    generate_figure_eight_motion_video(os.path.join(DATA_DIR, 'figure_eight_motion.mp4'), square_size=4)

    print("=" * 60)
    print("所有复杂运动视频生成完成！")
