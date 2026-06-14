"""
生成雪花马赛克滚动视频模块（微小尺寸版）。

与 snowflake.py 的区别：
  - 视频尺寸固定为 20×20 像素
  - 方块大小固定为 2 像素
  - 适用于快速测试和调试

依赖:
    - opencv-python (cv2)
    - numpy
"""

import cv2
import numpy as np
from PIL import Image


def generate_snowflake_video(
    filename,
    width=20,
    height=20,
    square_size=2,
    fps=30,
    duration=15,
    speed_x=1,
    speed_y=0,
    output_frames_folder=False,
):
    """
    生成雪花马赛克滚动视频（微小尺寸版，20×20）。

    该函数创建具有随机黑白方块纹理（雪花噪点风格）向固定方向循环滚动的视频。
    整个画面由随机黑白方块组成，纹理沿指定方向持续循环滚动，
    类似电视雪花噪点向一个方向流动的效果。

    参数:
        filename: 输出视频文件路径
        width: 视频宽度（默认20）
        height: 视频高度（默认20）
        square_size: 方块大小，影响纹理颗粒度（默认2）
        fps: 帧率（默认30）
        duration: 视频时长，单位秒（默认15）
        speed_x: 水平滚动速度，正数向右，负数向左，0表示不滚动（默认1）
        speed_y: 垂直滚动速度，正数向下，负数向上，0表示不滚动（默认0）
        output_frames_folder: 是否将所有帧输出到与视频文件同级文件夹中（默认False）

    返回:
        无返回值，生成视频文件保存到指定路径
    """
    # 计算总帧数
    num_frames = fps * duration

    # 将宽高调整为方块大小的整数倍，确保像素完整性
    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    # 计算小图像的行列数（以方块为单位）
    cols = width // square_size
    rows = height // square_size

    # 生成随机黑白小图像（0或255），类似雪花噪点
    small = np.random.choice([0, 255], size=(rows, cols)).astype(np.uint8)

    # 输出帧文件夹
    frames_dir = filename.rsplit('.', 1)[0] + '_frames'
    os.makedirs(frames_dir, exist_ok=True)

    print(f"开始渲染雪花马赛克滚动视频帧序列: {frames_dir}")
    print(f"视频尺寸: {width}×{height}, 方块边长: {square_size}像素, 滚动速度: ({speed_x}, {speed_y}) 方块/帧")

    # 逐帧生成并保存为 PNG
    for i in range(num_frames):
        # 将小图像放大为实际尺寸
        frame_gray = np.repeat(np.repeat(small, square_size, axis=0), square_size, axis=1)

        # 保存帧为 PNG（使用 PIL 避免 OpenCV 对小尺寸 PNG 的兼容问题）
        frame_path = os.path.join(frames_dir, f'frame_{i:05d}.png')
        Image.fromarray(frame_gray, mode='L').save(frame_path)

        # 纹理向指定方向循环滚动
        if speed_x != 0:
            small = np.roll(small, shift=speed_x, axis=1)
        if speed_y != 0:
            small = np.roll(small, shift=speed_y, axis=0)

        # 每30帧输出一次进度信息
        if i % 30 == 0:
            print(f"已生成帧: {i}/{num_frames}")

    print(f"帧序列生成完成: {frames_dir}  ({num_frames} 帧)")


if __name__ == "__main__":
    import os

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    os.makedirs(DATA_DIR, exist_ok=True)
    generate_snowflake_video(
        os.path.join(DATA_DIR, 'snowfall_tiny_50.avi'),
        width=50,
        height=50,
        square_size=4,
        speed_x=1,
        speed_y=1,
    )
