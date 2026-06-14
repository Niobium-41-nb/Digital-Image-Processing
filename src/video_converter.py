"""
视频转换模块。

提供MP4视频与灰度NumPy数组之间的相互转换功能。
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image

from src.progress_bar import update_progress, finish_progress


def _read_frames_from_directory(dir_path: str) -> np.ndarray:
    """从 PNG 帧序列文件夹读取灰度视频数组。

    文件夹中应包含按 frame_00000.png, frame_00001.png, ... 命名的 PNG 文件。

    返回:
        灰度三维数组，形状为 (帧数, 高度, 宽度)
    """
    import glob

    png_files = sorted(glob.glob(os.path.join(dir_path, "*.png")))
    if not png_files:
        raise FileNotFoundError(f"文件夹中未找到任何 PNG 文件: {dir_path}")

    print(f"开始读取帧序列，共 {len(png_files)} 帧")

    frames = []
    for i, png_path in enumerate(png_files):
        # 使用 PIL 读取（对中文路径兼容性更好）
        try:
            frame = np.array(Image.open(png_path).convert('L'), dtype=np.uint8)
        except Exception as e:
            raise RuntimeError(f"无法读取帧文件: {png_path}") from e
        frames.append(frame)
        update_progress(i + 1, len(png_files), prefix="读取帧序列")

    finish_progress(prefix="读取帧序列")

    if not frames:
        return np.empty((0, 0, 0), dtype=np.uint8)

    return np.stack(frames, axis=0)


def mp4_to_grayscale_array(mp4_path: str) -> np.ndarray:
    """读取视频文件或 PNG 帧序列文件夹，返回灰度三维NumPy数组。

    支持：
      - .mp4 / .avi 等视频文件（通过 OpenCV VideoCapture）
      - 包含 PNG 帧序列的文件夹（文件名如 frame_00000.png）

    返回的数组形状为 (帧数, 高度, 宽度)。
    """
    # 如果是文件夹，读取 PNG 帧序列
    if os.path.isdir(mp4_path):
        return _read_frames_from_directory(mp4_path)

    if not os.path.isfile(mp4_path):
        raise FileNotFoundError(f"未找到视频文件: {mp4_path}")

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {mp4_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"开始读取视频，共 {total_frames} 帧")

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        frame_idx += 1

        update_progress(frame_idx, total_frames, prefix="读取视频")

    cap.release()
    finish_progress(prefix="读取视频")

    if not frames:
        return np.empty((0, 0, 0), dtype=np.uint8)

    return np.stack(frames, axis=0)


def gray_array_to_mp4(array: np.ndarray, output_path: str, fps: float = 25.0, codec: str = "mp4v", output_frames_folder: bool = False) -> None:
    """将灰度三维NumPy数组写入MP4文件。

    输入数组形状必须为 (帧数, 高度, 宽度)。
    
    参数:
        array: 输入的三维灰度数组，形状为 (帧数, 高度, 宽度)
        output_path: 输出MP4文件路径
        fps: 帧率（默认25.0）
        codec: 视频编码器（默认"mp4v"）
        output_frames_folder: 是否将所有帧输出到与视频文件同级文件夹中（默认False）
    """
    if array.ndim != 3:
        raise ValueError("输入数组必须为形状 (帧数, 高度, 宽度)")

    frame_count, height, width = array.shape
    if frame_count == 0:
        raise ValueError("输入数组包含零帧")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    if not writer.isOpened():
        raise RuntimeError(f"无法打开视频写入器进行输出: {output_path}")

    print(f"开始写入视频，共 {frame_count} 帧")
    for i in range(frame_count):
        frame = array[i]
        if frame.shape != (height, width):
            raise ValueError(
                f"所有帧必须为 (高度, 宽度) 的形状; 第 {i} 帧的形状为 {frame.shape}"
            )

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        writer.write(frame)

        update_progress(i + 1, frame_count, prefix="写入视频")

    writer.release()
    finish_progress(prefix="写入视频")
    print(f"视频写入完成: {output_path}")

    # 如果需要输出帧文件夹
    if output_frames_folder:
        frames_dir = output_path.rsplit('.', 1)[0] + '_frames'
        os.makedirs(frames_dir, exist_ok=True)
        print(f"开始保存帧到: {frames_dir}")
        
        for i in range(frame_count):
            frame = array[i]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            frame_path = os.path.join(frames_dir, f'frame_{i:05d}.png')
            cv2.imwrite(frame_path, frame)
        
        print(f"帧保存完成: {frames_dir}")

def two_gray_array_to_GB_mp4(array_G: np.ndarray, array_B: np.ndarray, output_path: str, fps: float = 25.0, codec: str = "mp4v", output_frames_folder: bool = False) -> None:
    """将两个灰度三维NumPy数组作为彩色视频的G和B通道写入MP4文件。

    输入数组形状必须为 (帧数, 高度, 宽度)。
    输出视频的R=0, G通道来自array_G, B通道来自array_B。
    
    参数:
        array_G: G通道的三维灰度数组
        array_B: B通道的三维灰度数组
        output_path: 输出MP4文件路径
        fps: 帧率（默认25.0）
        codec: 视频编码器（默认"mp4v"）
        output_frames_folder: 是否将所有帧输出到与视频文件同级文件夹中（默认False）
    """
    if array_G.ndim != 3 or array_B.ndim != 3:
        raise ValueError("输入数组必须为形状 (帧数, 高度, 宽度)")
    if array_G.shape != array_B.shape:
        raise ValueError("两个输入数组必须具有相同的形状")

    frame_count, height, width = array_G.shape
    if frame_count == 0:
        raise ValueError("输入数组包含零帧")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"无法打开视频写入器进行输出: {output_path}")

    print(f"开始写入GB通道视频，共 {frame_count} 帧")
    for i in range(frame_count):
        frame_G = array_G[i]
        frame_B = array_B[i]
        if frame_G.shape != (height, width) or frame_B.shape != (height, width):
            raise ValueError(
                f"所有帧必须为 (高度, 宽度) 的形状; 第 {i} 帧的形状为 {frame_G.shape} 和 {frame_B.shape}"
            )

        color_frame = np.zeros((height, width, 3), dtype=np.uint8)
        if frame_G.dtype != np.uint8:
            frame_G = np.clip(frame_G, 0, 255).astype(np.uint8)
        if frame_B.dtype != np.uint8:
            frame_B = np.clip(frame_B, 0, 255).astype(np.uint8)
        color_frame[:, :, 1] = frame_G
        color_frame[:, :, 2] = frame_B

        writer.write(color_frame)

        update_progress(i + 1, frame_count, prefix="写入GB通道")

    writer.release()
    finish_progress(prefix="写入GB通道")
    print(f"GB通道视频写入完成: {output_path}")

    # 如果需要输出帧文件夹
    if output_frames_folder:
        frames_dir = output_path.rsplit('.', 1)[0] + '_frames'
        os.makedirs(frames_dir, exist_ok=True)
        print(f"开始保存帧到: {frames_dir}")
        
        for i in range(frame_count):
            frame_G = array_G[i]
            frame_B = array_B[i]
            if frame_G.dtype != np.uint8:
                frame_G = np.clip(frame_G, 0, 255).astype(np.uint8)
            if frame_B.dtype != np.uint8:
                frame_B = np.clip(frame_B, 0, 255).astype(np.uint8)
            
            color_frame = np.zeros((height, width, 3), dtype=np.uint8)
            color_frame[:, :, 1] = frame_G
            color_frame[:, :, 2] = frame_B
            
            frame_path = os.path.join(frames_dir, f'frame_{i:05d}.png')
            cv2.imwrite(frame_path, color_frame)
        
        print(f"帧保存完成: {frames_dir}")

def two_gray_array_to_RG_mp4(array_R: np.ndarray, array_G: np.ndarray, output_path: str, fps: float = 25.0, codec: str = "mp4v", output_frames_folder: bool = False) -> None:
    """将两个灰度三维NumPy数组作为彩色视频的R和G通道写入MP4文件。

    输入数组形状必须为 (帧数, 高度, 宽度)。
    输出视频的R通道来自array_R, G通道来自array_G, B=0。
    
    参数:
        array_R: R通道的三维灰度数组
        array_G: G通道的三维灰度数组
        output_path: 输出MP4文件路径
        fps: 帧率（默认25.0）
        codec: 视频编码器（默认"mp4v"）
        output_frames_folder: 是否将所有帧输出到与视频文件同级文件夹中（默认False）
    """
    if array_R.ndim != 3 or array_G.ndim != 3:
        raise ValueError("输入数组必须为形状 (帧数, 高度, 宽度)")
    if array_R.shape != array_G.shape:
        raise ValueError("两个输入数组必须具有相同的形状")

    frame_count, height, width = array_R.shape
    if frame_count == 0:
        raise ValueError("输入数组包含零帧")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"无法打开视频写入器进行输出: {output_path}")

    print(f"开始写入RG通道视频，共 {frame_count} 帧")
    for i in range(frame_count):
        frame_R = array_R[i]
        frame_G = array_G[i]
        if frame_R.shape != (height, width) or frame_G.shape != (height, width):
            raise ValueError(
                f"所有帧必须为 (高度, 宽度) 的形状; 第 {i} 帧的形状为 {frame_R.shape} 和 {frame_G.shape}"
            )

        color_frame = np.zeros((height, width, 3), dtype=np.uint8)
        if frame_R.dtype != np.uint8:
            frame_R = np.clip(frame_R, 0, 255).astype(np.uint8)
        if frame_G.dtype != np.uint8:
            frame_G = np.clip(frame_G, 0, 255).astype(np.uint8)
        color_frame[:, :, 2] = frame_R  # OpenCV 使用 BGR 格式，R 通道在 index 2
        color_frame[:, :, 1] = frame_G  # G 通道在 index 1

        writer.write(color_frame)

        update_progress(i + 1, frame_count, prefix="写入RG通道")

    writer.release()
    finish_progress(prefix="写入RG通道")
    print(f"RG通道视频写入完成: {output_path}")

    # 如果需要输出帧文件夹
    if output_frames_folder:
        frames_dir = output_path.rsplit('.', 1)[0] + '_frames'
        os.makedirs(frames_dir, exist_ok=True)
        print(f"开始保存帧到: {frames_dir}")
        
        for i in range(frame_count):
            frame_R = array_R[i]
            frame_G = array_G[i]
            if frame_R.dtype != np.uint8:
                frame_R = np.clip(frame_R, 0, 255).astype(np.uint8)
            if frame_G.dtype != np.uint8:
                frame_G = np.clip(frame_G, 0, 255).astype(np.uint8)
            
            color_frame = np.zeros((height, width, 3), dtype=np.uint8)
            color_frame[:, :, 2] = frame_R
            color_frame[:, :, 1] = frame_G
            
            frame_path = os.path.join(frames_dir, f'frame_{i:05d}.png')
            cv2.imwrite(frame_path, color_frame)
        
        print(f"帧保存完成: {frames_dir}")


def three_gray_array_to_RGB_mp4(array_R: np.ndarray, array_G: np.ndarray, array_B: np.ndarray, output_path: str, fps: float = 25.0, codec: str = "mp4v", output_frames_folder: bool = False) -> None:
    """将三个灰度三维NumPy数组作为彩色视频的RGB通道写入MP4文件。

    输入数组形状必须为 (帧数, 高度, 宽度)。
    输出视频的R通道来自array_R, G通道来自array_G, B通道来自array_B。
    
    参数:
        array_R: R通道的三维灰度数组
        array_G: G通道的三维灰度数组
        array_B: B通道的三维灰度数组
        output_path: 输出MP4文件路径
        fps: 帧率（默认25.0）
        codec: 视频编码器（默认"mp4v"）
        output_frames_folder: 是否将所有帧输出到与视频文件同级文件夹中（默认False）
    """
    if array_R.ndim != 3 or array_G.ndim != 3 or array_B.ndim != 3:
        raise ValueError("输入数组必须为形状 (帧数, 高度, 宽度)")
    if array_R.shape != array_G.shape or array_R.shape != array_B.shape:
        raise ValueError("三个输入数组必须具有相同的形状")

    frame_count, height, width = array_R.shape
    if frame_count == 0:
        raise ValueError("输入数组包含零帧")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"无法打开视频写入器进行输出: {output_path}")

    print(f"开始写入RGB通道视频，共 {frame_count} 帧")
    for i in range(frame_count):
        frame_R = array_R[i]
        frame_G = array_G[i]
        frame_B = array_B[i]
        if frame_R.shape != (height, width) or frame_G.shape != (height, width) or frame_B.shape != (height, width):
            raise ValueError(
                f"所有帧必须为 (高度, 宽度) 的形状; 第 {i} 帧的形状为 {frame_R.shape}, {frame_G.shape} 和 {frame_B.shape}"
            )

        color_frame = np.zeros((height, width, 3), dtype=np.uint8)
        if frame_R.dtype != np.uint8:
            frame_R = np.clip(frame_R, 0, 255).astype(np.uint8)
        if frame_G.dtype != np.uint8:
            frame_G = np.clip(frame_G, 0, 255).astype(np.uint8)
        if frame_B.dtype != np.uint8:
            frame_B = np.clip(frame_B, 0, 255).astype(np.uint8)
        color_frame[:, :, 2] = frame_R
        color_frame[:, :, 1] = frame_G
        color_frame[:, :, 0] = frame_B

        writer.write(color_frame)

        update_progress(i + 1, frame_count, prefix="写入RGB通道")

    writer.release()
    finish_progress(prefix="写入RGB通道")
    print(f"RGB通道视频写入完成: {output_path}")

    # 如果需要输出帧文件夹
    if output_frames_folder:
        frames_dir = output_path.rsplit('.', 1)[0] + '_frames'
        os.makedirs(frames_dir, exist_ok=True)
        print(f"开始保存帧到: {frames_dir}")
        
        for i in range(frame_count):
            frame_R = array_R[i]
            frame_G = array_G[i]
            frame_B = array_B[i]
            if frame_R.dtype != np.uint8:
                frame_R = np.clip(frame_R, 0, 255).astype(np.uint8)
            if frame_G.dtype != np.uint8:
                frame_G = np.clip(frame_G, 0, 255).astype(np.uint8)
            if frame_B.dtype != np.uint8:
                frame_B = np.clip(frame_B, 0, 255).astype(np.uint8)
            
            color_frame = np.zeros((height, width, 3), dtype=np.uint8)
            color_frame[:, :, 2] = frame_R
            color_frame[:, :, 1] = frame_G
            color_frame[:, :, 0] = frame_B
            
            frame_path = os.path.join(frames_dir, f'frame_{i:05d}.png')
            cv2.imwrite(frame_path, color_frame)
        
        print(f"帧保存完成: {frames_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="将MP4转换为灰度三维NumPy数组。")
    parser.add_argument("input", help="输入MP4文件路径")
    parser.add_argument("--output", "-o", help="保存输出.npy文件路径", default=None)
    args = parser.parse_args()

    array = mp4_to_grayscale_array(args.input)
    print(f"已加载视频: {args.input}")
    print(f"数组形状: {array.shape}")
    print(f"数组dtype: {array.dtype}")

    if args.output:
        np.save(args.output, array)
        print(f"已将灰度数组保存至: {args.output}")


if __name__ == "__main__":
    main()