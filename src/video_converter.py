"""
视频转换模块。

提供MP4视频与灰度NumPy数组之间的相互转换功能。
"""

import argparse
import os

import cv2
import numpy as np


def mp4_to_grayscale_array(mp4_path: str) -> np.ndarray:
    """读取MP4文件并返回灰度三维NumPy数组。

    返回的数组形状为 (帧数, 高度, 宽度)。
    """
    if not os.path.isfile(mp4_path):
        raise FileNotFoundError(f"未找到MP4文件: {mp4_path}")

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {mp4_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    cap.release()

    if not frames:
        return np.empty((0, 0, 0), dtype=np.uint8)

    return np.stack(frames, axis=0)


def gray_array_to_mp4(array: np.ndarray, output_path: str, fps: float = 25.0, codec: str = "mp4v") -> None:
    """将灰度三维NumPy数组写入MP4文件。

    输入数组形状必须为 (帧数, 高度, 宽度)。
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

    for i in range(frame_count):
        frame = array[i]
        if frame.shape != (height, width):
            raise ValueError(
                f"所有帧必须为 (高度, 宽度) 的形状; 第 {i} 帧的形状为 {frame.shape}"
            )

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()

def two_gray_array_to_GB_mp4(array_G: np.ndarray, array_B: np.ndarray, output_path: str, fps: float = 25.0, codec: str = "mp4v") -> None:
    """将两个灰度三维NumPy数组作为彩色视频的G和B通道写入MP4文件。

    输入数组形状必须为 (帧数, 高度, 宽度)。
    输出视频的R=0, G通道来自array_G, B通道来自array_B。
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

    writer.release()

def three_gray_array_to_RGB_mp4(array_R: np.ndarray, array_G: np.ndarray, array_B: np.ndarray, output_path: str, fps: float = 25.0, codec: str = "mp4v") -> None:
    """将三个灰度三维NumPy数组作为彩色视频的RGB通道写入MP4文件。

    输入数组形状必须为 (帧数, 高度, 宽度)。
    输出视频的R通道来自array_R, G通道来自array_G, B通道来自array_B。
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
        color_frame[:, :, 2] = frame_R  # OpenCV使用BGR顺序
        color_frame[:, :, 1] = frame_G
        color_frame[:, :, 0] = frame_B

        writer.write(color_frame)

    writer.release()

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