"""
灰度数组转MP4视频模块。

提供将三维灰度NumPy数组转换为MP4视频文件的功能。
支持自定义帧率和编解码器。
"""

import argparse
import os

import cv2
import numpy as np


def gray_array_to_mp4(array: np.ndarray, output_path: str, fps: float = 25.0, codec: str = "mp4v") -> None:
    """Write a grayscale 3D NumPy array to an MP4 file.

    The input array shape must be (frames, height, width).
    """
    if array.ndim != 3:
        raise ValueError("Input array must have shape (frames, height, width)")

    frame_count, height, width = array.shape
    if frame_count == 0:
        raise ValueError("Input array contains no frames")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for output: {output_path}")

    for i in range(frame_count):
        frame = array[i]
        if frame.shape != (height, width):
            raise ValueError(
                f"All frames must have shape (height, width); frame {i} has shape {frame.shape}"
            )

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a grayscale NumPy array to an MP4 file.")
    parser.add_argument("input", help="Path to the input .npy file containing a 3D array")
    parser.add_argument("output", help="Path to the output MP4 file")
    parser.add_argument("--fps", type=float, default=25.0, help="Frames per second for the output video")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec for MP4 output (default: mp4v)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    array = np.load(args.input)
    gray_array_to_mp4(array, args.output, args.fps, args.codec)
    print(f"Converted {args.input} to {args.output}")


if __name__ == "__main__":
    main()

