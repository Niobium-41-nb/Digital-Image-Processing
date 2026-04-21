"""
MP4视频转灰度数组模块。

提供将MP4视频文件读取为三维灰度NumPy数组的功能。
输出数组形状为(帧数, 高度, 宽度)。
"""

import argparse
import os

import cv2
import numpy as np


def mp4_to_grayscale_array(mp4_path: str) -> np.ndarray:
    """Read an MP4 file and return a grayscale 3D NumPy array.

    The returned array shape is (frame_count, height, width).
    """
    if not os.path.isfile(mp4_path):
        raise FileNotFoundError(f"MP4 file not found: {mp4_path}")

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {mp4_path}")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MP4 to a grayscale 3D NumPy array.")
    parser.add_argument("input", help="Path to the input MP4 file")
    parser.add_argument("--output", "-o", help="Path to save the output .npy file", default=None)
    args = parser.parse_args()

    array = mp4_to_grayscale_array(args.input)
    print(f"Loaded video: {args.input}")
    print(f"Array shape: {array.shape}")
    print(f"Array dtype: {array.dtype}")

    if args.output:
        np.save(args.output, array)
        print(f"Saved grayscale array to: {args.output}")


if __name__ == "__main__":
    main()
