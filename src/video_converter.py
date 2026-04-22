"""
视频转换模块。

提供MP4视频与灰度NumPy数组之间的相互转换功能。
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

def two_gray_array_to_GB_mp4(array_G: np.ndarray, array_B: np.ndarray, output_path: str, fps: float = 25.0, codec: str = "mp4v") -> None:
    """Write two grayscale 3D NumPy arrays to an MP4 file as a color video with G and B channels.

    The input arrays must have shape (frames, height, width).
    The output video will have R=0, G=from array_G, B=from array_B.
    """
    if array_G.ndim != 3 or array_B.ndim != 3:
        raise ValueError("Input arrays must have shape (frames, height, width)")
    if array_G.shape != array_B.shape:
        raise ValueError("Both input arrays must have the same shape")

    frame_count, height, width = array_G.shape
    if frame_count == 0:
        raise ValueError("Input arrays contain no frames")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for output: {output_path}")

    for i in range(frame_count):
        frame_G = array_G[i]
        frame_B = array_B[i]
        if frame_G.shape != (height, width) or frame_B.shape != (height, width):
            raise ValueError(
                f"All frames must have shape (height, width); frame {i} has shapes {frame_G.shape} and {frame_B.shape}"
            )

        # Create color frame: R=0, G=frame_G, B=frame_B
        color_frame = np.zeros((height, width, 3), dtype=np.uint8)
        if frame_G.dtype != np.uint8:
            frame_G = np.clip(frame_G, 0, 255).astype(np.uint8)
        if frame_B.dtype != np.uint8:
            frame_B = np.clip(frame_B, 0, 255).astype(np.uint8)
        color_frame[:, :, 1] = frame_G  # G channel
        color_frame[:, :, 2] = frame_B  # B channel

        writer.write(color_frame)

    writer.release()


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