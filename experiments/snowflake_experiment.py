"""
============================================================
实验：雪花马赛克视频处理
============================================================

对 data/snowfall.mp4 进行以下处理：

  1. 将视频旋转指定角度（用户可设置）
  2. 使用 3D 卷积核 (5,5,5) * 1/125 对视频做时间平滑处理
  3. 对每一帧进行水平方向的边缘检测
  4. 计算边缘检测结果的平均值并输出

依赖：
  - opencv-python (cv2)
  - numpy
  - scipy
============================================================
"""

import os
import sys

import cv2
import numpy as np

# 将项目根目录加入 sys.path，以便导入 src 模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.video_converter import mp4_to_grayscale_array, gray_array_to_mp4
from src.temporal_convolver import (
    create_temporal_motion_blur,
    apply_temporal_convolution,
    apply_frame_convolution,
)
from src.edge_process import SPATIAL_180


# ============================================================
# 1. 图像旋转
# ============================================================
def rotate_video(arr: np.ndarray, angle: float) -> np.ndarray:
    """
    将视频的每一帧旋转指定角度。

    参数:
        arr: 输入视频数组，形状 (frames, height, width)
        angle: 旋转角度（度），正数表示逆时针旋转

    返回:
        旋转后的视频数组，形状 (frames, height, width)
        旋转后超出原画面的区域用黑色（0）填充
    """
    frames, height, width = arr.shape
    # 获取旋转矩阵
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # 计算旋转后图像的新边界尺寸，使图像完整显示
    cos_abs = abs(rotation_matrix[0, 0])
    sin_abs = abs(rotation_matrix[0, 1])
    new_width = int(height * sin_abs + width * cos_abs)
    new_height = int(height * cos_abs + width * sin_abs)

    # 调整旋转矩阵，使图像居中
    rotation_matrix[0, 2] += new_width / 2 - center[0]
    rotation_matrix[1, 2] += new_height / 2 - center[1]

    print(f"旋转角度: {angle}°, 原始尺寸: {width}x{height}, 新尺寸: {new_width}x{new_height}")

    result = np.zeros((frames, new_height, new_width), dtype=arr.dtype)
    for i in range(frames):
        result[i] = cv2.warpAffine(
            arr[i], rotation_matrix, (new_width, new_height),
            flags=cv2.INTER_NEAREST,  # 最近邻插值，保持方块效果
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return result


# ============================================================
# 2. 计算边缘检测结果的平均值（排除旋转产生的黑色填充区域）
# ============================================================
def compute_edge_average(edge_arr: np.ndarray, mask: np.ndarray = None) -> float:
    """
    计算边缘检测结果的平均值，排除旋转产生的黑色填充区域。

    旋转后图像四角会出现黑色填充区域（像素值为0），
    这些区域不包含有效图像内容，应排除在平均值计算之外。

    参数:
        edge_arr: 边缘检测结果数组，形状 (frames, height, width)
        mask: 有效像素掩码，形状 (height, width)，True 表示有效像素。
              如果为 None，则使用 edge_arr > 0 作为掩码

    返回:
        有效区域内的全局平均值（float）
    """
    if mask is not None:
        # 使用传入的掩码，对所有帧应用相同的空间掩码
        masked_values = edge_arr[:, mask]
    else:
        # 如果没有掩码，排除值为0的像素
        masked_values = edge_arr[edge_arr > 0]

    if masked_values.size == 0:
        return 0.0

    avg = np.mean(masked_values)
    return float(avg)


# ============================================================
# 主流程
# ============================================================
def main():
    # ========== 输入参数 ==========
    video_path = os.path.join(PROJECT_ROOT, 'data', 'snowfall.mp4')
    rotate_angle = 45.0  # 用户可设置的旋转角度（度）

    # 如果提供了命令行参数，使用命令行参数
    if len(sys.argv) > 1:
        rotate_angle = float(sys.argv[1])

    print("=" * 60)
    print("雪花马赛克视频处理实验")
    print("=" * 60)
    print(f"输入视频: {video_path}")
    print(f"旋转角度: {rotate_angle}°")

    # ========== 读取视频 ==========
    print("\n>>> 步骤1: 读取视频")
    arr = mp4_to_grayscale_array(video_path)
    print(f"视频数组形状: {arr.shape}, dtype: {arr.dtype}")

    # ========== 旋转 ==========
    print("\n>>> 步骤2: 旋转视频")
    rotated = rotate_video(arr, rotate_angle)
    print(f"旋转后形状: {rotated.shape}")

    # 保存旋转后的视频
    output_dir = os.path.join(PROJECT_ROOT, 'output', 'snowflake_experiment')
    os.makedirs(output_dir, exist_ok=True)
    gray_array_to_mp4(rotated, os.path.join(output_dir, '01_rotated.mp4'), fps=30)
    print(f"旋转后视频已保存")

    # 创建有效像素掩码：排除旋转产生的黑色填充区域
    # 取第一帧的非零像素作为掩码（所有帧共享相同的空间掩码）
    valid_mask = rotated[0] > 0
    valid_pixel_count = np.sum(valid_mask)
    total_pixel_count = valid_mask.size
    print(f"有效像素: {valid_pixel_count}/{total_pixel_count} "
          f"({100.0 * valid_pixel_count / total_pixel_count:.1f}%)")

    # ========== 3D 时间平滑 ==========
    print("\n>>> 步骤3: 3D 时间平滑 (5,5,5) * 1/125")
    # 使用 src.temporal_convolver 创建 (5,5,5) 均匀平滑核并执行时间卷积
    smoothing_kernel = create_temporal_motion_blur(time_frames=5, height=5, width=5)
    print(f"3D 时间平滑核: {smoothing_kernel.shape}, 元素值: 1/125 = {1/125:.6f}")
    smoothed = apply_temporal_convolution(rotated, smoothing_kernel)
    print(f"平滑后形状: {smoothed.shape}")

    # 保存平滑后的视频
    gray_array_to_mp4(smoothed, os.path.join(output_dir, '02_smoothed.mp4'), fps=30)
    print(f"平滑后视频已保存")

    # ========== 水平边缘检测 ==========
    print("\n>>> 步骤4: 水平方向边缘检测")
    # 使用 src.edge_process.SPATIAL_180 核（水平边缘检测核）和 src.temporal_convolver.apply_frame_convolution
    edges = apply_frame_convolution(smoothed, SPATIAL_180)
    # 取绝对值（边缘强度）
    edges = np.abs(edges)
    print(f"边缘检测结果形状: {edges.shape}")

    # 保存边缘检测结果视频（归一化到 0-255 显示）
    edges_display = np.clip(edges / edges.max() * 255, 0, 255).astype(np.uint8) if edges.max() > 0 else edges.astype(np.uint8)
    gray_array_to_mp4(edges_display, os.path.join(output_dir, '03_edges_horizontal.mp4'), fps=30)
    print(f"边缘检测视频已保存")

    # ========== 计算平均值（排除黑色填充区域）==========
    print("\n>>> 步骤5: 计算边缘检测结果的平均值（排除旋转产生的黑色填充区域）")
    avg_value = compute_edge_average(edges, mask=valid_mask)
    print(f"\n{'=' * 60}")
    print(f"边缘检测结果平均值（排除空值）: {avg_value:.6f}")
    print(f"{'=' * 60}")

    # ========== 输出汇总 ==========
    print(f"\n所有结果已保存到: {output_dir}")
    print(f"  1. 旋转后视频: 01_rotated.mp4")
    print(f"  2. 时间平滑后: 02_smoothed.mp4")
    print(f"  3. 水平边缘检测: 03_edges_horizontal.mp4")
    print(f"  4. 边缘平均值: {avg_value:.6f}")


if __name__ == "__main__":
    main()
