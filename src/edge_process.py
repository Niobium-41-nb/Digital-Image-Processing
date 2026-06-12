"""
============================================================
边缘处理模块 —— 真正的3D卷积边缘检测核与流水线处理函数
============================================================

本模块将边缘检测视为 **真正的3D卷积** 操作：
  卷积核形状为 (T, 3, 3)，其中 T > 1 是时间维度，
  同时对视频的连续多帧进行时间-空间联合卷积。

设计原理：
  传统的"时间运动模糊 + 空间边缘检测"是分离的两步操作。
  本模块将时间平滑核与空间边缘检测核通过 **外积 (outer product)**
  组合为统一的 3D 卷积核，一次卷积同时完成时间平滑和空间边缘检测。

  3D 卷积核构造方式：
    1. 时间核 kernel_t: 长度为 T 的一维核（如均值平滑 [1,1,...,1]/T）
    2. 空间核 kernel_s: 形状为 (3, 3) 的二维方向边缘检测核（Prewitt 类）
    3. 3D 核: kernel_3d = kernel_t[:, np.newaxis, np.newaxis] * kernel_s[np.newaxis, :, :]
       形状为 (T, 3, 3)

  这样，scipy.ndimage.convolve 沿三维数组 (frames, H, W) 做三维卷积时，
  核同时在时间轴和空间轴上滑动，实现真正的 3D 卷积。

输出：
  - 各方向边缘检测结果分别输出为独立灰度视频
  - 帧间差分结果经最大池化后作为独立灰度视频输出

依赖：
  - numpy, scipy
  - src/video_converter.py
============================================================
"""

import gc
import os
import sys

import numpy as np
from scipy.ndimage import convolve, maximum_filter

from src.progress_bar import update_progress, finish_progress
from src.video_converter import gray_array_to_mp4


# ============================================================
# 空间方向边缘检测核（2D 核，形状 (3, 3)）
# ============================================================
# 这些是纯空间域的 Prewitt 类方向边缘检测核。
# 它们将与时间核组合为 3D 卷积核 (T, 3, 3)。
#
# 基础四方向：
#   1. 水平方向（0°）—— 检测垂直边缘
#      [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
#   2. 45° 方向
#      [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
#   3. 135° 方向
#      [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
#   4. 180° 方向（水平边缘）
#      [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
#
# 扩展四方向（通过转置得到）：
#   5. 垂直方向（90°）—— 水平核转置
#   6. -45° 方向 —— 45° 核转置
#   7. -135° 方向 —— 135° 核转置
#   8. 90° 方向（检测垂直边缘）—— 180° 核转置

# --- 1. 水平方向（0°）—— 检测垂直边缘 ---
SPATIAL_HORIZONTAL = np.array([
    [-1.0,  0.0,  1.0],
    [-1.0,  0.0,  1.0],
    [-1.0,  0.0,  1.0],
], dtype=np.float64)

# --- 5. 垂直方向（90°）—— 检测水平边缘（水平核转置） ---
SPATIAL_VERTICAL = SPATIAL_HORIZONTAL.T

# --- 2. 45° 方向 ---
SPATIAL_45 = np.array([
    [-2.0,  -1.0,  0.0],
    [-1.0,  0.0,  1.0],
    [ 0.0,  1.0,  2.0],
], dtype=np.float64)

# --- 6. -45° 方向（45° 核转置） ---
SPATIAL_NEG45 = SPATIAL_45.T

# --- 3. 135° 方向 ---
SPATIAL_135 = np.array([
    [ 0.0,  1.0,  2.0],
    [-1.0,  0.0,  1.0],
    [-2.0,  -1.0,  0.0],
], dtype=np.float64)

# --- 7. -135° 方向（135° 核转置） ---
SPATIAL_NEG135 = SPATIAL_135.T

# --- 4. 180° 方向（水平边缘） ---
SPATIAL_180 = np.array([
    [ 1.0,  1.0,  1.0],
    [ 0.0,  0.0,  0.0],
    [-1.0, -1.0, -1.0],
], dtype=np.float64)

# --- 8. 90° 方向（检测垂直边缘，180° 核转置） ---
SPATIAL_90 = SPATIAL_180.T

# 所有空间核的列表，用于批量生成 3D 核
_SPATIAL_KERNELS = [
    ("horizontal", SPATIAL_HORIZONTAL),
    ("45",         SPATIAL_45),
    ("135",        SPATIAL_135),
    ("180",        SPATIAL_180),
    ("vertical",   SPATIAL_VERTICAL),
    ("neg45",      SPATIAL_NEG45),
    ("neg135",     SPATIAL_NEG135),
    ("90",         SPATIAL_90),
]


# ============================================================
# 3D 卷积核生成函数
# ============================================================
def create_3d_edge_kernel(temporal_size: int, spatial_kernel_2d: np.ndarray) -> np.ndarray:
    """
    通过时间核与空间核的外积，构造真正的 3D 卷积核。

    将边缘检测视为 3D 卷积核 (T, 3, 3) 对 3D 矩阵 (frames, H, W) 的卷积。

    参数:
        temporal_size: 时间维度大小 T（即同时卷积的连续帧数）
        spatial_kernel_2d: 空间方向边缘检测核，形状 (3, 3)

    返回:
        3D 卷积核，形状 (T, 3, 3)
        其中时间维度使用均值平滑核 [1/T, 1/T, ..., 1/T]，
        空间维度使用传入的方向边缘检测核。

    数学原理:
        设时间核 k_t ∈ R^T，空间核 k_s ∈ R^{3×3}，
        则 3D 核 K ∈ R^{T×3×3} 定义为：
            K[t, i, j] = k_t[t] * k_s[i, j]

        对视频 V ∈ R^{F×H×W} 做 3D 卷积：
            (V * K)[f, i, j] = Σ_t Σ_u Σ_v V[f-t, i-u, j-v] * K[t, u, v]

        这等价于先对时间轴做 1D 卷积（平滑），再对空间轴做 2D 卷积（边缘检测），
        但由于 scipy.ndimage.convolve 直接支持 3D 卷积，一次调用即可完成。
    """
    # 时间核：均值平滑，长度为 temporal_size
    temporal_kernel = np.ones(temporal_size, dtype=np.float64) / temporal_size

    # 通过外积构造 3D 核：temporal_kernel (T,) ⊗ spatial_kernel_2d (3, 3)
    # 结果形状: (T, 3, 3)
    kernel_3d = temporal_kernel[:, np.newaxis, np.newaxis] * spatial_kernel_2d[np.newaxis, :, :]

    # 乘以 10 以增强边缘响应幅度（保持与原代码一致的缩放）
    kernel_3d *= 10.0

    return kernel_3d


def create_all_3d_edge_kernels(temporal_size: int) -> dict:
    """
    为所有 8 个方向创建 3D 卷积核。

    参数:
        temporal_size: 时间维度大小 T

    返回:
        字典，键为方向名称字符串，值为形状 (T, 3, 3) 的 3D 卷积核
    """
    return {
        name: create_3d_edge_kernel(temporal_size, kernel_2d)
        for name, kernel_2d in _SPATIAL_KERNELS
    }

def process_motion_blurred_array(blurred_arr, output_dir, suffix="",
                                  pool_size=4, max_memory=5 * 1024 ** 3,
                                  enable_diff=True,
                                  temporal_size=3,
                                  method="prewitt",
                                  tile_size=512):
    """
    方法分发器：根据 method 参数选择 Prewitt 3D 卷积或 3D FFT 频域分析。

    参数:
        blurred_arr: 输入视频的三维数组，形状 (frames, height, width)
        output_dir: 输出目录路径
        suffix: 输出文件名的后缀
        pool_size: 最大池化核大小（pool_size=1 时不进行池化降采样）
        max_memory: 最大允许内存（字节）
        enable_diff: 是否计算帧间差分
        temporal_size: 3D 卷积核的时间维度大小 T（Prewitt 方法使用）
        method: "prewitt"（3D 卷积）或 "fft"（3D FFT 频域分析）
        tile_size: FFT 方法的空间分块大小（仅 method="fft" 时生效）
    """
    if method == "fft":
        from src.fft_3d_analysis import process_fft_3d_analysis
        process_fft_3d_analysis(
            blurred_arr, output_dir, suffix=suffix,
            pool_size=pool_size, max_memory=max_memory,
            enable_diff=enable_diff, tile_size=tile_size,
        )
    else:
        _process_prewitt(
            blurred_arr, output_dir, suffix=suffix,
            pool_size=pool_size, max_memory=max_memory,
            enable_diff=enable_diff, temporal_size=temporal_size,
        )


def _process_prewitt(blurred_arr, output_dir, suffix="",
                     pool_size=4, max_memory=5 * 1024 ** 3,
                     enable_diff=True,
                     temporal_size=3):
    """
    对视频数组执行 **真正的3D卷积边缘检测** 和帧间差分。

    本函数将边缘检测视为 3D 卷积核 (T, 3, 3) 对 3D 矩阵 (frames, H, W)
    的卷积操作。卷积核的时间维度 T > 1，同时处理连续多帧，
    一次卷积同时完成时间平滑和空间边缘检测。

    流式流水线处理：3D卷积边缘检测 → 池化 → 差分 合并为单次遍历，
    避免存储大尺寸中间结果。

    八方向 3D 卷积边缘检测（每个核形状 (T, 3, 3)）：
      - 水平方向（0°）：时间均值平滑 × Prewitt 水平核
      - 45° 方向：时间均值平滑 × 45° 核
      - 135° 方向：时间均值平滑 × 135° 核
      - 180° 方向：时间均值平滑 × 180° 核
      - 垂直方向（90°）：时间均值平滑 × 垂直核
      - -45° 方向：时间均值平滑 × -45° 核
      - -135° 方向：时间均值平滑 × -135° 核
      - 90° 方向：时间均值平滑 × 90° 核

    输出：
      - 各方向边缘检测结果分别输出为独立灰度视频
      - 帧间差分结果经最大池化后作为独立灰度视频输出

    参数:
        blurred_arr: 输入视频的三维数组，形状 (frames, height, width)
        output_dir: 输出目录路径
        suffix: 输出文件名的后缀
        pool_size: 最大池化核大小（pool_size=1 时不进行池化降采样）
        max_memory: 最大允许内存（字节）
        enable_diff: 是否计算帧间差分（False 时不输出差分视频）
        temporal_size: 3D 卷积核的时间维度大小 T（默认 3，即同时卷积 3 帧）
    """
    num_frames = blurred_arr.shape[0]
    height, width = blurred_arr.shape[1], blurred_arr.shape[2]

    # 池化后的目标尺寸
    target_height = height // pool_size
    target_width = width // pool_size

    # 创建所有方向的 3D 卷积核 (T, 3, 3)
    kernels_3d = create_all_3d_edge_kernels(temporal_size)

    # 计算分块大小
    bytes_per_frame = height * width * 9 * 4  # 输入帧 + 8结果
    chunk_size = max(1, max_memory // bytes_per_frame)
    chunk_size = min(chunk_size, 15)

    if pool_size > 1:
        print(f"开始3D卷积边缘检测{suffix}，共 {num_frames} 帧，"
              f"尺寸 {height}x{width} → 池化后 {target_height}x{target_width}，"
              f"3D核时间维度 T={temporal_size}，分块 {chunk_size} 帧")
    else:
        print(f"开始3D卷积边缘检测{suffix}，共 {num_frames} 帧，"
              f"尺寸 {height}x{width}（不进行池化降采样），"
              f"3D核时间维度 T={temporal_size}，分块 {chunk_size} 帧")

    # 预分配结果数组（8方向，不再有纯空间四方向）
    result_horizontal = np.zeros((num_frames, target_height, target_width), dtype=np.float32)
    result_45 = np.zeros((num_frames, target_height, target_width), dtype=np.float32)
    result_135 = np.zeros((num_frames, target_height, target_width), dtype=np.float32)
    result_180 = np.zeros((num_frames, target_height, target_width), dtype=np.float32)
    result_vertical = np.zeros((num_frames, target_height, target_width), dtype=np.float32)
    result_neg45 = np.zeros((num_frames, target_height, target_width), dtype=np.float32)
    result_neg135 = np.zeros((num_frames, target_height, target_width), dtype=np.float32)
    result_90 = np.zeros((num_frames, target_height, target_width), dtype=np.float32)
    if enable_diff:
        result_diff = np.zeros((num_frames, target_height, target_width), dtype=np.float32)

    # 流式处理：逐块完成 3D卷积边缘检测 + 池化 + 差分
    prev_frame = None

    for start in range(0, num_frames, chunk_size):
        end = min(start + chunk_size, num_frames)

        # 处理当前块中的每一帧
        for i in range(start, end):
            # ============================================================
            # 真正的 3D 卷积边缘检测
            # ============================================================
            # 对于第 i 帧，取以 i 为中心的连续 temporal_size 帧
            # 构成 (T, H, W) 的子块，用 3D 核 (T, 3, 3) 做卷积。
            #
            # scipy.ndimage.convolve 直接支持对三维数组做三维卷积，
            # 核在时间轴和空间轴上同时滑动。
            #
            # 边界处理：对靠近边界的帧进行零填充。
            half_t = temporal_size // 2
            t_start = i - half_t
            t_end = t_start + temporal_size

            # 构建子块：对时间边界进行零填充
            if t_start >= 0 and t_end <= num_frames:
                sub_block = blurred_arr[t_start:t_end].astype(np.float64)
            else:
                sub_block = np.zeros((temporal_size, height, width), dtype=np.float64)
                valid_start = max(0, t_start)
                valid_end = min(num_frames, t_end)
                copy_start = valid_start - t_start
                sub_block[copy_start:copy_start + (valid_end - valid_start)] = \
                    blurred_arr[valid_start:valid_end].astype(np.float64)

            # 使用 3D 卷积核 (T, 3, 3) 对 3D 子块 (T, H, W) 做真正的 3D 卷积
            # 结果形状: (T, H, W)，取中间帧（第 half_t 帧）作为当前帧的边缘结果
            edge_h = convolve(sub_block, kernels_3d["horizontal"], mode='constant', cval=0.0)[half_t]
            edge_45 = convolve(sub_block, kernels_3d["45"], mode='constant', cval=0.0)[half_t]
            edge_135 = convolve(sub_block, kernels_3d["135"], mode='constant', cval=0.0)[half_t]
            edge_180 = convolve(sub_block, kernels_3d["180"], mode='constant', cval=0.0)[half_t]
            edge_vertical = convolve(sub_block, kernels_3d["vertical"], mode='constant', cval=0.0)[half_t]
            edge_neg45 = convolve(sub_block, kernels_3d["neg45"], mode='constant', cval=0.0)[half_t]
            edge_neg135 = convolve(sub_block, kernels_3d["neg135"], mode='constant', cval=0.0)[half_t]
            edge_90 = convolve(sub_block, kernels_3d["90"], mode='constant', cval=0.0)[half_t]

            # === 步骤2：池化 ===
            if pool_size > 1:
                result_horizontal[i] = maximum_filter(edge_h, size=pool_size, mode='constant', cval=0.0)[::pool_size, ::pool_size]
                result_45[i] = maximum_filter(edge_45, size=pool_size, mode='constant', cval=0.0)[::pool_size, ::pool_size]
                result_135[i] = maximum_filter(edge_135, size=pool_size, mode='constant', cval=0.0)[::pool_size, ::pool_size]
                result_180[i] = maximum_filter(edge_180, size=pool_size, mode='constant', cval=0.0)[::pool_size, ::pool_size]
                result_vertical[i] = maximum_filter(edge_vertical, size=pool_size, mode='constant', cval=0.0)[::pool_size, ::pool_size]
                result_neg45[i] = maximum_filter(edge_neg45, size=pool_size, mode='constant', cval=0.0)[::pool_size, ::pool_size]
                result_neg135[i] = maximum_filter(edge_neg135, size=pool_size, mode='constant', cval=0.0)[::pool_size, ::pool_size]
                result_90[i] = maximum_filter(edge_90, size=pool_size, mode='constant', cval=0.0)[::pool_size, ::pool_size]
            else:
                result_horizontal[i] = edge_h
                result_45[i] = edge_45
                result_135[i] = edge_135
                result_180[i] = edge_180
                result_vertical[i] = edge_vertical
                result_neg45[i] = edge_neg45
                result_neg135[i] = edge_neg135
                result_90[i] = edge_90

            # === 步骤3：帧间差分 ===
            if enable_diff:
                if i == 0:
                    result_diff[i] = np.zeros((target_height, target_width), dtype=np.float32)
                else:
                    if i == start and prev_frame is not None:
                        prev = prev_frame
                    else:
                        prev = blurred_arr[i - 1].astype(np.float32)
                    diff_frame = np.abs(blurred_arr[i].astype(np.float32) - prev)
                    if pool_size > 1:
                        result_diff[i] = maximum_filter(diff_frame, size=pool_size, mode='constant', cval=0.0)[::pool_size, ::pool_size]
                    else:
                        result_diff[i] = diff_frame

            # 释放临时变量
            del sub_block, edge_h, edge_45, edge_135, edge_180
            del edge_vertical, edge_neg45, edge_neg135, edge_90

            prefix = f"3D卷积边缘检测{suffix}" if pool_size > 1 else f"3D卷积边缘检测{suffix}"
            update_progress(i + 1, num_frames, prefix=prefix)

        if enable_diff:
            prev_frame = blurred_arr[end - 1].astype(np.float32)

        gc.collect()

    prefix = f"3D卷积边缘检测{suffix}" if pool_size > 1 else f"3D卷积边缘检测{suffix}"
    finish_progress(prefix=prefix)
    if pool_size > 1:
        print(f"3D卷积边缘检测完成{suffix}: {height}x{width} → {target_height}x{target_width}")
    else:
        print(f"3D卷积边缘检测完成{suffix}: 保持原尺寸 {height}x{width}")

    # ========== 保存各处理结果为独立灰度视频 ==========
    gray_array_to_mp4(blurred_arr, f"{output_dir}/output{suffix}.mp4")
    gray_array_to_mp4(result_horizontal, f"{output_dir}/output_edge_horizontal{suffix}.mp4")
    gray_array_to_mp4(result_45, f"{output_dir}/output_edge_45deg{suffix}.mp4")
    gray_array_to_mp4(result_135, f"{output_dir}/output_edge_135deg{suffix}.mp4")
    gray_array_to_mp4(result_180, f"{output_dir}/output_edge_180deg{suffix}.mp4")
    gray_array_to_mp4(result_vertical, f"{output_dir}/output_edge_vertical{suffix}.mp4")
    gray_array_to_mp4(result_neg45, f"{output_dir}/output_edge_neg45deg{suffix}.mp4")
    gray_array_to_mp4(result_neg135, f"{output_dir}/output_edge_neg135deg{suffix}.mp4")
    gray_array_to_mp4(result_90, f"{output_dir}/output_edge_90deg{suffix}.mp4")
    if enable_diff:
        gray_array_to_mp4(result_diff, f"{output_dir}/output_diff{suffix}.mp4")

    # 释放内存
    del result_horizontal, result_45, result_135, result_180
    del result_vertical, result_neg45, result_neg135, result_90
    if enable_diff:
        del result_diff
    gc.collect()


def modulate_by_diff(edge_result: np.ndarray, diff_result: np.ndarray) -> np.ndarray:
    """
    使用帧间差分结果作为亮度调制信号，对边缘检测结果进行亮度调制。

    原理：
      帧间差分值反映了运动强度（绝对值越大 = 运动越剧烈）。
      将差分值归一化到 [0, 1] 范围后，作为乘性因子与边缘强度相乘，
      使得运动区域的边缘被保留并增强，静止区域的边缘被抑制（变暗）。

    参数:
        edge_result: 边缘检测结果数组，形状 (frames, H, W)，float64
        diff_result: 帧间差分结果数组，形状 (frames, H, W)，float64

    返回:
        亮度调制后的边缘数组，形状与输入相同
    """
    diff_abs = np.abs(diff_result)
    max_vals = diff_abs.max(axis=(1, 2), keepdims=True)
    max_vals = np.where(max_vals == 0, 1.0, max_vals)
    diff_norm = diff_abs / max_vals
    return edge_result * diff_norm


if __name__ == "__main__":
    # 直接运行时添加项目根目录到 sys.path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.video_converter import mp4_to_grayscale_array

    # 简单测试：读取视频并测试流式处理
    videos = [f for f in os.listdir("data") if f.endswith(".mp4")]
    if videos:
        test_video = os.path.join("data", videos[0])
        print(f"测试视频: {test_video}")
        arr = mp4_to_grayscale_array(test_video)
        print(f"数组形状: {arr.shape}")
        process_motion_blurred_array(arr, "output", suffix="_test", pool_size=4, enable_diff=True)
    else:
        print("data/ 目录下没有视频文件，跳过测试")
