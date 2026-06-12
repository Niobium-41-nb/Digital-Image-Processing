"""
============================================================
3D FFT 时空频域边缘检测模块
============================================================

用真正的 3D FFT 频域方向滤波替代 Prewitt 3D 卷积，
在 (f_x, f_y, f_t) 频率空间中直接进行时空方向分析。

核心原理：
  - 3D FFT 将视频块 (T, H, W) 变换到频域
  - 空间边缘方向 → 频域 (f_x, f_y) 平面中的角度
  - 运动信号 → f_t ≠ 0 的频率分量
  - 静止纹理 → f_t = 0 的纯空间频率

方法：
  分块 3D FFT + 重叠相加（overlap-add）
  - 空间分块：将大帧切割为 tile_size × tile_size 块
  - 时间完整：每块保留全部帧以保持时间频率分辨率
  - Hann 窗 + 50% 重叠：保证无缝重建

输出：
  - 8 个方向的频域边缘检测视频
  - 时间平滑视频（时间低通）
  - 运动能量视频（时间高通）★ 新增
  - 帧间差分视频

依赖：
  - numpy, scipy
  - src/progress_bar.py
  - src/video_converter.py
============================================================
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.fft import fftn, ifftn

from src.progress_bar import update_progress, finish_progress
from src.video_converter import gray_array_to_mp4


# ============================================================
# 常量
# ============================================================

# 8 个空间方向（角度制，用于滤波器命名和中心角）
_DIRECTION_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

# 方向名称映射
_DIRECTION_NAMES = [
    "horizontal",    #  0°   — 检测垂直边缘
    "45deg",          #  45°  — 检测 45° 斜边缘
    "vertical",       #  90°  — 检测水平边缘
    "135deg",         # 135°  — 检测 135° 斜边缘
    "180deg",         # 180°  — 检测水平边缘（反向）
    "neg135deg",      # 225°  — 检测 -135° 斜边缘
    "neg45deg",       # 270°  — 检测 -45° 斜边缘
    "90deg",          # 315°  — 检测垂直边缘（反向）
]


# ============================================================
# 3D 正弦窗（分析窗 + 合成窗）
# ============================================================
# 使用正弦窗 sin(π(n+0.5)/N) 替代 Hann 窗。
# 该窗满足 WOLA-COLA：sin²(θ) + sin²(θ+π/2) = 1 精确成立，
# 确保 50% 重叠时重建无周期性亮度条纹。
#
# 时域不加窗（矩形窗），避免首尾帧因除以近零权重而放大噪声。
# 时域无分块重叠，矩形窗的频谱泄漏不影响重建正确性。
#
# 设计：
#   analysis_window:  空间正弦 + 时域矩形 → FFT 前乘
#   synthesis_window: 空间正弦 + 时域全 1 → IFFT 后乘
#   weight_window   = analysis_window * synthesis_window → 归一化分母


def create_3d_sine_window(shape: tuple) -> np.ndarray:
    """
    创建 3D 分析窗：空间正弦窗 + 时域矩形窗。

    空间维度使用 sin(π(n+0.5)/N)，精确满足 WOLA-COLA 条件。
    时域不加窗（全 1 矩形窗），避免首尾帧归一化放大噪声。

    参数:
        shape: (frames, height, width)

    返回:
        float32 数组，形状与 shape 相同
    """
    F, H, W = shape

    # 时域：矩形窗（全 1），不加窗
    w_t = np.ones(F, dtype=np.float32)

    # 空间：sin(π(n+0.5)/N)，满足 sin²(θ) + sin²(θ+π/2) = 1
    if H > 1:
        w_y = np.sin(np.pi * (np.arange(H, dtype=np.float32) + 0.5) / H)
    else:
        w_y = np.ones(1, dtype=np.float32)

    if W > 1:
        w_x = np.sin(np.pi * (np.arange(W, dtype=np.float32) + 0.5) / W)
    else:
        w_x = np.ones(1, dtype=np.float32)

    window = (
        w_t[:, np.newaxis, np.newaxis]
        * w_y[np.newaxis, :, np.newaxis]
        * w_x[np.newaxis, np.newaxis, :]
    )
    return window


def create_3d_synthesis_window(shape: tuple) -> np.ndarray:
    """
    创建 3D 合成窗：空间正弦窗 + 时域全 1。

    与 create_3d_sine_window 等价（时域均为全 1），
    单独定义以保持分析/合成窗语义清晰。

    参数:
        shape: (frames, height, width)

    返回:
        float32 数组，形状与 shape 相同
    """
    return create_3d_sine_window(shape)


# ============================================================
# 频域方向滤波器
# ============================================================

def create_directional_filters(
    freq_shape: tuple,
    n_directions: int = 8,
    angular_width: float = np.pi / 8,
    spatial_center: float = 0.30,
    spatial_sigma: float = 0.18,
    dc_suppress_width: float = 0.02,
) -> list[np.ndarray]:
    """
    在 3D 频域中创建方向扇形滤波器组（全频率空间）。

    改用 fftn 全频率空间（fx ∈ [-0.5, 0.5]），确保所有角度
    方向的频率分量都能被平等访问。

    每个滤波器：
      H(f_x, f_y, f_t) = A(θ) × B(|f_spatial|)
    其中：
      A(θ)   — 角度选择性（高斯扇区）
      B(r)    — 空间带通（高斯带通，抑制 DC 和 Nyquist）

    参数:
        freq_shape: fftn 输出的频域形状 (F, H, W)
        n_directions: 方向数（默认 8）
        angular_width: 角度高斯窗的标准差（弧度，默认 π/8 = 22.5°）
        spatial_center: 空间带通中心（相对于 Nyquist，默认 0.30，峰值 ≈ 3.3 像素宽特征）
        spatial_sigma: 空间带通宽度（默认 0.18，覆盖 2~10 像素特征）
        dc_suppress_width: DC 抑制区宽度（默认 0.02）

    返回:
        filters: list of np.ndarray，每个形状为 freq_shape，float32
    """
    F, H, W = freq_shape

    # 构建全频率坐标（fftn 模式）
    if F > 1:
        ft = np.fft.fftfreq(F).astype(np.float32)
        ft_grid = ft[:, np.newaxis, np.newaxis]  # (F, 1, 1)
    else:
        ft_grid = np.zeros((1, 1, 1), dtype=np.float32)

    # f_y: 全频率范围 [-0.5, 0.5]
    fy = np.fft.fftfreq(H).astype(np.float32)
    fy_grid = fy[np.newaxis, :, np.newaxis]  # (1, H, 1)

    # f_x: 全频率范围 [-0.5, 0.5]
    fx = np.fft.fftfreq(W).astype(np.float32)
    fx_grid = fx[np.newaxis, np.newaxis, :]  # (1, 1, W)

    # 空间频率幅值
    f_spatial = np.sqrt(fx_grid ** 2 + fy_grid ** 2)  # (1, H, W)

    # 空间角度（完整 [-π, π] 范围）
    theta = np.arctan2(fy_grid, fx_grid)  # (1, H, W)，范围 [-π, π]

    # 空间带通滤波器
    bandpass = np.exp(-((f_spatial - spatial_center) ** 2) / (2.0 * spatial_sigma ** 2))
    bandpass = bandpass.astype(np.float32)

    # DC 抑制
    dc_transition = np.clip(f_spatial / dc_suppress_width, 0.0, 1.0).astype(np.float32)
    bandpass = bandpass * dc_transition

    # 为每个方向创建角度滤波器
    direction_angles_rad = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)

    filters = []
    for dir_angle in direction_angles_rad:
        # 角度差（频域方向滤波器需要对 180° 对称：
        # 一条过原点的直线在频域中 θ 和 θ+π 表示同一个方向，
        # 滤波器必须满足 H(-f) = H(f)（共轭对称）才能让 ifftn 不起虚部）
        d1 = np.abs(theta - dir_angle)
        d1 = np.minimum(d1, 2.0 * np.pi - d1)          # 到 dir_angle 的角距离
        d2 = np.abs(theta - (dir_angle + np.pi))
        d2 = np.minimum(d2, 2.0 * np.pi - d2)          # 到 dir_angle + π 的角距离
        dtheta = np.minimum(d1, d2)                    # 取最近者

        # 高斯角度窗
        angular = np.exp(-(dtheta ** 2) / (2.0 * angular_width ** 2))

        # 组合滤波器：角度 × 空间带通
        fan_filter = (angular * bandpass).astype(np.float32)

        # 广播到时间维度
        fan_filter_3d = np.broadcast_to(fan_filter, (F, H, W)).copy()

        filters.append(fan_filter_3d)

    return filters


def create_temporal_motion_filter(
    freq_shape: tuple,
    ft_cutoff: float = 0.04,
    sharpness: float = 3.0,
) -> np.ndarray:
    """
    创建时间高通滤波器，用于隔离运动信号。

    M(f_t) = 1 − exp(−(|f_t| / f_cutoff)^p)

    低频（静止纹理）被抑制，高频（运动）通过。

    参数:
        freq_shape: fftn 输出的频域形状 (F, H, W)
        ft_cutoff: 截止频率（相对于 Nyquist，默认 0.04）
        sharpness: 过渡带锐度（默认 3.0，越大越陡）

    返回:
        float32 数组，形状为 freq_shape
    """
    F, H, W = freq_shape

    if F > 1:
        ft = np.fft.fftfreq(F).astype(np.float32)
    else:
        ft = np.zeros(F, dtype=np.float32)

    ft_abs = np.abs(ft)
    ft_grid = ft_abs[:, np.newaxis, np.newaxis]  # (F, 1, 1)

    # 高斯补（平滑高通）
    motion_filter = 1.0 - np.exp(-((ft_grid / ft_cutoff) ** sharpness))

    # 广播到空间维度
    motion_filter = np.broadcast_to(motion_filter.astype(np.float32), (F, H, W)).copy()

    return motion_filter


def create_temporal_lowpass_filter(
    freq_shape: tuple,
    ft_cutoff: float = 0.08,
    sharpness: float = 3.0,
) -> np.ndarray:
    """
    创建时间低通滤波器，用于提取平滑/静止纹理。

    M(f_t) = exp(−(|f_t| / f_cutoff)^p)

    参数:
        freq_shape: fftn 输出的频域形状 (F, H, W)
        ft_cutoff: 截止频率（相对于 Nyquist，默认 0.08）
        sharpness: 过渡带锐度（默认 3.0）

    返回:
        float32 数组，形状为 freq_shape
    """
    F, H, W = freq_shape

    if F > 1:
        ft = np.fft.fftfreq(F).astype(np.float32)
    else:
        ft = np.zeros(F, dtype=np.float32)

    ft_abs = np.abs(ft)
    ft_grid = ft_abs[:, np.newaxis, np.newaxis]  # (F, 1, 1)

    lowpass_filter = np.exp(-((ft_grid / ft_cutoff) ** sharpness))

    lowpass_filter = np.broadcast_to(lowpass_filter.astype(np.float32), (F, H, W)).copy()

    return lowpass_filter


# ============================================================
# 分块方案计算
# ============================================================

def compute_tile_positions(
    height: int,
    width: int,
    tile_size: int,
    overlap_ratio: float = 0.5,
) -> list[tuple]:
    """
    计算空间分块的位置列表（重叠相加方案）。

    每个分块在空间域中以 overlap_ratio 的比例重叠。
    tile_size 是分块的实际宽度/高度（所有分块等大）。

    参数:
        height: 视频帧高度
        width: 视频帧宽度
        tile_size: 分块大小（正方形边长）
        overlap_ratio: 重叠比例（默认 0.5 = 50%）

    返回:
        [(y_start, y_end, x_start, x_end), ...] 列表
    """
    stride = int(tile_size * (1.0 - overlap_ratio))
    stride = max(1, stride)
    half_tile = tile_size // 2

    # 分块延伸超出图像边界，确保图像边缘像素也获得完整双块覆盖
    positions = []
    for y in range(-half_tile, height + half_tile, stride):
        for x in range(-half_tile, width + half_tile, stride):
            # 跳过与图像完全无交叠的分块
            if y + tile_size <= 0 or y >= height:
                continue
            if x + tile_size <= 0 or x >= width:
                continue
            y_start = y
            y_end = y + tile_size
            x_start = x
            x_end = x + tile_size
            positions.append((y_start, y_end, x_start, x_end))

    return positions


# ============================================================
# 核心处理函数
# ============================================================

def process_fft_3d_analysis(
    video_array: np.ndarray,
    output_dir: str,
    suffix: str = "",
    pool_size: int = 2,
    max_memory: int = 5 * 1024 ** 3,
    enable_diff: bool = True,
    tile_size: int = 512,
    overlap_ratio: float = 0.5,
) -> None:
    """
    主流水线：分块 3D FFT 时空频域方向分析。

    参数:
        video_array: 输入视频 (F, H, W)，uint8 或 float64
        output_dir: 输出目录
        suffix: 输出文件名后缀
        pool_size: 最大池化核大小（1=不池化）
        max_memory: 内存预算（字节）
        enable_diff: 是否计算帧间差分
        tile_size: 空间分块大小（默认 512）
        overlap_ratio: 分块重叠比例（默认 0.5）
    """
    # 确保在 float32 下工作
    if video_array.dtype == np.uint8:
        arr = video_array.astype(np.float32)
    else:
        arr = video_array.astype(np.float32)

    num_frames, height, width = arr.shape
    spatial_size = height * width
    total_pixels = num_frames * spatial_size

    # 如果视频尺寸小于 tile_size，使用单块模式
    if height <= tile_size and width <= tile_size:
        tile_size_actual = max(height, width)
        tile_size_actual = ((tile_size_actual + 15) // 16) * 16  # 对齐到 16
        _process_single_block(arr, tile_size_actual, output_dir, suffix,
                              pool_size, enable_diff)
        return

    # 计算目标尺寸
    if pool_size > 1:
        target_height = height // pool_size
        target_width = width // pool_size
    else:
        target_height = height
        target_width = width

    print(f"\n{'=' * 70}")
    print(f"3D FFT 时空频域分析")
    print(f"{'=' * 70}")
    print(f"  视频尺寸: {num_frames} 帧 × {height}×{width}")
    print(f"  分块大小: {tile_size}×{tile_size}，重叠比: {overlap_ratio}")
    if pool_size > 1:
        print(f"  池化: {pool_size}×{pool_size} → {target_height}×{target_width}")
    else:
        print(f"  池化: 不进行")
    print(f"  帧间差分: {'启用' if enable_diff else '禁用'}")

    # ============================================================
    # 创建频域滤波器（对所有分块通用）
    # ============================================================
    # 分块在时间轴上完整保留
    freq_shape = (num_frames, tile_size, tile_size)

    print(f"\n>>> 创建 8 方向频域扇形滤波器")
    directional_filters = create_directional_filters(freq_shape, n_directions=8)

    print(f">>> 创建时间低通滤波器（平滑）")
    lowpass_filter = create_temporal_lowpass_filter(freq_shape)

    print(f">>> 创建时间高通滤波器（运动能量）")
    motion_filter = create_temporal_motion_filter(freq_shape)

    # ============================================================
    # 计算分块方案
    # ============================================================
    tile_positions = compute_tile_positions(height, width, tile_size, overlap_ratio)
    num_tiles = len(tile_positions)
    print(f"\n>>> 分块方案: {num_tiles} 个空间分块")

    # ============================================================
    # 创建 memmap 累加器（磁盘后备，节省内存）
    # ============================================================
    tmp_dir = os.path.join(output_dir, ".fft_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    accumulators = []
    for name in _DIRECTION_NAMES:
        acc_path = os.path.join(tmp_dir, f"acc_{name}{suffix}.dat")
        acc = np.memmap(acc_path, dtype=np.float32, mode='w+',
                        shape=(num_frames, height, width))
        accumulators.append(acc)

    # 平滑纹理累加器
    smoothed_path = os.path.join(tmp_dir, f"acc_smoothed{suffix}.dat")
    smoothed_acc = np.memmap(smoothed_path, dtype=np.float32, mode='w+',
                             shape=(num_frames, height, width))

    # 运动能量累加器
    motion_path = os.path.join(tmp_dir, f"acc_motion{suffix}.dat")
    motion_acc = np.memmap(motion_path, dtype=np.float32, mode='w+',
                           shape=(num_frames, height, width))

    # 权重累加器（用于归一化）
    weight_path = os.path.join(tmp_dir, f"acc_weight{suffix}.dat")
    weight_acc = np.memmap(weight_path, dtype=np.float32, mode='w+',
                           shape=(num_frames, height, width))

    # ============================================================
    # 创建 3D 窗（分析窗 + 合成窗）
    # ============================================================
    # 所有分块统一使用 tile_size × tile_size 窗，边界分块同样零填充到 tile_size
    tile_shape_3d = (num_frames, tile_size, tile_size)
    analysis_window = create_3d_sine_window(tile_shape_3d)
    synthesis_window = create_3d_synthesis_window(tile_shape_3d)
    weight_window = analysis_window * synthesis_window  # 空间: sin², 时域: 全 1

    # ============================================================
    # 逐块处理
    # ============================================================
    for tile_idx, (y_s, y_e, x_s, x_e) in enumerate(tile_positions):
        # 分块中有效图像数据的范围（处理超出图像边界的分块）
        data_y_s = max(0, y_s)
        data_y_e = min(height, y_e)
        data_x_s = max(0, x_s)
        data_x_e = min(width, x_e)
        # 有效数据在 tile 内的偏移
        tile_y_off = data_y_s - y_s
        tile_x_off = data_x_s - x_s
        tile_data_h = data_y_e - data_y_s
        tile_data_w = data_x_e - data_x_s

        # 所有分块统一零填充到 tile_size，使用相同的窗和滤波器
        tile_data = np.zeros(tile_shape_3d, dtype=np.float32)
        if tile_data_h > 0 and tile_data_w > 0:
            tile_data[:, tile_y_off:tile_y_off + tile_data_h,
                      tile_x_off:tile_x_off + tile_data_w] = \
                arr[:, data_y_s:data_y_e, data_x_s:data_x_e]

        # 分析窗
        windowed = tile_data * analysis_window

        # === 3D FFT ===
        fft_data = fftn(windowed)  # complex64, full frequency space

        # 累加器写入范围（仅图像有效区域）
        acc_y_s = max(0, y_s)
        acc_y_e = min(height, y_e)
        acc_x_s = max(0, x_s)
        acc_x_e = min(width, x_e)
        acc_tile_y_off = acc_y_s - y_s
        acc_tile_x_off = acc_x_s - x_s
        acc_h = acc_y_e - acc_y_s
        acc_w = acc_x_e - acc_x_s

        # === 8 方向滤波 ===
        for di in range(8):
            filtered_fft = fft_data * directional_filters[di]
            spatial_result = ifftn(filtered_fft).real.astype(np.float32)

            # 合成窗 × 重叠相加（仅写入图像有效区域）
            weighted_result = spatial_result * synthesis_window
            accumulators[di][:, acc_y_s:acc_y_e, acc_x_s:acc_x_e] += \
                weighted_result[:, acc_tile_y_off:acc_tile_y_off + acc_h,
                                acc_tile_x_off:acc_tile_x_off + acc_w]

        # 权重累加器（所有方向共用）
        weight_acc[:, acc_y_s:acc_y_e, acc_x_s:acc_x_e] += \
            weight_window[:, acc_tile_y_off:acc_tile_y_off + acc_h,
                          acc_tile_x_off:acc_tile_x_off + acc_w]

        # === 时间低通（平滑纹理）===
        smoothed_fft = fft_data * lowpass_filter
        smoothed_result = ifftn(smoothed_fft).real.astype(np.float32)
        weighted_smoothed = smoothed_result * synthesis_window
        smoothed_acc[:, acc_y_s:acc_y_e, acc_x_s:acc_x_e] += \
            weighted_smoothed[:, acc_tile_y_off:acc_tile_y_off + acc_h,
                              acc_tile_x_off:acc_tile_x_off + acc_w]

        # === 时间高通（运动能量）===
        motion_fft = fft_data * motion_filter
        motion_result = ifftn(motion_fft).real.astype(np.float32)
        motion_abs = np.abs(motion_result)
        weighted_motion = motion_abs * synthesis_window
        motion_acc[:, acc_y_s:acc_y_e, acc_x_s:acc_x_e] += \
            weighted_motion[:, acc_tile_y_off:acc_tile_y_off + acc_h,
                            acc_tile_x_off:acc_tile_x_off + acc_w]

        # 释放中间变量
        del fft_data, windowed, tile_data, spatial_result
        del smoothed_result, motion_result, motion_abs
        gc.collect()

        update_progress(tile_idx + 1, num_tiles, prefix="3D FFT 分块处理")

    finish_progress(prefix="3D FFT 分块处理")
    print(f"3D FFT 处理完成: {num_tiles} 个分块")

    # ============================================================
    # 归一化：除以权重累加器
    # ============================================================
    print(f"\n>>> 重叠相加归一化")
    epsilon = np.finfo(np.float32).eps
    valid_weight = np.maximum(weight_acc, epsilon)

    for di in range(8):
        accumulators[di] /= valid_weight
    smoothed_acc /= valid_weight
    motion_acc /= valid_weight

    # 限制值范围（仅对平滑和运动能量做软限幅，8 方向边缘保持全动态范围）
    smoothed_acc_norm = np.clip(smoothed_acc, 0, 255)
    motion_acc_norm = np.clip(motion_acc, 0, 255)

    # ============================================================
    # 最大池化（可选）
    # ============================================================
    if pool_size > 1:
        from scipy.ndimage import maximum_filter

        print(f">>> 最大池化: {pool_size}×{pool_size}")

        # 先读取 memmap 到内存（避免直接原地修改尺寸不匹配的 memmap）
        # 逐帧池化后收集到新数组
        for di in range(8):
            pooled_frames = []
            for i in range(num_frames):
                frame = np.array(accumulators[di][i])  # 从 memmap 读一帧
                pooled = maximum_filter(frame, size=pool_size, mode='constant', cval=0.0)[::pool_size, ::pool_size]
                pooled_frames.append(pooled)
                update_progress(i + 1, num_frames,
                                prefix=f"池化 {_DIRECTION_NAMES[di]}")
            accumulators[di] = np.stack(pooled_frames, axis=0)
            del pooled_frames
            gc.collect()

        # 平滑也做池化
        pooled_smooth = []
        for i in range(num_frames):
            pool_s = maximum_filter(smoothed_acc_norm[i], size=pool_size, mode='constant', cval=0.0)[::pool_size, ::pool_size]
            pooled_smooth.append(pool_s)
        smoothed_acc_norm = np.stack(pooled_smooth, axis=0)
        del pooled_smooth

        # 运动能量也做池化
        pooled_motion = []
        for i in range(num_frames):
            pool_m = maximum_filter(motion_acc_norm[i], size=pool_size, mode='constant', cval=0.0)[::pool_size, ::pool_size]
            pooled_motion.append(pool_m)
        motion_acc_norm = np.stack(pooled_motion, axis=0)
        del pooled_motion

        finish_progress(prefix="池化")
        print(f"池化完成")

    # ============================================================
    # 帧间差分
    # ============================================================
    if enable_diff:
        print(f">>> 计算帧间差分")
        if pool_size > 1:
            diff_shape = (num_frames, target_height, target_width)
        else:
            diff_shape = (num_frames, height, width)
        diff_result = np.zeros(diff_shape, dtype=np.float32)

        for i in range(1, num_frames):
            if pool_size > 1:
                prev_frame = smoothed_acc_norm[i - 1]
                curr_frame = smoothed_acc_norm[i]
            else:
                prev_frame = smoothed_acc_norm[i - 1]
                curr_frame = smoothed_acc_norm[i]
            diff_result[i] = np.abs(curr_frame - prev_frame)
            update_progress(i, num_frames, prefix="帧间差分")

        diff_result[0] = 0.0
        finish_progress(prefix="帧间差分")
    else:
        diff_result = None

    # ============================================================
    # 保存输出视频
    # ============================================================
    print(f"\n>>> 保存输出视频到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 平滑视频
    smoothed_uint8 = np.clip(smoothed_acc_norm, 0, 255).astype(np.uint8)
    gray_array_to_mp4(smoothed_uint8,
                      os.path.join(output_dir, f"output_fft_smoothed{suffix}.mp4"))

    # 8 方向边缘视频（用 99 百分位数归一化，避免离群值压暗整体画面）
    for di, name in enumerate(_DIRECTION_NAMES):
        edge_abs = np.abs(accumulators[di])
        p99 = np.percentile(edge_abs, 99.0)
        scale = max(p99, 1.0)
        edge_uint8 = np.clip(edge_abs / scale * 255, 0, 255).astype(np.uint8)
        gray_array_to_mp4(edge_uint8,
                          os.path.join(output_dir, f"output_fft_{name}{suffix}.mp4"))

    # 运动能量视频
    motion_uint8 = np.clip(motion_acc_norm / max(motion_acc_norm.max(), 1.0) * 255, 0, 255).astype(np.uint8)
    gray_array_to_mp4(motion_uint8,
                      os.path.join(output_dir, f"output_fft_motion_energy{suffix}.mp4"))

    # 帧间差分
    if enable_diff and diff_result is not None:
        diff_uint8 = np.clip(diff_result, 0, 255).astype(np.uint8)
        gray_array_to_mp4(diff_uint8,
                          os.path.join(output_dir, f"output_fft_diff{suffix}.mp4"))

    # ============================================================
    # 清理临时文件
    # ============================================================
    print(f"\n>>> 清理临时文件")
    del accumulators, smoothed_acc, motion_acc, weight_acc
    gc.collect()

    # 删除 memmap 临时文件
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"3D FFT 时空频域分析完成！")


def _process_single_block(
    arr: np.ndarray,
    tile_size: int,
    output_dir: str,
    suffix: str,
    pool_size: int,
    enable_diff: bool,
) -> None:
    """单块模式：视频尺寸小于 tile_size 时直接处理，不做空间分块。"""
    num_frames, height, width = arr.shape

    # 补齐到 tile_size
    pad_h = tile_size - height
    pad_w = tile_size - width
    padded = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)),
                    mode='constant', constant_values=0)

    tile_shape = (num_frames, tile_size, tile_size)
    freq_shape = (num_frames, tile_size, tile_size)

    print(f"\n单块模式: {num_frames}×{tile_size}×{tile_size}")

    # 创建滤波器和窗
    directional_filters = create_directional_filters(freq_shape, n_directions=8)
    lowpass_filter = create_temporal_lowpass_filter(freq_shape)
    motion_filter = create_temporal_motion_filter(freq_shape)
    window_3d = create_3d_sine_window(tile_shape)

    windowed = padded * window_3d
    fft_data = fftn(windowed)

    # 目标尺寸
    if pool_size > 1:
        target_h, target_w = height // pool_size, width // pool_size
    else:
        target_h, target_w = height, width

    # 处理每个方向
    edge_results = []
    for di in range(8):
        filtered = ifftn(fft_data * directional_filters[di]).real.astype(np.float32)
        # 裁剪掉补齐的部分
        result = filtered[:, :height, :width]
        edge_results.append(result)
        update_progress(di + 1, 8, prefix="单块方向滤波")

    # 平滑
    smoothed = ifftn(fft_data * lowpass_filter).real.astype(np.float32)
    smoothed = smoothed[:, :height, :width]

    # 运动能量
    motion = ifftn(fft_data * motion_filter).real.astype(np.float32)
    motion = np.abs(motion[:, :height, :width])

    finish_progress(prefix="单块方向滤波")

    # 池化
    if pool_size > 1:
        from scipy.ndimage import maximum_filter
        for di in range(8):
            for i in range(num_frames):
                edge_results[di][i] = maximum_filter(
                    edge_results[di][i], size=pool_size, mode='constant', cval=0.0
                )[::pool_size, ::pool_size]

    # 帧间差分
    if enable_diff:
        diff_result = np.zeros((num_frames, target_h, target_w), dtype=np.float32)
        for i in range(1, num_frames):
            diff_result[i] = np.abs(smoothed[i] - smoothed[i - 1])
    else:
        diff_result = None

    # 保存
    os.makedirs(output_dir, exist_ok=True)

    smoothed_uint8 = np.clip(smoothed, 0, 255).astype(np.uint8)
    gray_array_to_mp4(smoothed_uint8,
                      os.path.join(output_dir, f"output_fft_smoothed{suffix}.mp4"))

    for di, name in enumerate(_DIRECTION_NAMES):
        edge_abs = np.abs(edge_results[di])
        p99 = np.percentile(edge_abs, 99.0)
        scale = max(p99, 1.0)
        edge_uint8 = np.clip(edge_abs / scale * 255, 0, 255).astype(np.uint8)
        gray_array_to_mp4(edge_uint8,
                          os.path.join(output_dir, f"output_fft_{name}{suffix}.mp4"))

    motion_m = motion.max()
    motion_uint8 = np.clip(motion / max(motion_m, 1.0) * 255, 0, 255).astype(np.uint8)
    gray_array_to_mp4(motion_uint8,
                      os.path.join(output_dir, f"output_fft_motion_energy{suffix}.mp4"))

    if enable_diff and diff_result is not None:
        diff_uint8 = np.clip(diff_result, 0, 255).astype(np.uint8)
        gray_array_to_mp4(diff_uint8,
                          os.path.join(output_dir, f"output_fft_diff{suffix}.mp4"))

    print(f"单块模式处理完成！")


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.video_converter import mp4_to_grayscale_array

    # 简单测试：读取 data/ 目录下第一个视频
    data_dir = os.path.join(project_root, "data")
    videos = [f for f in os.listdir(data_dir) if f.endswith(".mp4")] if os.path.isdir(data_dir) else []

    if videos:
        test_video = os.path.join(data_dir, videos[0])
        print(f"测试视频: {test_video}")
        arr = mp4_to_grayscale_array(test_video)
        print(f"数组形状: {arr.shape}")

        test_output = os.path.join(project_root, "output", "fft_test")
        process_fft_3d_analysis(
            arr, test_output,
            suffix="_test",
            pool_size=2,
            enable_diff=True,
            tile_size=256,  # 小块快速测试
        )
    else:
        print("data/ 目录下没有视频文件，跳过测试")
