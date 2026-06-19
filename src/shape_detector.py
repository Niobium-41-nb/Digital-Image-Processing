"""
============================================================
形状检测模块 v2 —— 全局运动估计 + 运动残差分析
============================================================

核心思路：
  马赛克视频中背景块统一往一个方向移动，形状内部块往另一个方向移动。

  算法（v2 改进版）：
    1. 自动检测马赛克块大小（多尺度方差分析）
    2. 降采样到块级别
    3. 全局运动估计：尝试所有可能的块级位移，找到使大多数块匹配的方向
       → 这就是背景运动方向
    4. 运动残差：背景运动补偿后，不匹配的块区域就是形状
    5. 多数投票确定形状运动方向（从形状区域内统计）
    6. 后处理：形态学操作 + 最大连通域

  相比 v1 的改进：
    - 不再尝试为每个块单独匹配（歧义太大）
    - 改为"先找全局背景运动 → 再找偏离区域" 的策略
    - 更鲁棒、更快、更简单

依赖：
  - numpy, scipy, opencv-python
============================================================
"""

import os
import sys
from typing import Tuple, Optional

import cv2
import numpy as np
from scipy import ndimage


# ============================================================
# 1. 块大小自动检测
# ============================================================

def detect_block_size(frame: np.ndarray,
                      max_block_size: int = 32,
                      min_block_size: int = 2) -> int:
    """
    自动检测马赛克块大小。

    对每个候选块大小，将图像分割为块，统计"均匀块"的比例
    （块内所有像素值相同）。选择均匀比例高且块尽可能小的候选。

    参数:
        frame: 单帧灰度图，形状 (H, W)
        max_block_size: 最大检测的块大小
        min_block_size: 最小检测的块大小

    返回:
        检测到的块大小（像素）
    """
    H, W = frame.shape

    best_size = 4
    best_score = -1.0

    for bs in range(min_block_size, max_block_size + 1):
        h_blocks = H // bs
        w_blocks = W // bs
        if h_blocks < 4 or w_blocks < 4:
            continue

        # 裁剪到整数倍
        region = frame[:h_blocks * bs, :w_blocks * bs]

        # 重塑为 (h_blocks, bs, w_blocks, bs)
        blocks = region.reshape(h_blocks, bs, w_blocks, bs)

        # 计算每个块内的标准差
        # 对于均匀的马赛克块，标准差应为 0
        block_stds = np.std(blocks.astype(np.float64), axis=(1, 3))

        # 均匀块比例
        uniform = block_stds < 0.5  # 允许微小浮点误差
        uniform_ratio = np.mean(uniform)

        # 评分: 偏爱高均匀比例 + 小块（避免过分割）
        # 均匀比例是主要指标
        score = uniform_ratio

        if score > best_score + 0.01:  # 显著更好才换
            best_score = score
            best_size = bs
        elif abs(score - best_score) < 0.01 and bs < best_size:
            # 分数接近时选更小的块
            best_size = bs
            best_score = score

    print(f"  自动检测块大小: {best_size} px (均匀比例: {best_score:.3f})")
    return best_size


# ============================================================
# 2. 降采样到块级别
# ============================================================

def downsample_to_blocks(video: np.ndarray, block_size: int) -> np.ndarray:
    """
    将视频降采样到块级别。每块取第一个像素（块内均匀）。

    参数:
        video: 形状 (F, H, W)，uint8
        block_size: 块大小

    返回:
        形状 (F, H//block_size, W//block_size)，uint8
    """
    F, H, W = video.shape
    h_blocks = H // block_size
    w_blocks = W // block_size

    ds = video[:, :h_blocks * block_size:block_size,
               :w_blocks * block_size:block_size]
    return ds


# ============================================================
# 3. 全局运动估计
# ============================================================

def estimate_global_motion(ds_video: np.ndarray,
                           max_frames: int = 20) -> Tuple[int, int, float]:
    """
    估计全局（背景）运动方向。

    尝试所有可能的块级位移 (±1 范围，共 9 种)，
    找到使匹配块数量最多的位移方向。

    对于每对相邻帧，将前一帧按位移 d 平移后与后一帧比较，
    统计匹配的块数。在所有帧对上累积，选出全局最佳位移。

    参数:
        ds_video: 块级视频，形状 (F, H, W)，uint8
        max_frames: 最多使用的帧数

    返回:
        (dy, dx, match_ratio) — 背景运动方向和匹配比例
    """
    F, H, W = ds_video.shape
    max_frames = min(max_frames, F - 1)

    # 所有可能的位移
    displacements = [(di, dj) for di in [-1, 0, 1] for dj in [-1, 0, 1]]
    n_disps = len(displacements)

    # 累积每个位移的总匹配数
    total_matches = np.zeros(n_disps, dtype=np.int64)
    total_valid = 0

    frames_used = 0
    for t in range(F - 1):
        if frames_used >= max_frames:
            break

        prev = ds_video[t]
        curr = ds_video[t + 1]

        # 跳过完全相同的帧
        if np.array_equal(prev, curr):
            continue

        frames_used += 1

        for idx, (di, dj) in enumerate(displacements):
            # 平移前一帧
            shifted = np.roll(np.roll(prev, di, axis=0), dj, axis=1)

            # 统计匹配的块（排除边界区域，因为 np.roll 会 wrap）
            match = (shifted == curr)

            # 排除因 wrap-around 产生的虚假匹配
            if di > 0:
                match[:di, :] = False
            elif di < 0:
                match[di:, :] = False
            if dj > 0:
                match[:, :dj] = False
            elif dj < 0:
                match[:, dj:] = False

            total_matches[idx] += np.sum(match)

        total_valid += H * W

    if total_valid == 0:
        return 0, 0, 0.0

    # 找最佳位移
    best_idx = np.argmax(total_matches)
    best_di, best_dj = displacements[best_idx]
    match_ratio = total_matches[best_idx] / (total_valid * frames_used
                                              if frames_used > 0 else 1)

    print(f"  全局运动估计: ({best_di}, {best_dj}), "
          f"匹配率: {match_ratio:.3f} "
          f"(使用 {frames_used} 帧对)")

    return best_di, best_dj, match_ratio


def estimate_secondary_motion(ds_video: np.ndarray,
                              primary_di: int, primary_dj: int,
                              candidate_mask: np.ndarray = None,
                              max_frames: int = 20) -> Tuple[int, int, float]:
    """
    在排除主要运动区域后，估计第二运动方向（形状内部运动）。

    参数:
        ds_video: 块级视频
        primary_di, primary_dj: 主要运动方向（背景）
        candidate_mask: 形状候选区域的掩码（True=可能是形状）。
                       如果为 None，则用"与背景运动不匹配"的区域作为候选。
        max_frames: 最多使用的帧数

    返回:
        (dy, dx, match_ratio) — 第二运动方向
    """
    F, H, W = ds_video.shape
    max_frames = min(max_frames, F - 1)

    displacements = [(di, dj) for di in [-1, 0, 1] for dj in [-1, 0, 1]]
    n_disps = len(displacements)

    total_matches = np.zeros(n_disps, dtype=np.int64)
    total_valid = 0
    frames_used = 0

    for t in range(F - 1):
        if frames_used >= max_frames:
            break

        prev = ds_video[t]
        curr = ds_video[t + 1]

        if np.array_equal(prev, curr):
            continue

        frames_used += 1

        # 如果提供了候选掩码，则只在候选区域内统计
        for idx, (di, dj) in enumerate(displacements):
            shifted = np.roll(np.roll(prev, di, axis=0), dj, axis=1)
            match = (shifted == curr)

            # 排除 wrap-around
            if di > 0:
                match[:di, :] = False
            elif di < 0:
                match[di:, :] = False
            if dj > 0:
                match[:, :dj] = False
            elif dj < 0:
                match[:, dj:] = False

            # 只统计候选区域
            if candidate_mask is not None:
                match = match & candidate_mask

            total_matches[idx] += np.sum(match)

        if candidate_mask is not None:
            total_valid += np.sum(candidate_mask)
        else:
            total_valid += H * W

    if total_valid == 0:
        return 0, 0, 0.0

    best_idx = np.argmax(total_matches)
    best_di, best_dj = displacements[best_idx]
    # 排除与主要运动相同的位移（选第二好的）
    if best_di == primary_di and best_dj == primary_dj:
        sorted_indices = np.argsort(total_matches)[::-1]
        for idx in sorted_indices:
            di, dj = displacements[idx]
            if di != primary_di or dj != primary_dj:
                best_di, best_dj = di, dj
                best_idx = idx
                break

    match_ratio = total_matches[best_idx] / (total_valid * max(1, frames_used))

    print(f"  第二运动估计: ({best_di}, {best_dj}), "
          f"匹配率: {match_ratio:.3f}")

    return best_di, best_dj, match_ratio


# ============================================================
# 4. 运动残差掩码 — 核心：找出偏离背景运动的区域
# ============================================================

def compute_motion_residual_mask(ds_video: np.ndarray,
                                 bg_di: int, bg_dj: int,
                                 consistency_threshold: float = 0.4,
                                 max_frames: int = 30) -> np.ndarray:
    """
    通过背景运动补偿残差，找出运动不一致的块（即形状内部）。

    对每对相邻帧：
      1. 将前一帧按背景运动方向平移
      2. 平移后的帧与后一帧比较：不匹配的块 = 残差
      3. 累积所有帧对的残差

    累积残差高的区域 → 运动方向与背景不同 → 形状内部。

    参数:
        ds_video: 块级视频，形状 (F, H, W)
        bg_di, bg_dj: 背景运动方向
        consistency_threshold: 判定为"与背景一致"的匹配率阈值
        max_frames: 最多使用的帧数

    返回:
        二值掩码 (H, W)，True=形状候选区域
    """
    F, H, W = ds_video.shape
    max_frames = min(max_frames, F - 1)

    # 累积不匹配计数
    mismatch_count = np.zeros((H, W), dtype=np.float32)
    frame_count = 0

    for t in range(F - 1):
        if frame_count >= max_frames:
            break

        prev = ds_video[t]
        curr = ds_video[t + 1]

        if np.array_equal(prev, curr):
            continue

        frame_count += 1

        # 按背景方向平移
        shifted = np.roll(np.roll(prev, bg_di, axis=0), bg_dj, axis=1)

        # 不匹配的块
        mismatch = (shifted != curr).astype(np.float32)

        # 排除 wrap-around 导致的虚假不匹配
        if bg_di > 0:
            mismatch[:bg_di, :] = 0
        elif bg_di < 0:
            mismatch[bg_di:, :] = 0
        if bg_dj > 0:
            mismatch[:, :bg_dj] = 0
        elif bg_dj < 0:
            mismatch[:, bg_dj:] = 0

        mismatch_count += mismatch

    if frame_count == 0:
        return np.zeros((H, W), dtype=bool)

    # 归一化为不匹配率 [0, 1]
    mismatch_rate = mismatch_count / frame_count

    # 阈值分割
    candidate_mask = mismatch_rate > consistency_threshold

    print(f"  运动残差分析: {np.sum(candidate_mask)}/{H*W} 块候选 "
          f"({100*np.sum(candidate_mask)/(H*W):.1f}%), "
          f"使用 {frame_count} 帧对")

    return candidate_mask


def refine_shape_mask(candidate_mask: np.ndarray,
                      ds_video: np.ndarray,
                      bg_di: int, bg_dj: int,
                      max_frames: int = 30) -> np.ndarray:
    """
    精炼形状掩码：通过在两候选运动方向之间做二选一来确定每个候选块。

    对每个候选块，比较它在背景方向和形状方向下的匹配情况，
    确定它更接近哪种运动。

    参数:
        candidate_mask: 初始候选掩码 (H, W)，True=候选
        ds_video: 块级视频
        bg_di, bg_dj: 背景运动方向
        max_frames: 最多使用的帧数

    返回:
        修正后的 (H, W) 二值掩码，True=形状
    """
    H, W = candidate_mask.shape
    if not np.any(candidate_mask):
        return np.zeros((H, W), dtype=bool)

    # 在候选区域内估计第二运动方向
    shape_di, shape_dj, _ = estimate_secondary_motion(
        ds_video, bg_di, bg_dj,
        candidate_mask=candidate_mask,
        max_frames=max_frames,
    )

    # 如果第二运动方向和背景差不多，说明没有真正的第二运动
    if shape_di == bg_di and shape_dj == bg_dj:
        print("  未检测到明显的第二运动方向，可能只有一种运动")
        return np.zeros((H, W), dtype=bool)

    F = ds_video.shape[0]
    max_frames = min(max_frames, F - 1)

    # 累积两种方向下的匹配得分
    bg_score = np.zeros((H, W), dtype=np.float32)
    shape_score = np.zeros((H, W), dtype=np.float32)
    frame_count = 0

    for t in range(F - 1):
        if frame_count >= max_frames:
            break

        prev = ds_video[t]
        curr = ds_video[t + 1]

        if np.array_equal(prev, curr):
            continue

        frame_count += 1

        # 背景方向匹配
        shifted_bg = np.roll(np.roll(prev, bg_di, axis=0), bg_dj, axis=1)
        match_bg = (shifted_bg == curr).astype(np.float32)

        # 形状方向匹配
        shifted_shape = np.roll(np.roll(prev, shape_di, axis=0), shape_dj, axis=1)
        match_shape = (shifted_shape == curr).astype(np.float32)

        # 排除 wrap-around
        for di, dj, arr in [(bg_di, bg_dj, match_bg),
                             (shape_di, shape_dj, match_shape)]:
            if di > 0:
                arr[:di, :] = 0
            elif di < 0:
                arr[di:, :] = 0
            if dj > 0:
                arr[:, :dj] = 0
            elif dj < 0:
                arr[:, dj:] = 0

        bg_score += match_bg
        shape_score += match_shape

    if frame_count == 0:
        return candidate_mask

    # 归一化
    bg_score /= frame_count
    shape_score /= frame_count

    # 二选一：形状方向匹配更好 → 形状块
    refined = (shape_score > bg_score) & candidate_mask

    # 再加一个宽松条件：只要形状方向匹配率明显高于背景方向
    refined |= ((shape_score > 0.5) & (shape_score > bg_score + 0.1))

    print(f"  精炼后形状掩码: {np.sum(refined)} 块 "
          f"({100*np.sum(refined)/(H*W):.1f}%)")

    return refined


# ============================================================
# 5. 像素级掩码提取与后处理
# ============================================================

def blocks_to_pixel_mask(block_mask: np.ndarray,
                         block_size: int,
                         frame_shape: Tuple[int, int]) -> np.ndarray:
    """
    将块级二值掩码上采样到像素级。

    参数:
        block_mask: 块级掩码 (h_blocks, w_blocks)，bool
        block_size: 块大小
        frame_shape: 原始帧尺寸 (H, W)

    返回:
        像素级 uint8 掩码 (H, W)，255=形状，0=背景
    """
    H, W = frame_shape
    pixel_mask = np.repeat(np.repeat(
        block_mask.astype(np.uint8) * 255,
        block_size, axis=0), block_size, axis=1)

    # 裁剪到原始尺寸
    return pixel_mask[:H, :W]


def postprocess_mask(pixel_mask: np.ndarray,
                     min_area_ratio: float = 0.005) -> np.ndarray:
    """
    形态学后处理：闭运算补孔 → 开运算去噪 → 提取最大连通域。

    参数:
        pixel_mask: 像素级掩码 (H, W)，uint8
        min_area_ratio: 最小面积占比

    返回:
        后处理后的掩码 (H, W)，uint8
    """
    H, W = pixel_mask.shape
    min_area = H * W * min_area_ratio

    # 形态学核大小
    ksize = max(2, min(H, W) // 200)

    # 闭运算：填补内部小孔
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (ksize * 3, ksize * 3))
    pixel_mask = cv2.morphologyEx(pixel_mask, cv2.MORPH_CLOSE, kernel_close)

    # 开运算：移除孤立噪点
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (ksize * 2, ksize * 2))
    pixel_mask = cv2.morphologyEx(pixel_mask, cv2.MORPH_OPEN, kernel_open)

    # 最大连通域
    num_labels, connected, stats, _ = cv2.connectedComponentsWithStats(
        pixel_mask, connectivity=8)

    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        if stats[largest_label, cv2.CC_STAT_AREA] >= min_area:
            pixel_mask = (connected == largest_label).astype(np.uint8) * 255
        else:
            pixel_mask[:] = 0

    return pixel_mask


# ============================================================
# 6. 可视化
# ============================================================

def create_overlay_frame(frame: np.ndarray, mask: np.ndarray,
                         color: Tuple[int, int, int] = (0, 255, 0),
                         alpha: float = 0.4) -> np.ndarray:
    """将形状掩码叠加到原帧上，绘制绿色轮廓。"""
    if frame.ndim == 2:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        frame_bgr = frame.copy()

    overlay = frame_bgr.copy()
    overlay[mask > 0] = color
    result = cv2.addWeighted(frame_bgr, 1 - alpha, overlay, alpha, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    return result


def create_direction_map(block_labels: np.ndarray,
                         block_size: int,
                         frame_shape: Tuple[int, int]) -> np.ndarray:
    """
    创建运动方向彩色图：背景=蓝，形状=红。
    """
    H, W = frame_shape
    labels_pixel = np.repeat(np.repeat(
        block_labels.astype(np.uint8), block_size, axis=0),
        block_size, axis=1)[:H, :W]

    result = np.zeros((H, W, 3), dtype=np.uint8)
    result[labels_pixel == 0] = (255, 0, 0)    # 背景 = 蓝色
    result[labels_pixel == 1] = (0, 0, 255)    # 形状 = 红色
    return result


def get_contours(mask: np.ndarray) -> list:
    """从掩码提取轮廓。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    return contours


# ============================================================
# 8. 检测过程可视化 — 保存每一步的中间结果
# ============================================================

def save_process_visualization(ds_video: np.ndarray,
                               original_frame: np.ndarray,
                               block_size: int,
                               bg_di: int, bg_dj: int,
                               candidate_mask: np.ndarray,
                               refined_mask: np.ndarray,
                               pixel_mask: np.ndarray,
                               output_dir: str) -> None:
    """
    保存检测过程中每一步的可视化，帮助理解算法流程。

    生成 6 张图：
      00_original.png        — 原始帧
      01_block_level.png     — 块级降采样
      02_motion_residual.png — 运动残差（偏离背景运动的区域）
      03_refined_blocks.png  — 二选一精炼后的块级掩码
      04_pixel_mask.png      — 上采样后的像素级掩码
      05_final_overlay.png   — 最终叠加结果

    参数:
        ds_video: 块级视频，形状 (F, h, w)
        original_frame: 原始单帧 (H, W)
        block_size: 块大小
        bg_di, bg_dj: 背景运动方向
        candidate_mask: 运动残差候选掩码 (h, w)
        refined_mask: 精炼后块级掩码 (h, w)
        pixel_mask: 像素级掩码 (H, W)
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    H, W = original_frame.shape
    h, w = ds_video.shape[1], ds_video.shape[2]

    # --- 00: 原始帧 ---
    cv2.imwrite(os.path.join(output_dir, '00_original.png'), original_frame)

    # --- 01: 块级降采样（放大到可见大小）---
    ds_frame = ds_video[0]
    ds_display = np.repeat(np.repeat(ds_frame, 4, axis=0), 4, axis=1)
    cv2.imwrite(os.path.join(output_dir, '01_block_level.png'), ds_display)

    # --- 02: 运动残差热力图 ---
    # candidate_mask 上采样到像素级
    residual_pixel = np.repeat(np.repeat(
        candidate_mask.astype(np.uint8) * 255,
        block_size, axis=0), block_size, axis=1)[:H, :W]

    # 转成彩色：红色=残差高（可能是形状），蓝色=残差低（背景）
    residual_color = np.zeros((H, W, 3), dtype=np.uint8)
    residual_color[:, :, 2] = residual_pixel   # R 通道 = 残差
    residual_color[:, :, 0] = 255 - residual_pixel  # B 通道 = 背景

    # 用箭头标注背景方向
    arrow_color = (0, 255, 255)  # 黄色
    _draw_direction_arrow(residual_color, bg_di, bg_dj,
                          block_size, (H, W), arrow_color)

    cv2.imwrite(os.path.join(output_dir, '02_motion_residual.png'), residual_color)

    # --- 03: 精炼后的块级掩码 ---
    refined_pixel = np.repeat(np.repeat(
        refined_mask.astype(np.uint8) * 255,
        block_size, axis=0), block_size, axis=1)[:H, :W]

    refined_color = np.zeros((H, W, 3), dtype=np.uint8)
    refined_color[:, :, 2] = refined_pixel  # 红色 = 形状

    cv2.imwrite(os.path.join(output_dir, '03_refined_blocks.png'), refined_color)

    # --- 04: 像素级掩码（后处理前）---
    cv2.imwrite(os.path.join(output_dir, '04_pixel_mask_raw.png'), pixel_mask)

    # --- 05: 最终叠加 ---
    final = create_overlay_frame(original_frame, pixel_mask)
    cv2.imwrite(os.path.join(output_dir, '05_final_overlay.png'), final)

    print(f"  过程可视化已保存: {output_dir}/")


def _draw_direction_arrow(img: np.ndarray, di: int, dj: int,
                          block_size: int, frame_shape: tuple,
                          color: tuple) -> None:
    """在图像上绘制运动方向箭头。"""
    H, W = frame_shape
    # 箭头起点（右上角）
    start_x = W - 120
    start_y = 60
    # 箭头方向
    arrow_len = 40
    end_x = start_x + dj * arrow_len
    end_y = start_y + di * arrow_len

    cv2.arrowedLine(img, (start_x, start_y), (end_x, end_y),
                    color, 3, tipLength=0.3)

    # 标注文字
    dir_names = {
        (0, 1): 'Right', (0, -1): 'Left',
        (1, 0): 'Down', (-1, 0): 'Up',
        (1, 1): 'Down-Right', (1, -1): 'Down-Left',
        (-1, 1): 'Up-Right', (-1, -1): 'Up-Left',
        (0, 0): 'Static',
    }
    name = dir_names.get((di, dj), f'({di},{dj})')
    cv2.putText(img, f'BG: {name}', (start_x - 80, start_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ============================================================
# 9. 形状 → 文字描述
# ============================================================

def describe_shape(mask: np.ndarray, block_size: int = None) -> str:
    """
    将检测到的形状掩码转化为文字描述。

    分析内容包括：
      - 形状类型（正方形/矩形/圆形/多边形/星形/不规则）
      - 位置（中心坐标、是否在画面中心）
      - 大小（像素和相对比例）
      - 几何属性（顶点数、周长、面积、圆度、宽高比）

    参数:
        mask: 像素级二值掩码 (H, W)，255=形状
        block_size: 块大小（用于换算马赛克块数），可选

    返回:
        多行文字描述
    """
    H, W = mask.shape
    total_px = H * W
    shape_px = int(np.sum(mask > 0))
    shape_pct = 100.0 * shape_px / total_px

    if shape_px == 0:
        return ("未检测到形状。\n\n"
                "可能原因：\n"
                "  1. 画面中所有马赛克都朝同一个方向移动（没有\"形状区域\"）\n"
                "  2. 背景和形状的移动方向相同（无法区分）\n"
                "  3. 视频中的形状是移动的而非固定区域内的纹理滚动\n\n"
                "建议：\n"
                "  - 确认背景和形状内部确实朝不同方向移动\n"
                "  - 尝试用\"生成视频\"功能做一个测试视频来验证")

    # 轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Shape mask exists but no contours could be extracted."

    # 取最大轮廓
    cnt = max(contours, key=cv2.contourArea)

    # --- 基础几何 ---
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        cx, cy = 0, 0
    else:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

    x, y, bw, bh = cv2.boundingRect(cnt)
    bbox_area = bw * bh
    extent = area / bbox_area if bbox_area > 0 else 0

    # 最小外接矩形（可旋转）
    rect = cv2.minAreaRect(cnt)
    (rx, ry), (rw, rh), angle = rect

    # 最小外接圆
    (ccx, ccy), radius = cv2.minEnclosingCircle(cnt)

    # --- 形状分类特征 ---
    # 1. 圆度: 4*pi*area / perimeter^2 （圆=1，直线≈0）
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

    # 2. 宽高比
    aspect_ratio = bw / bh if bh > 0 else 0

    # 3. 顶点数（轮廓近似）
    epsilon = 0.02 * perimeter  # 2% 的周长作为近似精度
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    num_vertices = len(approx)

    # 用更宽松的近似判断主顶点
    epsilon_coarse = 0.04 * perimeter
    approx_coarse = cv2.approxPolyDP(cnt, epsilon_coarse, True)

    # 4. 凸性
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0

    # 5. 实心度
    solidity = area / bbox_area if bbox_area > 0 else 0

    # --- 形状类型判断 ---
    shape_type = _classify_shape(num_vertices, circularity, convexity,
                                  aspect_ratio, solidity, len(approx_coarse))

    # --- 字母/数字识别 ---
    char_matches = recognize_character(mask)

    # --- 位置描述 ---
    frame_cx, frame_cy = W / 2, H / 2
    offset_x = cx - frame_cx
    offset_y = cy - frame_cy
    offset_pct_x = 100.0 * offset_x / W
    offset_pct_y = 100.0 * offset_y / H

    if abs(offset_pct_x) < 5 and abs(offset_pct_y) < 5:
        position_desc = "画面正中央"
    else:
        h_pos = "中央" if abs(offset_pct_x) < 10 else ("左侧" if offset_x < 0 else "右侧")
        v_pos = "中央" if abs(offset_pct_y) < 10 else ("上方" if offset_y < 0 else "下方")
        position_desc = f"画面{v_pos}{h_pos}，偏离中心 ({offset_pct_x:+.1f}%, {offset_pct_y:+.1f}%)"

    # --- 形状类型中文名 ---
    SHAPE_CN = {
        'Square': '正方形', 'Square-like shape': '近似正方形',
        'Rectangle': '长方形', 'Rectangular shape': '近似长方形',
        'Circle': '圆形', 'Ellipse': '椭圆形', 'Near-circle (rounded shape)': '近圆形',
        'Triangle': '三角形', 'Star': '星形', 'Star-like shape': '近似星形',
        'Pentagon': '五边形', 'Hexagon': '六边形', 'Heptagon': '七边形', 'Octagon': '八边形',
        'Quadrilateral': '四边形', 'Concave / star-shaped': '凹形/星形',
        'Convex shape': '凸多边形', 'Irregular concave shape': '不规则凹形',
    }
    shape_cn = SHAPE_CN.get(shape_type, shape_type)

    # --- 构建中文报告 ---
    lines = []
    lines.append("=" * 50)
    lines.append("        马赛克视频形状识别报告")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"  【形状类型】 {shape_cn}")
    if char_matches:
        top_char, top_score = char_matches[0]
        # 检查是否与第二名接近（歧义情况）
        ambiguous = (len(char_matches) >= 2 and
                     char_matches[1][1] >= top_score - 0.03)
        if top_score > 0.6:
            if ambiguous:
                candidates = ', '.join(f'{c}({s:.0%})' for c, s in char_matches[:3])
                lines.append(f"  【字母识别】 {top_char}  (置信度 {top_score:.0%})")
                lines.append(f"  【其他可能】 {candidates}")
            else:
                lines.append(f"  【字母识别】 {top_char}  (置信度 {top_score:.0%})")
        elif top_score > 0.35:
            candidates = ', '.join(f'{c}({s:.0%})' for c, s in char_matches[:3])
            lines.append(f"  【可能字母】 {candidates}")
    lines.append(f"  【位    置】 {position_desc}")
    lines.append(f"  【中心坐标】 ({cx:.0f}, {cy:.0f})   (画面中心是 {W/2:.0f}, {H/2:.0f})")
    lines.append("")
    lines.append("  —— 尺寸信息 ——")
    lines.append(f"  形状面积: {area:.0f} 像素  (占画面 {shape_pct:.1f}%)")
    lines.append(f"  外接矩形: {bw:.0f} × {bh:.0f} 像素")
    lines.append(f"  宽 高 比: {aspect_ratio:.2f}  (1.0=正方形, >1=扁宽, <1=瘦高)")
    if block_size:
        lines.append(f"  马赛克块数: 约 {bw//block_size} × {bh//block_size} 块  (块大小 {block_size}px)")
    lines.append("")
    lines.append("  —— 几何特征 ——")
    lines.append(f"  顶点数量: {num_vertices} 个")
    lines.append(f"  周    长: {perimeter:.0f} 像素")
    lines.append(f"  接 近 圆: {circularity:.1%}  (100%=完美圆形, 正方形约 78%)")
    lines.append(f"  凸 程 度: {convexity:.1%}  (100%=完全凸形, 星形<90%)")
    lines.append(f"  饱 满 度: {solidity:.1%}  (100%=填满外接矩形)")
    if block_size:
        lines.append(f"  内部包含约 {int(shape_px // (block_size**2))} 个马赛克方块")
    lines.append("")
    lines.append("  —— 判断依据 ——")
    lines.append(f"  精细顶点: {num_vertices}  粗略顶点: {len(approx_coarse)}")
    if num_vertices == 4 and 0.90 < aspect_ratio < 1.11:
        lines.append(f"  4个顶点 + 宽高比≈1 → 判定为正方形")
    elif num_vertices == 4:
        lines.append(f"  4个顶点 + 宽高比={aspect_ratio:.2f} → 判定为长方形")
    elif circularity > 0.85:
        lines.append(f"  圆度={circularity:.1%} 接近 100% → 判定为圆形")
    elif convexity < 0.85:
        lines.append(f"  凹形 + {len(approx_coarse)}个顶点 → 判定为星形/凹多边形")
    if 0.6 < circularity < 0.95 and convexity > 0.9:
        sides = _estimate_polygon_sides(num_vertices, len(approx_coarse),
                                         aspect_ratio)
        if sides:
            lines.append(f"  正多边形特征 → 估计 {sides} 条边")
    lines.append("=" * 50)

    return "\n".join(lines)


def _classify_shape(num_vertices: int, circularity: float, convexity: float,
                    aspect_ratio: float, solidity: float,
                    coarse_vertices: int) -> str:
    """根据几何特征分类形状类型。"""
    # 圆形/椭圆
    if circularity > 0.85:
        if 0.9 < aspect_ratio < 1.1:
            return "Circle"
        else:
            return "Ellipse"
    if circularity > 0.75 and convexity > 0.95:
        if 0.9 < aspect_ratio < 1.1:
            return "Near-circle (rounded shape)"

    # 正方形
    if 0.9 < aspect_ratio < 1.1 and 0.9 < solidity < 1.1 and convexity > 0.95:
        if 3 <= coarse_vertices <= 5:
            return "Square"
        return "Square-like shape"

    # 矩形
    if (aspect_ratio < 0.9 or aspect_ratio > 1.1) and solidity > 0.9 and convexity > 0.95:
        if 3 <= coarse_vertices <= 5:
            return f"Rectangle (aspect {aspect_ratio:.2f})"
        return "Rectangular shape"

    # 三角形
    if 2 <= coarse_vertices <= 3 and convexity > 0.9:
        return "Triangle"

    # 星形（凸性低、顶点多）
    if convexity < 0.85 and circularity < 0.6:
        pts = coarse_vertices
        if pts >= 8:
            return f"Star ({pts//2}-pointed)"
        elif pts >= 5:
            return f"Star-like shape ({coarse_vertices} vertices)"
        return "Concave / star-shaped"

    # 正多边形
    if convexity > 0.9 and circularity > 0.6:
        sides = coarse_vertices
        names = {3: 'Triangle', 4: 'Quadrilateral', 5: 'Pentagon',
                 6: 'Hexagon', 7: 'Heptagon', 8: 'Octagon'}
        if sides in names:
            return names[sides]
        if sides > 8:
            return f"Regular polygon ({sides} sides)"
        return f"Polygon ({sides} vertices)"

    # 兜底
    if convexity > 0.9:
        return f"Convex shape ({coarse_vertices} vertices)"
    else:
        return f"Irregular concave shape ({coarse_vertices} vertices)"


def _estimate_polygon_sides(fine_v: int, coarse_v: int,
                            aspect_ratio: float) -> str | None:
    """估计正多边形的边数。"""
    sides = coarse_v if coarse_v >= 3 else fine_v
    if 3 <= sides <= 10:
        names = {3: '3 (triangle)', 4: '4 (square/rectangle)',
                 5: '5 (pentagon)', 6: '6 (hexagon)',
                 7: '7 (heptagon)', 8: '8 (octagon)',
                 9: '9 (nonagon)', 10: '10 (decagon)'}
        return names.get(sides, str(sides))
    return None


# ============================================================
# 10. 字母/数字 OCR — 裁剪归一化 + 多尺度滑动匹配
# ============================================================

# 缓存：字符 → 裁剪归一化后的参考图像
_TEMPLATE_NORM_CACHE = {}  # char -> (H,W) uint8 binary


def _render_char_norm(char: str, target_size: int = 100) -> np.ndarray:
    """渲染一个字符并裁剪+归一化到 target_size×target_size。"""
    if char in _TEMPLATE_NORM_CACHE:
        return _TEMPLATE_NORM_CACHE[char]

    import cv2
    # 在较大画布上渲染，确保字符完整
    canvas = np.zeros((400, 400), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 3.5
    thickness = 16
    (tw, th), _ = cv2.getTextSize(char, font, font_scale, thickness)
    cx, cy = (400 - tw) // 2, (400 + th) // 2
    cv2.putText(canvas, char, (cx, cy), font, font_scale, 255, thickness)

    # 裁剪到内容边界（加小 margin）
    ys, xs = np.where(canvas > 0)
    if len(ys) < 10:
        _TEMPLATE_NORM_CACHE[char] = np.zeros((target_size, target_size), dtype=np.uint8)
        return _TEMPLATE_NORM_CACHE[char]

    margin = 4
    y1, y2 = max(0, ys.min()-margin), min(400, ys.max()+margin+1)
    x1, x2 = max(0, xs.min()-margin), min(400, xs.max()+margin+1)
    cropped = canvas[y1:y2, x1:x2]

    # 保持宽高比缩放到 target_size
    h, w = cropped.shape
    scale = (target_size - 8) / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    if new_h < 4 or new_w < 4:
        _TEMPLATE_NORM_CACHE[char] = np.zeros((target_size, target_size), dtype=np.uint8)
        return _TEMPLATE_NORM_CACHE[char]

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 放到 target_size 画布中央
    result = np.zeros((target_size, target_size), dtype=np.uint8)
    dy, dx = (target_size - new_h) // 2, (target_size - new_w) // 2
    result[dy:dy+new_h, dx:dx+new_w] = resized

    _TEMPLATE_NORM_CACHE[char] = result
    return result


def _crop_and_normalize(mask: np.ndarray, target_size: int = 100) -> np.ndarray:
    """将检测到的掩码裁剪到内容边界并归一化到 target_size。"""
    import cv2
    ys, xs = np.where(mask > 0)
    if len(ys) < 10:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    margin = 6
    y1, y2 = max(0, ys.min()-margin), min(mask.shape[0], ys.max()+margin+1)
    x1, x2 = max(0, xs.min()-margin), min(mask.shape[1], xs.max()+margin+1)
    cropped = (mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255

    # 保持宽高比缩放
    h, w = cropped.shape
    scale = (target_size - 8) / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    if new_h < 4 or new_w < 4:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    result = np.zeros((target_size, target_size), dtype=np.uint8)
    dy, dx = (target_size - new_h) // 2, (target_size - new_w) // 2
    result[dy:dy+new_h, dx:dx+new_w] = resized
    return result


def recognize_character(mask: np.ndarray) -> list:
    """
    OCR 识别检测到的形状是哪个字母/数字。

    方法：裁剪 + 归一化到统一尺寸 → 多尺度滑动模板匹配。
    对位置、大小不敏感，能准确识别马赛克化的字母。

    参数:
        mask: 像素级二值掩码 (H, W), uint8, 255=前景

    返回:
        [(字符, 综合得分), ...], 按分数降序, 最多 5 个
    """
    H, W = mask.shape
    detected = mask > 0
    detected_area = np.sum(detected)
    if detected_area < 100:
        return []

    # 归一化检测到的形状
    norm_det = _crop_and_normalize(mask, target_size=100)
    if np.sum(norm_det) < 20:
        return []

    # 待匹配的字符池
    chars = (list('ABCDEFGHJKLMNPQRSTUVWXYZ') +
             list('23456789'))

    scores = []
    for char in chars:
        ref = _render_char_norm(char, target_size=100)
        ref_area = np.sum(ref)
        if ref_area < 20:
            continue

        # 多尺度 + 滑动窗口匹配
        best_score = 0.0
        # 尝试不同尺度（±15%）
        for scale in [0.85, 0.92, 1.0, 1.08, 1.15]:
            s = int(100 * scale)
            if s < 20:
                continue
            ref_scaled = cv2.resize(ref, (s, s), interpolation=cv2.INTER_NEAREST) if scale != 1.0 else ref
            det_scaled = cv2.resize(norm_det, (s, s), interpolation=cv2.INTER_NEAREST) if scale != 1.0 else norm_det

            # 基本 IoU（居中对齐）
            det_bin = det_scaled > 127
            ref_bin = ref_scaled > 127
            intersection = np.sum(det_bin & ref_bin)
            union = np.sum(det_bin | ref_bin)
            iou = intersection / union if union > 0 else 0

            # 小幅度滑动（±3px），取最佳
            for dy in [-3, 0, 3]:
                for dx in [-3, 0, 3]:
                    if dy == 0 and dx == 0:
                        continue
                    det_shifted = np.roll(np.roll(det_bin, dy, axis=0), dx, axis=1)
                    # 边界清零
                    if dy > 0:
                        det_shifted[:dy, :] = False
                    elif dy < 0:
                        det_shifted[dy:, :] = False
                    if dx > 0:
                        det_shifted[:, :dx] = False
                    elif dx < 0:
                        det_shifted[:, dx:] = False
                    inter = np.sum(det_shifted & ref_bin)
                    un = np.sum(det_shifted | ref_bin)
                    iou_s = inter / un if un > 0 else 0
                    if iou_s > iou:
                        iou = iou_s

            if iou > best_score:
                best_score = iou

        if best_score > 0.2:
            scores.append((char, round(best_score, 3)))

    scores.sort(key=lambda x: x[1], reverse=True)

    # 做一次"与次优的差距"检查来提升置信度
    if len(scores) >= 2 and scores[0][1] < scores[1][1] * 1.15:
        # 前两名太接近，标记为不太确定
        pass  # 仍然返回，但调用方可看分数差距

    return scores[:5]


# ============================================================
# 7. 主入口
# ============================================================

def detect_shape(video_path: str,
                 block_size: Optional[int] = None,
                 output_dir: Optional[str] = None,
                 visualize: bool = True) -> dict:
    """
    从马赛克视频中检测形状 —— 端到端入口。

    流程:
      读取视频 → 检测块大小 → 降采样 → 全局运动估计
      → 运动残差掩码 → 精炼 → 后处理 → 可视化

    参数:
        video_path: 输入视频 .mp4 路径
        block_size: 手动指定块大小（None=自动检测）
        output_dir: 输出目录
        visualize: 是否生成可视化

    返回:
        dict: {
            'mask': 像素级形状掩码 (H, W) uint8,
            'block_labels': 块级标签 (h, w) bool,
            'block_size': int,
            'bg_direction': (dy, dx),
            'shape_direction': (dy, dx),
            'contours': list,
            'overlay_frame': BGR 叠加图,
            'direction_map': BGR 方向图,
        }
    """
    # 避免循环导入
    from src.video_converter import mp4_to_grayscale_array

    print(f"\n{'='*60}")
    print(f"形状检测: {os.path.basename(video_path)}")
    print(f"{'='*60}")

    # ── 读取 ──
    print(">>> 读取视频")
    video = mp4_to_grayscale_array(video_path, verbose=False)
    F, H, W = video.shape
    print(f"  视频: {F} 帧, {H}×{W}")

    # ── 块大小 ──
    if block_size is None:
        print(">>> 自动检测块大小")
        block_size = detect_block_size(video[0])
    else:
        print(f">>> 使用指定块大小: {block_size} px")

    # ── 降采样 ──
    print(f">>> 降采样到块级别")
    ds_video = downsample_to_blocks(video, block_size)
    ds_h, ds_w = ds_video.shape[1], ds_video.shape[2]
    print(f"  块级尺寸: {ds_h}×{ds_w}")

    # ── 全局运动估计 ──
    print(">>> 全局运动估计（找背景方向）")
    bg_di, bg_dj, bg_match = estimate_global_motion(ds_video)

    if bg_match < 0.1:
        print("  警告: 全局运动匹配率很低，可能背景静止或无规律运动")

    # ── 运动残差掩码 ──
    print(">>> 运动残差分析（找偏离背景的区域）")
    candidate_mask = compute_motion_residual_mask(ds_video, bg_di, bg_dj)

    # ── 精炼 ──
    print(">>> 精炼形状掩码")
    shape_block_mask = refine_shape_mask(
        candidate_mask, ds_video, bg_di, bg_dj)

    # 提取第二运动方向
    if np.any(shape_block_mask):
        shape_di, shape_dj, _ = estimate_secondary_motion(
            ds_video, bg_di, bg_dj,
            candidate_mask=shape_block_mask,
        )
    else:
        shape_di, shape_dj = 0, 0

    # ── 上采样到像素级 ──
    print(">>> 后处理")
    pixel_mask = blocks_to_pixel_mask(shape_block_mask, block_size, (H, W))
    pixel_mask = postprocess_mask(pixel_mask)

    shape_area = np.sum(pixel_mask > 0)
    print(f"  形状面积: {shape_area} 像素 ({100*shape_area/(H*W):.1f}%)")

    # ── 块级标签用于可视化 ──
    block_labels = shape_block_mask.astype(np.uint8)

    # ── 轮廓 ──
    contours = get_contours(pixel_mask)
    print(f"  轮廓数: {len(contours)}")

    # ── 可视化 ──
    result = {
        'mask': pixel_mask,
        'block_labels': block_labels,
        'block_size': block_size,
        'bg_direction': (float(bg_di), float(bg_dj)),
        'shape_direction': (float(shape_di), float(shape_dj)),
        'contours': contours,
    }

    if visualize:
        print(">>> 生成可视化")
        result['overlay_frame'] = create_overlay_frame(video[0], pixel_mask)
        result['direction_map'] = create_direction_map(
            block_labels, block_size, (H, W))

    # ── 保存 ──
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # 保存掩码
        cv2.imwrite(os.path.join(output_dir, 'shape_mask.png'), pixel_mask)

        # 保存过程可视化（6 张步骤图）
        if visualize:
            print(">>> 保存检测过程可视化")
            save_process_visualization(
                ds_video, video[0], block_size, bg_di, bg_dj,
                candidate_mask, shape_block_mask, pixel_mask, output_dir)

            cv2.imwrite(os.path.join(output_dir, 'overlay.png'),
                        result['overlay_frame'])
            cv2.imwrite(os.path.join(output_dir, 'direction_map.png'),
                        result['direction_map'])
            print(f"  可视化已保存")

        # 生成形状文字描述
        print(">>> 生成形状文字描述")
        description = describe_shape(pixel_mask, block_size)
        desc_path = os.path.join(output_dir, 'shape_description.txt')
        with open(desc_path, 'w', encoding='utf-8') as f:
            f.write(description)
        print(f"  文字描述已保存: {desc_path}")
        # 也打印到控制台
        print()
        print(description)

        # 掩码视频（前 30 帧）
        from src.video_converter import gray_array_to_mp4
        n_disp = min(F, 30)
        mask_video = np.tile(pixel_mask[np.newaxis, :, :], (n_disp, 1, 1))
        gray_array_to_mp4(mask_video,
                          os.path.join(output_dir, 'shape_mask_video.mp4'),
                          fps=10, verbose=False)

    print(f"{'='*60}")
    print("形状检测完成!")
    print(f"{'='*60}\n")

    return result


# ============================================================
# 模块自测
# ============================================================

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    test_video = os.path.join(project_root, 'data',
                              'dual_scroll_background_right_foreground_down.mp4')

    if os.path.exists(test_video):
        result = detect_shape(test_video, block_size=4,
                              output_dir=os.path.join(project_root, 'output', 'shape_test_v2'))
        print(f"\n最终结果:")
        print(f"  块大小: {result['block_size']}")
        print(f"  背景方向: {result['bg_direction']}")
        print(f"  形状方向: {result['shape_direction']}")
        print(f"  轮廓数: {len(result['contours'])}")
    else:
        print(f"测试视频不存在: {test_video}")
