r"""
====================================================================
实验3：视频分块角度扫描与统计分析
====================================================================

功能：
  1. 读取 data 目录下的视频（支持多个数据目录）
  2. 将视频每帧划分为多个 40×40 的小块（patch）
  3. 对每个小块进行角度扫描（旋转→(3,3,3)*1/27 时空平滑→水平边缘检测）
  4. 找到每个小块边缘平均值最大的角度
  5. 对所有小块的最佳角度进行统计分析：众数 + 条形图

用法:
  python run.py                              # 扫描本地 + 项目根目录 data/
  python run.py --data-dir ../data           # 指定自定义数据目录
  python run.py --data-dir ./data --data-dir ../data  # 多个数据目录
  python run.py --only-local                  # 仅扫描 experiments/3/data/

依赖：
  - opencv-python (cv2)
  - numpy
  - scipy（仅用于 2D 卷积）
  - matplotlib
  - tqdm（可选，用于进度条）
====================================================================
"""

from __future__ import annotations

import os
import sys
import glob
import argparse
from collections import deque, Counter
from typing import Tuple, List

import cv2
import numpy as np
from scipy.ndimage import convolve
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# matplotlib 中文字体配置
# ============================================================
_ZH_FONTS: list[str] = [
    'Microsoft YaHei', 'SimHei', 'DengXian',
    'SimSun', 'KaiTi', 'FangSong',
]
for _font in _ZH_FONTS:
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [_font]
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

# ============================================================
# 路径配置（所有路径基于脚本所在目录）
# ============================================================
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_DIR: str = os.path.join(SCRIPT_DIR, 'output')

# 将项目根目录加入 sys.path 以导入 src 模块
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 项目根目录的 data/ 路径
ROOT_DATA_DIR: str = os.path.join(PROJECT_ROOT, 'data')

from src.video_converter import mp4_to_grayscale_array

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ============================================================
# 常量
# ============================================================
PATCH_SIZE: int = 40

# 2D 均值卷积核 (3×3)*1/9
_KERNEL_2D_AVG: np.ndarray = np.ones((3, 3), dtype=np.float32) / 9.0

# 水平边缘检测 Sobel 核 (3×3)
_KERNEL_EDGE_H: np.ndarray = np.array([
    [1.0,  1.0,  1.0],
    [0.0,  0.0,  0.0],
    [-1.0, -1.0, -1.0],
], dtype=np.float32)


# ============================================================
# 旋转工具（从实验2完全复用）
# ============================================================
def get_rotation_params(height: int, width: int, angle: float) -> Tuple[np.ndarray, int, int]:
    """计算旋转仿射矩阵和旋转后画布尺寸。"""
    center = (width / 2.0, height / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    cos_abs = abs(M[0, 0])
    sin_abs = abs(M[0, 1])
    new_w = int(height * sin_abs + width * cos_abs)
    new_h = int(height * cos_abs + width * sin_abs)
    M[0, 2] += new_w / 2.0 - center[0]
    M[1, 2] += new_h / 2.0 - center[1]
    return M, new_w, new_h


def rotate_frame(frame: np.ndarray, M: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """对单帧执行旋转。"""
    return cv2.warpAffine(
        frame, M, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def compute_valid_mask(M: np.ndarray, new_w: int, new_h: int,
                       orig_h: int, orig_w: int) -> np.ndarray:
    """计算旋转后的有效像素掩码（原始图像区域为 True）。"""
    corners = np.array([[0, 0], [orig_w, 0], [orig_w, orig_h], [0, orig_h]], dtype=np.float32)
    mapped = cv2.transform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
    mask = np.zeros((new_h, new_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(mapped).astype(np.int32), 1)
    return mask.astype(bool)


# ============================================================
# 3D 平滑 + 边缘检测（从实验2完全复用）
# ============================================================
def process_three_frames(f0: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """
    对连续三帧执行 (3,3,3)*1/27 时空平滑 → 水平边缘检测。
    返回边缘强度绝对值。
    """
    # 逐帧 2D 空间平滑 (3×3)*1/9
    s0 = convolve(f0.astype(np.float32), _KERNEL_2D_AVG, mode='constant', cval=0.0)
    s1 = convolve(f1.astype(np.float32), _KERNEL_2D_AVG, mode='constant', cval=0.0)
    s2 = convolve(f2.astype(np.float32), _KERNEL_2D_AVG, mode='constant', cval=0.0)
    # 时间平均 = (1/3) * 三个空间平滑结果 = 等效于 (3,3,3)*1/27
    smoothed = (s0 + s1 + s2) / 3.0
    # 水平边缘检测
    edges = convolve(smoothed, _KERNEL_EDGE_H, mode='constant', cval=0.0)
    return np.abs(edges)


# ============================================================
# 对单个 patch 进行角度粗扫+精扫，返回最佳角度
# ============================================================
def scan_patch_angles_coarse_to_fine(
    patch_arr: np.ndarray,
    coarse_step: float = 5.0,
    fine_step: float = 0.5,
    fine_range: float = 10.0,
) -> Tuple[float, float]:
    """
    对单个小块的视频数组进行两阶段角度扫描。

    阶段1（粗扫）：以 coarse_step 步长扫描 0~180°，找到粗略最佳角度。
    阶段2（精扫）：在粗扫最佳角度 ± fine_range 范围内以 fine_step 步长精扫。

    参数:
        patch_arr: 形状为 (帧数, PATCH_SIZE, PATCH_SIZE) 的灰度数组
        coarse_step: 粗扫步长（度），默认 5°
        fine_step: 精扫步长（度），默认 0.5°
        fine_range: 精扫范围（度），默认 ±10°

    返回:
        (best_angle, best_value): 边缘平均值最大的角度及其对应的值
    """
    total_frames, height, width = patch_arr.shape

    # ---- 阶段1：粗扫 ----
    coarse_angles = np.arange(0.0, 181.0 + 1e-9, coarse_step)
    coarse_results = _batch_scan(patch_arr, coarse_angles)
    best_coarse_idx = int(np.argmax(coarse_results))
    best_coarse_angle = float(coarse_angles[best_coarse_idx])

    # ---- 阶段2：精扫 ----
    fine_start = max(0.0, best_coarse_angle - fine_range)
    fine_end = min(180.0, best_coarse_angle + fine_range)
    fine_angles = np.arange(fine_start, fine_end + fine_step / 2, fine_step)
    if len(fine_angles) <= 2:
        # 如果精扫范围太小，用粗扫结果
        return best_coarse_angle, float(coarse_results[best_coarse_idx])

    fine_results = _batch_scan(patch_arr, fine_angles)
    best_fine_idx = int(np.argmax(fine_results))
    best_fine_angle = float(fine_angles[best_fine_idx])
    best_fine_value = float(fine_results[best_fine_idx])

    return best_fine_angle, best_fine_value


def _batch_scan(patch_arr: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    对小块视频在指定角度列表上进行批处理扫描。
    返回每个角度的边缘平均值数组。
    """
    total_frames, height, width = patch_arr.shape
    num_angles = len(angles)

    # 预计算所有角度的旋转参数和有效掩码
    rot_params: list = []
    valid_masks: list[np.ndarray] = []
    for angle in angles:
        M, new_w, new_h = get_rotation_params(height, width, angle)
        rot_params.append((M, new_w, new_h))
        valid_masks.append(compute_valid_mask(M, new_w, new_h, height, width))

    edge_sum = np.zeros(num_angles, dtype=np.float64)
    valid_counts = np.array([np.sum(m) for m in valid_masks], dtype=np.int64)
    buffers: list[deque] = [deque(maxlen=3) for _ in range(num_angles)]

    for frame_idx in range(total_frames):
        frame = patch_arr[frame_idx]
        for ai in range(num_angles):
            M, new_w, new_h = rot_params[ai]
            rotated = rotate_frame(frame, M, new_w, new_h)
            buffers[ai].append(rotated)
            if frame_idx >= 2:
                edges = process_three_frames(buffers[ai][0], buffers[ai][1], buffers[ai][2])
                edge_sum[ai] += np.sum(edges[valid_masks[ai]])

    # 计算平均值
    averages = np.divide(edge_sum, valid_counts,
                         out=np.zeros_like(edge_sum), where=valid_counts > 0)
    return averages


# ============================================================
# 将视频划分为多个 patch 并分别扫描
# ============================================================
def process_video_by_patches(
    arr: np.ndarray,
    patch_size: int = PATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    将视频划分为多个小块，对每个小块进行角度扫描。

    参数:
        arr: 形状为 (帧数, 高度, 宽度) 的灰度视频数组
        patch_size: 小块边长（像素）

    返回:
        best_angles:  每个小块的最佳角度组成的二维数组 (行数, 列数)
        best_values:  每个小块的最佳边缘平均值组成的二维数组
        patch_positions: 每个小块在原图中的起始 (y, x) 位置列表
    """
    total_frames, height, width = arr.shape

    # 计算可以完整划分的 patch 数量（丢弃边缘不足部分）
    n_patches_h = height // patch_size
    n_patches_w = width // patch_size

    # 裁剪到 patch 整数倍
    crop_h = n_patches_h * patch_size
    crop_w = n_patches_w * patch_size
    if crop_h != height or crop_w != width:
        print(f"  原始尺寸 {width}x{height}，裁剪到 {crop_w}x{crop_h} 以适应 patch 划分")
        arr = arr[:, :crop_h, :crop_w]

    print(f"  视频尺寸: {crop_w}x{crop_h}, Patch大小: {patch_size}x{patch_size}")
    print(f"  划分为 {n_patches_h} 行 × {n_patches_w} 列 = {n_patches_h * n_patches_w} 个 patch")

    best_angles = np.full((n_patches_h, n_patches_w), -1.0, dtype=np.float64)
    best_values = np.zeros((n_patches_h, n_patches_w), dtype=np.float64)
    patch_positions: List[Tuple[int, int]] = []

    total_patches = n_patches_h * n_patches_w
    patch_iter = range(total_patches)
    if tqdm is not None:
        patch_iter = tqdm(patch_iter, desc="处理Patch", unit="patch")

    for idx in patch_iter:
        pi = idx // n_patches_w
        pj = idx % n_patches_w
        y_start = pi * patch_size
        x_start = pj * patch_size
        patch_positions.append((y_start, x_start))

        # 提取该 patch 在所有帧中的内容
        patch_arr = arr[:, y_start:y_start + patch_size, x_start:x_start + patch_size]

        # 跳过几乎纯色（无有效纹理）的 patch 以节省时间
        if np.max(patch_arr) - np.min(patch_arr) < 20:
            best_angles[pi, pj] = -1.0
            best_values[pi, pj] = 0.0
            continue

        best_angle, best_value = scan_patch_angles_coarse_to_fine(patch_arr)
        best_angles[pi, pj] = best_angle
        best_values[pi, pj] = best_value

    return best_angles, best_values, patch_positions


# ============================================================
# 角度统计分析：众数 + 条形图 + 热力图
# ============================================================
def analyze_angles(best_angles: np.ndarray, best_values: np.ndarray,
                   out_dir: str, video_name: str = ""):
    """对最佳角度进行统计分析，生成众数、条形图和热力图。"""
    valid_mask = best_angles >= 0
    valid_angles = best_angles[valid_mask]

    if len(valid_angles) == 0:
        print("  ✗ 没有有效的角度数据")
        return

    n_patches_h, n_patches_w = best_angles.shape

    # ---- 统计：直方图（0.5° 分桶） ----
    bin_width = 1.0
    bins = np.arange(0.0, 181.0 + bin_width, bin_width)
    counts, bin_edges = np.histogram(valid_angles, bins=bins)

    # 众数 = 频次最高的角度区间中点
    mode_bin_idx = int(np.argmax(counts))
    mode_angle = (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2.0
    mode_count = int(counts[mode_bin_idx])
    total_valid = len(valid_angles)

    print()
    print("╔══════════════════════════════════════════╗")
    print("║          角度统计分析结果                 ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  总 Patch 数:  {n_patches_h * n_patches_w:3d}                      ║")
    print(f"║  有效 Patch:   {total_valid:3d}                         ║")
    print(f"║  众数角度:     {mode_angle:6.1f}°                    ║")
    print(f"║  众数频次:     {mode_count:3d} / {total_valid:3d}                 ║")
    print(f"║  众数占比:     {mode_count / total_valid * 100:5.1f}%                    ║")
    print("╚══════════════════════════════════════════╝")

    # ---- 保存 CSV 统计 ----
    csv_path = os.path.join(out_dir, 'patch_angle_statistics.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("AngleCenter(deg),Count\n")
        for i in range(len(bins) - 1):
            center = (bin_edges[i] + bin_edges[i + 1]) / 2.0
            f.write(f"{center:.1f},{counts[i]}\n")
    print(f"  统计数据: {csv_path}")

    # ---- 保存每个 patch 的详细角度值 CSV ----
    patch_csv = os.path.join(out_dir, 'patch_angles.csv')
    with open(patch_csv, 'w', encoding='utf-8') as f:
        f.write("Row,Col,BestAngle(deg),BestValue\n")
        for pi in range(n_patches_h):
            for pj in range(n_patches_w):
                f.write(f"{pi},{pj},{best_angles[pi, pj]:.1f},{best_values[pi, pj]:.6f}\n")
    print(f"  Patch角度明细: {patch_csv}")

    # ---- 绘制统计图（条形图 + 热力图） ----
    plt.figure(figsize=(16, 7))

    # 子图1：条形分布图
    ax1 = plt.subplot(1, 2, 1)
    bar_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_colors = ['#1f77b4'] * len(bar_centers)
    bar_colors[mode_bin_idx] = '#d62728'  # 众数列标红
    ax1.bar(bar_centers, counts, width=bin_width * 0.8, color=bar_colors,
            edgecolor='black', linewidth=0.3)
    ax1.set_xlabel('最佳角度 (deg)', fontsize=12)
    ax1.set_ylabel('Patch 数量', fontsize=12)
    title1 = '各最佳角度 Patch 数量分布'
    if video_name:
        title1 += f' - {video_name}'
    ax1.set_title(title1, fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim(-2, 182)

    # 标注众数
    ax1.annotate(
        f'众数: {mode_angle:.1f}°\n(n={mode_count})',
        xy=(mode_angle, mode_count),
        xytext=(mode_angle + 25, mode_count + max(counts) * 0.08),
        arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
        fontsize=11, color='red', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9),
    )

    # 子图2：热力图
    ax2 = plt.subplot(1, 2, 2)
    # 将无效 patch (-1) 映射为 NaN 以使其显示为白色/灰色
    display_angles = best_angles.copy()
    display_angles[~valid_mask] = np.nan
    im = ax2.imshow(display_angles, cmap='hsv', vmin=0, vmax=180, aspect='auto')
    cbar = plt.colorbar(im, ax=ax2, label='最佳角度 (deg)')
    ax2.set_xlabel('Patch 列索引', fontsize=12)
    ax2.set_ylabel('Patch 行索引', fontsize=12)
    ax2.set_title('各 Patch 最佳角度热力图', fontsize=14)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        path = os.path.join(out_dir, f'angle_statistics.{ext}')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  统计图: {path}")
    plt.close()

    # ---- 单独保存大尺寸热力图 ----
    plt.figure(figsize=(12, 10))
    plt.imshow(display_angles, cmap='hsv', vmin=0, vmax=180, aspect='auto')
    plt.colorbar(label='最佳角度 (deg)')
    plt.xlabel('Patch 列索引')
    plt.ylabel('Patch 行索引')
    plt.title(f'各 Patch 最佳角度热力图 - {video_name}', fontsize=14)
    path = os.path.join(out_dir, 'angle_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  热力图: {path}")
    plt.close()


# ============================================================
# 扫描 data/ 下的视频文件
# ============================================================
def scan_video_files(data_dir: str | None = None) -> List[Tuple[str, str]]:
    """扫描目录下的 mp4/avi 视频文件和 _frames 帧序列文件夹。"""
    if data_dir is None:
        data_path = DATA_DIR
    else:
        data_path = data_dir if os.path.isabs(data_dir) else os.path.join(PROJECT_ROOT, data_dir)

    if not os.path.isdir(data_path):
        print(f"  ✗ 数据目录不存在: {data_path}")
        return []

    result: list[tuple[str, str]] = []

    for ext in ('*.mp4', '*.avi'):
        for vp in sorted(glob.glob(os.path.join(data_path, ext))):
            name = os.path.splitext(os.path.basename(vp))[0]
            result.append((name, vp))

    for entry in sorted(glob.glob(os.path.join(data_path, "*_frames"))):
        if os.path.isdir(entry):
            pngs = glob.glob(os.path.join(entry, "*.png"))
            if pngs:
                name = os.path.basename(entry)
                if name.endswith('_frames'):
                    name = name[:-7]
                result.append((name, entry))

    if not result:
        print(f"  ✗ 在 {data_path} 下未找到视频文件或帧序列")
        return []

    return result


# ============================================================
# 解析命令行参数
# ============================================================
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='实验3：视频分块角度扫描与统计分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py                               # 扫描本地 + 项目根目录 data/
  python run.py --only-local                   # 仅扫描 experiments/3/data/
  python run.py --data-dir ../data             # 仅扫描项目根目录 data/
  python run.py --data-dir D:/videos           # 扫描自定义目录
        """,
    )
    parser.add_argument(
        '--data-dir', '-d', action='append', dest='data_dirs', default=[],
        help='指定要扫描的数据目录（可多次使用）',
    )
    parser.add_argument(
        '--only-local', action='store_true',
        help='仅扫描 experiments/3/data/，不扫描项目根目录 data/',
    )
    return parser.parse_args()


# ============================================================
# 主流程
# ============================================================
def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          实验3：视频分块角度扫描与统计分析                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    args = parse_args()

    # 确定要扫描的数据目录列表
    scan_dirs: list[str] = []

    if args.data_dirs:
        # 用户指定了 --data-dir，只使用指定的目录
        for d in args.data_dirs:
            resolved = d if os.path.isabs(d) else os.path.join(SCRIPT_DIR, d)
            scan_dirs.append(resolved)
    elif args.only_local:
        # 仅扫描本地 data/
        scan_dirs.append(DATA_DIR)
    else:
        # 默认：同时扫描本地 data/ 和项目根目录 data/
        scan_dirs.append(DATA_DIR)
        if os.path.isdir(ROOT_DATA_DIR):
            scan_dirs.append(ROOT_DATA_DIR)

    # 去重
    unique_dirs: list[str] = []
    for d in scan_dirs:
        real = os.path.realpath(d)
        if real not in [os.path.realpath(u) for u in unique_dirs]:
            unique_dirs.append(d)
    scan_dirs = unique_dirs

    # 扫描所有目录收集视频
    all_videos: list[tuple[str, str]] = []
    for data_path in scan_dirs:
        if not os.path.isdir(data_path):
            print(f"  ! 数据目录不存在，跳过: {data_path}")
            continue
        print(f"\n  ◆ 扫描数据目录: {data_path}")
        videos = scan_video_files(data_path)
        # 避免同名视频重复（后出现的覆盖先出现的）
        seen_names: set[str] = set()
        for name, path in videos:
            if name not in seen_names:
                all_videos.append((name, path))
                seen_names.add(name)

    if not all_videos:
        print("\n  ✗ 未找到任何视频文件，退出。")
        sys.exit(1)

    print(f"\n  ◆ 共发现 {len(all_videos)} 个视频:")
    for vname, vpath in all_videos:
        print(f"      - {vname} ({vpath})")

    for vname, vpath in all_videos:
        print(f"\n{'=' * 60}")
        print(f"  处理视频: {vname}")
        print(f"{'=' * 60}")

        video_out = os.path.join(OUTPUT_DIR, 'patch_angle_scan', vname)
        os.makedirs(video_out, exist_ok=True)

        print(f"  ◆ 读取视频...")
        arr = mp4_to_grayscale_array(vpath)
        print(f"    数组形状: {arr.shape}")

        print(f"  ◆ 划分 patch 并进行角度扫描（粗扫 5° → 精扫 0.5°）...")
        best_angles, best_values, positions = process_video_by_patches(arr, PATCH_SIZE)
        del arr  # 释放内存

        print(f"  ◆ 统计分析...")
        analyze_angles(best_angles, best_values, video_out, video_name=vname)

    out_dir = os.path.join(OUTPUT_DIR, 'patch_angle_scan')
    print(f"\n{'=' * 60}")
    print(f"  全部处理完成！输出目录: {out_dir}")
    print(f"  共处理 {len(all_videos)} 个视频")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()