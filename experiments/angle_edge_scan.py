"""
============================================================
角度-边缘平均值扫描实验（优化版 v5 —— 多视频支持）
============================================================

以旋转角度 rotate_angle 为自变量（步长 1°），
边缘检测结果平均值为因变量，绘制曲线图。

v5 新增功能：
  - 自动扫描 data/ 目录下所有 .mp4 视频，逐个处理
  - 每个视频的结果保存在独立子目录中
  - 生成多视频对比叠加曲线图
  - 支持通过命令行参数指定要处理的视频（支持通配符匹配）

优化策略（v4）：
  - 预计算每个角度的有效像素掩码（旋转后非黑区域），避免重复计算
  - 使用 collections.deque 替代 list.pop(0) 管理滑动窗口
  - 使用 float32 代替 float64 减少内存带宽
  - 添加 tqdm 进度条（如可用），否则回退到原生打印
  - 添加类型注解，提升代码可读性
  - 预计算 valid_count（每个角度有效像素数），避免每帧重复求和

依赖：
  - opencv-python (cv2)
  - numpy
  - scipy（仅用于 2D 卷积）
  - matplotlib
  - tqdm（可选，用于进度条）
============================================================
"""

from __future__ import annotations

import os
import sys
import glob
from collections import deque
from typing import Tuple, List

import cv2
import numpy as np
from scipy.ndimage import convolve
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 全局 matplotlib 中文字体配置
# ============================================================
# Windows 系统常见中文字体回退列表
_ZH_FONTS: list[str] = [
    'Microsoft YaHei',      # 微软雅黑
    'SimHei',               # 黑体
    'DengXian',             # 等线
    'SimSun',               # 宋体
    'KaiTi',                # 楷体
    'FangSong',             # 仿宋
]
for _font in _ZH_FONTS:
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [_font]
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

# 将项目根目录加入 sys.path
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.video_converter import mp4_to_grayscale_array

# 尝试导入 tqdm，不可用时回退到简单打印
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


# ============================================================
# 常量定义
# ============================================================

# 2D 平均核（3×3）
_KERNEL_2D_AVG: np.ndarray = np.ones((3, 3), dtype=np.float32) / 9.0


# ============================================================
# 旋转 + 缓存
# ============================================================

def get_rotation_params(height: int, width: int, angle: float) -> Tuple[np.ndarray, int, int]:
    """计算旋转矩阵和新尺寸。

    参数:
        height: 原始图像高度
        width: 原始图像宽度
        angle: 旋转角度（度）

    返回:
        (rotation_matrix, new_width, new_height) 三元组
    """
    center: Tuple[float, float] = (width / 2.0, height / 2.0)
    rotation_matrix: np.ndarray = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    cos_abs: float = abs(rotation_matrix[0, 0])
    sin_abs: float = abs(rotation_matrix[0, 1])
    new_width: int = int(height * sin_abs + width * cos_abs)
    new_height: int = int(height * cos_abs + width * sin_abs)

    rotation_matrix[0, 2] += new_width / 2.0 - center[0]
    rotation_matrix[1, 2] += new_height / 2.0 - center[1]

    return rotation_matrix, new_width, new_height


def rotate_frame(frame: np.ndarray, rot_mat: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """旋转单帧图像。

    参数:
        frame: 输入灰度帧，形状 (height, width)
        rot_mat: 2×3 旋转矩阵
        new_w: 输出图像宽度
        new_h: 输出图像高度

    返回:
        旋转后的帧，形状 (new_h, new_w)
    """
    return cv2.warpAffine(
        frame, rot_mat, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def compute_valid_mask(rot_mat: np.ndarray, new_w: int, new_h: int,
                       orig_h: int, orig_w: int) -> np.ndarray:
    """计算旋转后的有效像素掩码（非黑色填充区域）。

    通过将原始图像四个角点映射到旋转后图像，生成一个凸多边形掩码，
    标记哪些像素属于原始图像内容（非黑色填充）。

    参数:
        rot_mat: 2×3 旋转矩阵
        new_w: 旋转后图像宽度
        new_h: 旋转后图像高度
        orig_h: 原始图像高度
        orig_w: 原始图像宽度

    返回:
        布尔掩码，形状 (new_h, new_w)，True 表示有效像素
    """
    # 原始图像四个角点
    corners: np.ndarray = np.array([
        [0, 0],
        [orig_w, 0],
        [orig_w, orig_h],
        [0, orig_h],
    ], dtype=np.float32)

    # 映射到旋转后坐标
    mapped_corners: np.ndarray = cv2.transform(
        corners.reshape(1, -1, 2), rot_mat
    ).reshape(-1, 2)

    # 生成凸多边形掩码
    mask: np.ndarray = np.zeros((new_h, new_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(mapped_corners).astype(np.int32), 1)
    return mask.astype(bool)


# ============================================================
# 3D 平滑 + 边缘检测
# ============================================================

# 水平边缘检测核（float32 版本，避免 SPATIAL_180 的 float64 导致内存膨胀）
_KERNEL_EDGE_H: np.ndarray = np.array([
    [1.0,  1.0,  1.0],
    [0.0,  0.0,  0.0],
    [-1.0, -1.0, -1.0],
], dtype=np.float32)


def process_three_frames(f0: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """
    对连续 3 帧执行：(3,3,3)*1/27 平滑 → 水平边缘检测 → 返回边缘强度。

    等价于完整 3D 卷积 (3,3,3)*1/27 后做边缘检测。
    由于卷积是线性的，顺序可交换。
    这里先做 3×3 空间平均，再三帧时间平均，最后边缘检测。

    参数:
        f0, f1, f2: 连续三帧，形状 (height, width)，uint8

    返回:
        边缘强度图，形状 (height, width)，float32
    """
    # 3×3 空间平均（全部使用 float32，避免内存膨胀）
    s0: np.ndarray = convolve(f0.astype(np.float32), _KERNEL_2D_AVG, mode='constant', cval=0.0)
    s1: np.ndarray = convolve(f1.astype(np.float32), _KERNEL_2D_AVG, mode='constant', cval=0.0)
    s2: np.ndarray = convolve(f2.astype(np.float32), _KERNEL_2D_AVG, mode='constant', cval=0.0)

    # 三帧时间平均（原地操作减少内存分配）
    smoothed: np.ndarray = s0
    smoothed += s1
    smoothed += s2
    smoothed /= 3.0

    # 水平边缘检测（使用 float32 核）
    edges: np.ndarray = convolve(smoothed, _KERNEL_EDGE_H, mode='constant', cval=0.0)
    return np.abs(edges)


# ============================================================
# 扫描主流程
# ============================================================

def scan_angles(arr: np.ndarray, start_angle: float = 0.0, end_angle: float = 90.0,
                step: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """扫描指定角度范围，计算每个角度的边缘检测平均值。

    对每个角度，逐帧处理并累积边缘强度。
    使用滑动窗口（3 帧）管理内存，预计算有效掩码避免重复计算。

    参数:
        arr: 视频数组，形状 (frames, height, width)，uint8
        start_angle: 起始角度（度）
        end_angle: 终止角度（度）
        step: 角度步长（度，默认 1.0）

    返回:
        (angles, averages) 元组
            angles: 角度数组，形状 (num_angles,)
            averages: 对应的边缘平均值数组，形状 (num_angles,)
    """
    total_frames: int
    height: int
    width: int
    total_frames, height, width = arr.shape

    angles: np.ndarray = np.arange(start_angle, end_angle + 1e-9, step)
    num_angles: int = len(angles)

    # --------------------------------------------------
    # 预计算所有角度的旋转参数和有效掩码
    # --------------------------------------------------
    rot_params: list = []
    valid_masks: list[np.ndarray] = []
    for angle in angles:
        rot_mat, new_w, new_h = get_rotation_params(height, width, angle)
        rot_params.append((rot_mat, new_w, new_h))
        mask: np.ndarray = compute_valid_mask(rot_mat, new_w, new_h, height, width)
        valid_masks.append(mask)

    # --------------------------------------------------
    # 初始化累积器 & 预计算每个角度的有效像素数
    # --------------------------------------------------
    edge_sum: np.ndarray = np.zeros(num_angles, dtype=np.float64)
    valid_counts: np.ndarray = np.array([np.sum(m) for m in valid_masks], dtype=np.int64)

    # 每个角度的滑动窗口 buffer（3 帧旋转结果），使用 deque 优化
    buffers: list[deque] = [deque(maxlen=3) for _ in range(num_angles)]

    # --------------------------------------------------
    # 主循环：逐帧处理
    # --------------------------------------------------
    frame_iterator: range = range(total_frames)
    if tqdm is not None:
        frame_iterator = tqdm(frame_iterator, desc="处理帧", unit="frame")

    for frame_idx in frame_iterator:
        frame: np.ndarray = arr[frame_idx]

        for ai in range(num_angles):
            rot_mat, new_w, new_h = rot_params[ai]

            # 旋转当前帧
            rotated: np.ndarray = rotate_frame(frame, rot_mat, new_w, new_h)

            # 更新滑动窗口 buffer
            buffers[ai].append(rotated)

            # 从第 3 帧开始处理（buffer 填满后）
            if frame_idx >= 2:
                edges: np.ndarray = process_three_frames(
                    buffers[ai][0], buffers[ai][1], buffers[ai][2]
                )

                # 使用预计算的有效掩码
                mask: np.ndarray = valid_masks[ai]
                edge_sum[ai] += np.sum(edges[mask])

    if tqdm is None:
        print(f"\r处理完成: {total_frames}/{total_frames} 帧")

    # --------------------------------------------------
    # 计算平均值
    # --------------------------------------------------
    averages: np.ndarray = np.divide(
        edge_sum, valid_counts,
        out=np.zeros_like(edge_sum),
        where=valid_counts > 0,
    )

    return angles, averages


# ============================================================
# 结果可视化与保存（单视频）
# ============================================================

def plot_results(angles: np.ndarray, averages: np.ndarray, output_dir: str,
                 video_name: str = "") -> None:
    """绘制角度-边缘平均值曲线图并保存。

    参数:
        angles: 角度数组
        averages: 边缘平均值数组
        output_dir: 输出目录路径
        video_name: 视频名称（用于标题）
    """
    plt.figure(figsize=(12, 6))

    plt.plot(angles, averages, 'b-o', markersize=3, linewidth=1.5, label='边缘平均值')

    max_idx: int = int(np.argmax(averages))
    min_idx: int = int(np.argmin(averages))

    plt.plot(angles[max_idx], averages[max_idx], 'r*', markersize=12,
             label=f'最大值: {averages[max_idx]:.4f} @ {angles[max_idx]:.0f}°')
    plt.plot(angles[min_idx], averages[min_idx], 'g*', markersize=12,
             label=f'最小值: {averages[min_idx]:.4f} @ {angles[min_idx]:.0f}°')

    plt.xlabel('旋转角度 (度)', fontsize=12)
    plt.ylabel('边缘检测结果平均值', fontsize=12)
    title = '旋转角度 vs 边缘检测平均值'
    if video_name:
        title += f' — {video_name}'
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    plot_path: str = os.path.join(output_dir, 'angle_vs_edge_average.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    pdf_path: str = os.path.join(output_dir, 'angle_vs_edge_average.pdf')
    plt.savefig(pdf_path, dpi=150)

    plt.close()


def save_results(angles: np.ndarray, averages: np.ndarray, output_dir: str) -> None:
    """将扫描结果保存为 CSV 文件。

    参数:
        angles: 角度数组
        averages: 边缘平均值数组
        output_dir: 输出目录路径
    """
    csv_path: str = os.path.join(output_dir, 'angle_edge_average_data.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("角度(度),边缘平均值\n")
        for angle, avg in zip(angles, averages):
            f.write(f"{angle:.0f},{avg:.6f}\n")


# ============================================================
# 多视频对比叠加图
# ============================================================

def plot_multi_video_comparison(all_results: List[dict], output_dir: str) -> None:
    """绘制多视频对比叠加曲线图。

    将所有视频的角度-边缘平均值曲线绘制在同一张图上，
    使用不同颜色和线型区分不同视频。

    参数:
        all_results: 列表，每个元素为包含以下键的字典：
            - 'name': 视频名称
            - 'angles': 角度数组
            - 'averages': 边缘平均值数组
        output_dir: 输出目录路径
    """
    plt.figure(figsize=(14, 7))

    # 预定义颜色和线型循环
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for idx, result in enumerate(all_results):
        name = result['name']
        angles = result['angles']
        averages = result['averages']
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        plt.plot(angles, averages, color=color, marker=marker,
                 markersize=2, linewidth=1.2, label=name, alpha=0.85)

    plt.xlabel('旋转角度 (度)', fontsize=12)
    plt.ylabel('边缘检测结果平均值', fontsize=12)
    plt.title('多视频对比：旋转角度 vs 边缘检测平均值', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc='best', ncol=2)

    plot_path: str = os.path.join(output_dir, 'multi_video_comparison.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    pdf_path: str = os.path.join(output_dir, 'multi_video_comparison.pdf')
    plt.savefig(pdf_path, dpi=150)

    plt.close()
    print(f"    多视频对比图已保存: {plot_path}")


def save_multi_video_summary(all_results: List[dict], output_dir: str) -> None:
    """保存多视频扫描结果汇总 CSV。

    参数:
        all_results: 列表，每个元素为包含以下键的字典：
            - 'name': 视频名称
            - 'angles': 角度数组
            - 'averages': 边缘平均值数组
        output_dir: 输出目录路径
    """
    csv_path: str = os.path.join(output_dir, 'multi_video_summary.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        # 表头：角度, 视频1平均值, 视频2平均值, ...
        header_parts = ["角度(度)"]
        for result in all_results:
            header_parts.append(result['name'])
        f.write(','.join(header_parts) + '\n')

        # 数据行
        for i in range(len(all_results[0]['angles'])):
            row_parts = [f"{all_results[0]['angles'][i]:.0f}"]
            for result in all_results:
                row_parts.append(f"{result['averages'][i]:.6f}")
            f.write(','.join(row_parts) + '\n')

    print(f"    多视频汇总 CSV 已保存: {csv_path}")


# ============================================================
# 视频扫描工具
# ============================================================

def scan_video_files(data_dir: str = "data") -> List[Tuple[str, str]]:
    """扫描 data 目录下所有 .mp4 文件，返回 (文件名, 完整路径) 列表。

    参数:
        data_dir: 数据目录路径（相对于项目根目录或绝对路径）

    返回:
        [(name, path), ...] 列表，按文件名排序
    """
    data_path = os.path.join(PROJECT_ROOT, data_dir)
    if not os.path.isdir(data_path):
        print(f"  ✗ 错误: 数据目录不存在: {data_path}")
        return []

    video_files = sorted(glob.glob(os.path.join(data_path, "*.mp4")))
    if not video_files:
        print(f"  ✗ 错误: 在 {data_path} 下未找到任何 .mp4 文件")
        return []

    result = []
    for vp in video_files:
        name = os.path.splitext(os.path.basename(vp))[0]
        result.append((name, vp))

    return result


# ============================================================
# 单视频处理函数
# ============================================================

def process_single_video(video_name: str, video_path: str,
                         start_angle: float, end_angle: float, step: float,
                         base_output_dir: str) -> dict | None:
    """处理单个视频：读取 → 扫描 → 保存结果。

    参数:
        video_name: 视频名称（不含扩展名）
        video_path: 视频完整路径
        start_angle: 起始角度
        end_angle: 终止角度
        step: 角度步长
        base_output_dir: 基础输出目录

    返回:
        包含 'name', 'angles', 'averages' 的字典，失败时返回 None
    """
    # 每个视频独立输出目录
    output_dir: str = os.path.join(base_output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  处理视频: {video_name:<34s} ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  视频路径 : {video_path}")
    print(f"  扫描范围 : {start_angle:.0f}° ~ {end_angle:.0f}°，步长 {step}°")
    print(f"  输出目录 : {output_dir}")
    print()

    # 检查视频文件是否存在
    if not os.path.isfile(video_path):
        print(f"  ✗ 错误: 未找到视频文件: {video_path}")
        return None

    # ── 读取视频 ──
    print("  ◆ 步骤1: 读取视频")
    try:
        arr: np.ndarray = mp4_to_grayscale_array(video_path)
    except Exception as e:
        print(f"  ✗ 读取视频失败: {e}")
        return None
    print(f"    视频数组: {arr.shape}, dtype: {arr.dtype}")
    print()

    # ── 扫描 ──
    num_angles: int = int((end_angle - start_angle) / step) + 1
    print(f"  ◆ 步骤2: 角度扫描 ({start_angle:.0f}° ~ {end_angle:.0f}°，步长 {step}°，共 {num_angles} 个角度)")
    angles: np.ndarray
    averages: np.ndarray
    angles, averages = scan_angles(arr, start_angle, end_angle, step)

    # 释放大数组内存
    del arr

    # ── 结果统计 ──
    max_val: float = averages.max()
    min_val: float = averages.min()
    max_angle: float = angles[np.argmax(averages)]
    min_angle: float = angles[np.argmin(averages)]
    mean_val: float = averages.mean()
    std_val: float = averages.std()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║                    扫描结果                          ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  角度范围      {angles[0]:6.0f}° ~ {angles[-1]:5.0f}°          ║")
    print(f"║  数据点数      {len(angles):>3d}                      ║")
    print(f"║  平均值范围    {min_val:8.6f} ~ {max_val:8.6f}  ║")
    print(f"║  最大值        {max_val:8.6f}  @  {max_angle:5.0f}°          ║")
    print(f"║  最小值        {min_val:8.6f}  @  {min_angle:5.0f}°          ║")
    print(f"║  总体均值      {mean_val:8.6f}                    ║")
    print(f"║  标准差        {std_val:8.6f}                    ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ── 保存 ──
    print()
    print("  ◆ 步骤3: 保存结果")
    save_results(angles, averages, output_dir)
    plot_results(angles, averages, output_dir, video_name=video_name)
    print(f"    输出目录: {output_dir}")
    print(f"    数据文件: angle_edge_average_data.csv")
    print(f"    曲线图  : angle_vs_edge_average.png")
    print(f"    矢量图  : angle_vs_edge_average.pdf")
    print()

    return {
        'name': video_name,
        'angles': angles,
        'averages': averages,
    }


# ============================================================
# 入口
# ============================================================

def main() -> None:
    """主函数：扫描 data/ 目录下所有视频，逐个执行角度扫描。

    命令行参数（可选，按位置）:
        [video_filter]  视频名称过滤关键字（支持部分匹配，如 "snow" 匹配所有含 snow 的视频）
                        不提供则处理 data/ 下所有 .mp4 文件
        [start]         起始角度（默认 0）
        [end]           终止角度（默认 180）
        [step]          角度步长（默认 0.5）
    """
    # ── 解析命令行参数 ──
    video_filter: str | None = None
    start_angle: float = 0.0
    end_angle: float = 180.0
    step: float = 0.5

    if len(sys.argv) >= 2:
        video_filter = sys.argv[1]
    if len(sys.argv) >= 3:
        start_angle = float(sys.argv[2])
    if len(sys.argv) >= 4:
        end_angle = float(sys.argv[3])
    if len(sys.argv) >= 5:
        step = float(sys.argv[4])

    # ── 扫描视频文件 ──
    all_videos = scan_video_files()
    if not all_videos:
        sys.exit(1)

    # 如果指定了过滤关键字，只处理匹配的视频
    if video_filter:
        filtered_videos = [
            (name, path) for name, path in all_videos
            if video_filter.lower() in name.lower()
        ]
        if not filtered_videos:
            print(f"  ✗ 未找到名称包含 '{video_filter}' 的视频文件")
            print(f"    可用视频: {', '.join(n for n, _ in all_videos)}")
            sys.exit(1)
        videos_to_process = filtered_videos
    else:
        videos_to_process = all_videos

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          角度-边缘平均值扫描实验（多视频版）                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  扫描范围 : {start_angle:.0f}° ~ {end_angle:.0f}°，步长 {step}°")
    print(f"  待处理视频: {len(videos_to_process)} 个")
    for i, (name, path) in enumerate(videos_to_process, 1):
        print(f"    [{i:2d}] {name}")
    print()

    # ── 基础输出目录 ──
    base_output_dir: str = os.path.join(PROJECT_ROOT, 'output', 'angle_edge_scan')
    os.makedirs(base_output_dir, exist_ok=True)

    # ── 逐个处理视频 ──
    all_results: list[dict] = []
    for idx, (video_name, video_path) in enumerate(videos_to_process, 1):
        print(f"\n{'=' * 60}")
        print(f"  处理进度: [{idx}/{len(videos_to_process)}]")
        print(f"{'=' * 60}")

        result = process_single_video(
            video_name, video_path,
            start_angle, end_angle, step,
            base_output_dir,
        )
        if result is not None:
            all_results.append(result)

    # ── 多视频对比 ──
    if len(all_results) >= 2:
        print()
        print("=" * 60)
        print("  生成多视频对比叠加图...")
        print("=" * 60)
        plot_multi_video_comparison(all_results, base_output_dir)
        save_multi_video_summary(all_results, base_output_dir)
    elif len(all_results) == 1:
        print()
        print("  仅处理了 1 个视频，跳过对比叠加图生成")

    # ── 完成 ──
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║              所有视频处理完成！                       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  输出目录: {base_output_dir}")
    print(f"  处理视频数: {len(all_results)}/{len(videos_to_process)}")
    print()


if __name__ == "__main__":
    main()
