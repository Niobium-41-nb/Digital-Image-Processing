"""
============================================================
实验2：雪花马赛克视频处理 — 融合版
============================================================

功能：
  1. 生成雪花马赛克测试视频帧序列
  2. 对生成的视频执行旋转→(3,3,3)*1/27 时空平滑→水平边缘检测
  3. 自动扫描 0~180° 所有角度，输出边缘平均值 CSV + 曲线图

用法:
  python run.py                    # 生成+扫描（默认）
  python run.py generate           # 仅生成测试视频
  python run.py scan               # 仅扫描已存在的视频

依赖：
  - opencv-python (cv2)
  - numpy
  - scipy（仅用于 2D 卷积）
  - matplotlib
  - tqdm（可选，用于进度条）
  - Pillow
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
from PIL import Image
from scipy.ndimage import convolve
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIZE: int = 1000

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

from src.video_converter import mp4_to_grayscale_array

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ============================================================
# 常量
# ============================================================
_KERNEL_2D_AVG: np.ndarray = np.ones((3, 3), dtype=np.float32) / 9.0
_KERNEL_EDGE_H: np.ndarray = np.array([
    [1.0,  1.0,  1.0],
    [0.0,  0.0,  0.0],
    [-1.0, -1.0, -1.0],
], dtype=np.float32)

# ============================================================
# 旋转工具
# ============================================================
def get_rotation_params(height: int, width: int, angle: float) -> Tuple[np.ndarray, int, int]:
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
    return cv2.warpAffine(
        frame, M, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def compute_valid_mask(M: np.ndarray, new_w: int, new_h: int,
                       orig_h: int, orig_w: int) -> np.ndarray:
    corners = np.array([[0, 0], [orig_w, 0], [orig_w, orig_h], [0, orig_h]], dtype=np.float32)
    mapped = cv2.transform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
    mask = np.zeros((new_h, new_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(mapped).astype(np.int32), 1)
    return mask.astype(bool)


def rotate_video(arr: np.ndarray, angle: float) -> np.ndarray:
    frames, h, w = arr.shape
    M, new_w, new_h = get_rotation_params(h, w, angle)
    print(f"旋转角度: {angle}°, 原始尺寸: {w}x{h}, 新尺寸: {new_w}x{new_h}")
    result = np.zeros((frames, new_h, new_w), dtype=arr.dtype)
    for i in range(frames):
        result[i] = rotate_frame(arr[i], M, new_w, new_h)
    return result

# ============================================================
# 3D 平滑 + 边缘检测
# ============================================================
def process_three_frames(f0: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    s0 = convolve(f0.astype(np.float32), _KERNEL_2D_AVG, mode='constant', cval=0.0)
    s1 = convolve(f1.astype(np.float32), _KERNEL_2D_AVG, mode='constant', cval=0.0)
    s2 = convolve(f2.astype(np.float32), _KERNEL_2D_AVG, mode='constant', cval=0.0)
    smoothed = s0 + s1 + s2
    smoothed /= 3.0
    edges = convolve(smoothed, _KERNEL_EDGE_H, mode='constant', cval=0.0)
    return np.abs(edges)


def compute_edge_average(edge_arr: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        masked = edge_arr[:, mask]
    else:
        masked = edge_arr[edge_arr > 0]
    return float(np.mean(masked)) if masked.size > 0 else 0.0

# ============================================================
# 扫描视频文件
# ============================================================
def scan_video_files(data_dir: str | None = None) -> List[Tuple[str, str]]:
    if data_dir is None:
        data_path = DATA_DIR
    else:
        data_path = data_dir if os.path.isabs(data_dir) else os.path.join(PROJECT_ROOT, data_dir)

    if not os.path.isdir(data_path):
        print(f"  ✗ 数据目录不存在: {data_path}")
        return []

    result: list[tuple[str, str]] = []

    # 搜索 .mp4 / .avi 视频文件
    for ext in ('*.mp4', '*.avi'):
        for vp in sorted(glob.glob(os.path.join(data_path, ext))):
            name = os.path.splitext(os.path.basename(vp))[0]
            result.append((name, vp))

    # 搜索 PNG 帧序列文件夹（以 _frames 结尾的目录）
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
# 单一角度快速评估
# ============================================================
def evaluate_single_angle(arr: np.ndarray, angle: float) -> float:
    """对视频数组在指定角度下执行旋转→平滑→边缘检测，返回边缘平均值。"""
    # 旋转
    M, new_w, new_h = get_rotation_params(arr.shape[1], arr.shape[2], angle)
    rotated = np.zeros((arr.shape[0], new_h, new_w), dtype=arr.dtype)
    for i in range(arr.shape[0]):
        rotated[i] = rotate_frame(arr[i], M, new_w, new_h)

    valid_mask = rotated[0] > 0

    # (3,3,3)*1/27 平滑 + 边缘检测（逐三帧滑动）
    total_frames = arr.shape[0]
    edge_buffer = deque(maxlen=3)
    edge_sum = 0.0
    valid_count = np.sum(valid_mask)
    frame_count = 0

    for i in range(total_frames):
        edge_buffer.append(rotated[i])
        if i >= 2:
            edges = process_three_frames(edge_buffer[0], edge_buffer[1], edge_buffer[2])
            edge_sum += np.sum(edges[valid_mask])
            frame_count += 1

    return edge_sum / (valid_count * frame_count) if frame_count > 0 and valid_count > 0 else 0.0


# ============================================================
# 生成测试视频
# ============================================================
def generate_snowflake_video(
    data_dir: str,
    width: int = SIZE,
    height: int = SIZE,
    square_size: int = 4,
    fps: int = 30,
    duration: int = 15,
    speed_x: int = 1,
    speed_y: int = 1,
) -> str:
    """生成雪花马赛克滚动视频帧序列，返回帧序列文件夹路径。

    该函数创建具有随机黑白方块纹理（雪花噪点风格）向固定方向循环滚动的视频帧序列。
    整个画面由随机黑白方块组成，纹理沿指定方向持续循环滚动。
    """
    num_frames = fps * duration
    width = (width // square_size) * square_size
    height = (height // square_size) * square_size
    cols = width // square_size
    rows = height // square_size

    small = np.random.choice([0, 255], size=(rows, cols)).astype(np.uint8)

    name = f'snowfall_tiny_{width}x{height}'
    frames_dir = os.path.join(data_dir, f'{name}_frames')
    os.makedirs(frames_dir, exist_ok=True)

    print(f"开始渲染雪花马赛克帧序列: {frames_dir}")
    print(f"尺寸: {width}×{height}, 方块边长: {square_size}像素, 速度: ({speed_x}, {speed_y}) 方块/帧")

    for i in range(num_frames):
        frame_gray = np.repeat(np.repeat(small, square_size, axis=0), square_size, axis=1)
        frame_path = os.path.join(frames_dir, f'frame_{i:05d}.png')
        Image.fromarray(frame_gray, mode='L').save(frame_path)

        if speed_x != 0:
            small = np.roll(small, shift=speed_x, axis=1)
        if speed_y != 0:
            small = np.roll(small, shift=speed_y, axis=0)

        if i % 30 == 0:
            print(f"已生成帧: {i}/{num_frames}")

    print(f"帧序列生成完成: {frames_dir}  ({num_frames} 帧)")
    return frames_dir


# ============================================================
# 角度扫描
# ============================================================
def scan_angles(arr: np.ndarray, start_angle: float = 0.0, end_angle: float = 180.0,
                step: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    total_frames, height, width = arr.shape
    angles = np.arange(start_angle, end_angle + 1e-9, step)
    num_angles = len(angles)

    print(f"预计算 {num_angles} 个角度的旋转参数和掩码...")
    rot_params: list = []
    valid_masks: list[np.ndarray] = []
    for angle in angles:
        M, new_w, new_h = get_rotation_params(height, width, angle)
        rot_params.append((M, new_w, new_h))
        valid_masks.append(compute_valid_mask(M, new_w, new_h, height, width))

    edge_sum = np.zeros(num_angles, dtype=np.float64)
    valid_counts = np.array([np.sum(m) for m in valid_masks], dtype=np.int64)
    buffers: list[deque] = [deque(maxlen=3) for _ in range(num_angles)]

    frame_iter = range(total_frames)
    if tqdm is not None:
        frame_iter = tqdm(frame_iter, desc="处理帧", unit="frame")

    for frame_idx in frame_iter:
        frame = arr[frame_idx]
        for ai in range(num_angles):
            M, new_w, new_h = rot_params[ai]
            rotated = rotate_frame(frame, M, new_w, new_h)
            buffers[ai].append(rotated)
            if frame_idx >= 2:
                edges = process_three_frames(buffers[ai][0], buffers[ai][1], buffers[ai][2])
                edge_sum[ai] += np.sum(edges[valid_masks[ai]])

    if tqdm is None:
        print(f"\r处理完成: {total_frames}/{total_frames} 帧")

    averages = np.divide(edge_sum, valid_counts,
                         out=np.zeros_like(edge_sum), where=valid_counts > 0)
    return angles, averages


def save_results(angles: np.ndarray, averages: np.ndarray, out_dir: str) -> None:
    csv_path = os.path.join(out_dir, 'angle_edge_average_data.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Angle(deg),EdgeAverage\n")
        for a, v in zip(angles, averages):
            f.write(f"{a:.0f},{v:.6f}\n")
    print(f"  数据文件: {csv_path}")


def plot_results(angles: np.ndarray, averages: np.ndarray, out_dir: str,
                 video_name: str = "") -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(angles, averages, 'b-o', markersize=3, linewidth=1.5, label='Edge Average')

    max_idx = int(np.argmax(averages))
    min_idx = int(np.argmin(averages))
    plt.plot(angles[max_idx], averages[max_idx], 'r*', markersize=12,
             label=f'Max: {averages[max_idx]:.4f} @ {angles[max_idx]:.0f}°')
    plt.plot(angles[min_idx], averages[min_idx], 'g*', markersize=12,
             label=f'Min: {averages[min_idx]:.4f} @ {angles[min_idx]:.0f}°')

    plt.xlabel('Rotation Angle (deg)', fontsize=12)
    plt.ylabel('Edge Detection Average', fontsize=12)
    title = 'Rotation Angle vs Edge Detection Average'
    if video_name:
        title += f' - {video_name}'
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    for name in ('png', 'pdf'):
        path = os.path.join(out_dir, f'angle_vs_edge_average.{name}')
        plt.savefig(path, dpi=150)
        print(f"  曲线图: {path}")
    plt.close()


def plot_multi_video_comparison(all_results: List[dict], out_dir: str) -> None:
    plt.figure(figsize=(14, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for idx, result in enumerate(all_results):
        plt.plot(result['angles'], result['averages'],
                 color=colors[idx % len(colors)], marker=markers[idx % len(markers)],
                 markersize=2, linewidth=1.2, label=result['name'], alpha=0.85)

    plt.xlabel('Rotation Angle (deg)', fontsize=12)
    plt.ylabel('Edge Detection Average', fontsize=12)
    plt.title('Multi-Video Comparison: Rotation Angle vs Edge Detection Average', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc='best', ncol=2)

    for name in ('png', 'pdf'):
        path = os.path.join(out_dir, f'multi_video_comparison.{name}')
        plt.savefig(path, dpi=150)
        print(f"  对比图: {path}")
    plt.close()


def save_multi_video_summary(all_results: List[dict], out_dir: str) -> None:
    csv_path = os.path.join(out_dir, 'multi_video_summary.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        headers = ["Angle(deg)"] + [r['name'] for r in all_results]
        f.write(','.join(headers) + '\n')
        for i in range(len(all_results[0]['angles'])):
            row = [f"{all_results[0]['angles'][i]:.0f}"]
            row += [f"{r['averages'][i]:.6f}" for r in all_results]
            f.write(','.join(row) + '\n')
    print(f"  汇总CSV: {csv_path}")


def mode_angle_scan(video_input: str | None = None,
                    start_angle: float = 0.0, end_angle: float = 180.0, step: float = 0.5):
    """对指定视频或 data/ 下所有视频进行角度扫描。"""
    videos_to_process: list[tuple[str, str]] = []

    if video_input is None:
        videos_to_process = scan_video_files()
    elif os.path.isdir(video_input):
        input_dir = os.path.abspath(video_input)
        png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
        if png_files:
            name = os.path.basename(input_dir)
            if name.endswith('_frames'):
                name = name[:-7]
            videos_to_process = [(name, input_dir)]
        else:
            for ext in ('*.mp4', '*.avi'):
                for f in sorted(glob.glob(os.path.join(input_dir, ext))):
                    n = os.path.splitext(os.path.basename(f))[0]
                    videos_to_process.append((n, f))
    elif os.path.isfile(video_input):
        n = os.path.splitext(os.path.basename(video_input))[0]
        videos_to_process = [(n, video_input)]
    else:
        # 尝试按名称匹配
        all_v = scan_video_files()
        filtered = [(n, p) for n, p in all_v if video_input.lower() in n.lower()]
        if filtered:
            videos_to_process = filtered
        else:
            candidate = os.path.join(DATA_DIR, video_input)
            for ext in ('.mp4', '.avi'):
                fp = candidate + ext
                if os.path.isfile(fp):
                    videos_to_process = [(video_input, fp)]
                    break
            if not videos_to_process:
                print(f"  ✗ 未找到匹配的视频: {video_input}")
                sys.exit(1)

    if not videos_to_process:
        print("  ✗ 没有需要处理的视频")
        sys.exit(1)

    scan_out_dir = os.path.join(OUTPUT_DIR, 'angle_edge_scan')
    os.makedirs(scan_out_dir, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          模式2：角度-边缘平均值扫描实验                       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  扫描范围: {start_angle:.0f}° ~ {end_angle:.0f}°, 步长 {step}°")
    print(f"  待处理视频: {len(videos_to_process)} 个")

    all_results: list[dict] = []
    for idx, (vname, vpath) in enumerate(videos_to_process, 1):
        print(f"\n{'=' * 60}")
        print(f"  处理进度: [{idx}/{len(videos_to_process)}] {vname}")
        print(f"{'=' * 60}")

        video_out = os.path.join(scan_out_dir, vname)
        os.makedirs(video_out, exist_ok=True)

        if not os.path.isfile(vpath) and not os.path.isdir(vpath):
            print(f"  ✗ 找不到: {vpath}")
            continue

        print(f"  ◆ 读取视频...")
        try:
            arr = mp4_to_grayscale_array(vpath)
        except Exception as e:
            print(f"  ✗ 读取失败: {e}")
            continue
        print(f"    数组: {arr.shape}")

        num_angles = int((end_angle - start_angle) / step) + 1
        print(f"  ◆ 角度扫描 ({num_angles} 个角度)...")
        angles, averages = scan_angles(arr, start_angle, end_angle, step)
        del arr

        max_v, min_v = averages.max(), averages.min()
        max_a = angles[np.argmax(averages)]
        min_a = angles[np.argmin(averages)]

        print()
        print("╔══════════════════════════════════════════╗")
        print(f"║  最大值: {max_v:8.6f} @ {max_a:5.0f}°  ║")
        print(f"║  最小值: {min_v:8.6f} @ {min_a:5.0f}°  ║")
        print("╚══════════════════════════════════════════╝")

        save_results(angles, averages, video_out)
        plot_results(angles, averages, video_out, video_name=vname)
        all_results.append({'name': vname, 'angles': angles, 'averages': averages})

    if len(all_results) >= 2:
        print("\n生成多视频对比叠加图...")
        plot_multi_video_comparison(all_results, scan_out_dir)
        save_multi_video_summary(all_results, scan_out_dir)

    print(f"\n处理完成！输出目录: {scan_out_dir}")

# ============================================================
# 入口
# ============================================================
def print_usage():
    print(__doc__)
    print()
    print("用法:")
    print("  python run.py                    # 生成+扫描（默认）")
    print("  python run.py generate           # 仅生成测试视频")
    print("  python run.py scan               # 仅扫描已存在的视频")
    print()
    print("示例:")
    print("  python run.py                           # 一键生成+扫描")
    print("  python run.py generate                  # 仅生成测试视频")
    print("  python run.py scan                      # 仅扫描 data/ 下已有视频")


def main():
    mode = sys.argv[1] if len(sys.argv) >= 2 else 'all'

    if mode in ('-h', '--help', 'help'):
        print_usage()
        return

    if mode == 'generate':
        # 仅生成
        generate_snowflake_video(DATA_DIR)
        return

    if mode == 'scan':
        # 仅扫描 data/ 下已有视频
        mode_angle_scan(None, 0.0, 180.0, 0.5)
        return

    # 默认：生成 + 扫描
    print("=" * 60)
    print("步骤1：生成测试视频")
    print("=" * 60)
    frames_dir = generate_snowflake_video(DATA_DIR)
    print()

    print("=" * 60)
    print("步骤2：角度扫描")
    print("=" * 60)
    mode_angle_scan(frames_dir, 0.0, 180.0, 0.5)


if __name__ == "__main__":
    main()
