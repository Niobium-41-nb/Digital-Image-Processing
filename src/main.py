"""
============================================================
主程序 —— 运动感知与边缘检测流水线
============================================================

支持两种处理方法：
  1. Prewitt 3D 卷积（默认）：时间平滑核 ⊗ 空间边缘核 → 3D 卷积核 (T, 3, 3)
  2. 3D FFT 频域分析（--method fft）：在 (f_x, f_y, f_t) 频域做方向滤波

主要流程如下：

  Prewitt 模式：
        a) 图像放大 —— 将每帧图像放大 SCALE_FACTOR 倍
        b) 3D卷积边缘检测 —— 对每个 T 值做 3D 卷积
        c) 最大池化还原 + 帧间差分 + 输出
  FFT 模式：
        a) 直接对原始分辨率的视频做 3D FFT
        b) 频域 8 方向扇形滤波 + 运动能量提取
        c) 重叠相加重建 + 输出

配置方式：
  参数通过 .env 文件配置，视频通过命令行参数或扫描 data/ 目录选择。

用法:
  python src/main.py [video_path] [temporal_min] [temporal_max] [--method fft|prewitt]

依赖模块：
  - src/config.py           : 配置加载
  - src/video_selector.py   : 视频文件扫描
  - src/video_converter.py  : MP4 与 NumPy 数组转换
  - src/temporal_convolver.py : 图像放大
  - src/edge_process.py     : Prewitt 3D 卷积 / FFT 分发
  - src/fft_3d_analysis.py  : 3D FFT 频域分析

作者：数字图像处理课程项目
============================================================
"""

import gc
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# 将项目根目录加入 sys.path，确保能导入同级 src 模块
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import get_config
from src.video_selector import scan_video_files
from src.video_converter import mp4_to_grayscale_array
from src.temporal_convolver import scale_and_convolve
from src.edge_process import process_motion_blurred_array


def select_video(video_arg: str | None = None) -> tuple:
    """选择待处理的视频文件。

    优先级：
      1. 命令行参数指定的视频路径
      2. data/ 目录下唯一视频
      3. 弹出交互式选择（多个视频时让用户输入编号）

    返回:
        (selected_name, selected_path)
    """
    # 1. 命令行参数指定
    if video_arg:
        video_path = Path(video_arg)
        if not video_path.exists():
            print(f"[错误] 指定的视频文件不存在: {video_arg}")
            return None, None
        return video_path.stem, str(video_path.resolve())

    # 2. 扫描 data/ 目录
    videos = scan_video_files()
    if not videos:
        print("[错误] data/ 目录下没有找到任何视频文件！")
        print("  请将 .mp4 文件放入 data/ 目录，或通过命令行参数指定视频路径。")
        return None, None

    if len(videos) == 1:
        name, path = videos[0]
        print(f"data/ 目录下仅有一个视频文件，自动选择: {name}")
        return name, path

    # 3. 多个视频，让用户选择
    print("\ndata/ 目录下找到多个视频文件，请选择：")
    for i, (name, path) in enumerate(videos, 1):
        print(f"  [{i}] {name}  ({path})")
    while True:
        try:
            choice = input(f"\n请输入编号 (1-{len(videos)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(videos):
                return videos[idx]
            print(f"  输入无效，请输入 1-{len(videos)} 之间的数字。")
        except ValueError:
            print("  输入无效，请输入数字。")


def parse_args() -> tuple:
    """解析命令行参数。

    用法:
        python src/main.py [video_path] [temporal_min] [temporal_max] [--method fft|prewitt]

    返回:
        (video_path, temporal_min, temporal_max, method_override)
    """
    video_path: str | None = None
    temporal_min: int | None = None
    temporal_max: int | None = None
    method_override: str | None = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--method':
            if i + 1 < len(args):
                method_override = args[i + 1]
                i += 2
            else:
                i += 1
        elif arg.startswith('--method='):
            method_override = arg.split('=', 1)[1]
            i += 1
        elif video_path is None:
            video_path = arg
            i += 1
        elif temporal_min is None:
            try:
                temporal_min = int(arg)
            except ValueError:
                video_path = arg
            i += 1
        elif temporal_max is None:
            try:
                temporal_max = int(arg)
            except ValueError:
                pass
            i += 1
        else:
            i += 1

    return video_path, temporal_min, temporal_max, method_override


# ============================================================
# 加载配置
# ============================================================
config = get_config()
CONVOLUTION_SIZE = config['CONVOLUTION_SIZE']
SCALE_FACTOR = config['SCALE_FACTOR']
POOL_SIZE = config['POOL_SIZE']
METHOD = config['METHOD']
FFT_TILE_SIZE = config['FFT_TILE_SIZE']

# ============================================================
# 第一步：选择视频 + 解析参数
# ============================================================
video_arg, cli_conv_min, cli_conv_max, method_override = parse_args()

# CLI 覆盖 .env 中的 METHOD
if method_override:
    METHOD = method_override
    print(f"[CLI] 方法覆盖: {METHOD}")

selected_video_name, selected_video_path = select_video(video_arg)
if selected_video_name is None:
    exit(1)

print(f"\n已选择视频: {selected_video_name}")
print(f"视频路径: {selected_video_path}")
print(f"处理方法: {METHOD.upper()}")

# ========== 将原始视频复制到 output 目录 ==========
video_output_dir = f"output/{selected_video_name}"
os.makedirs(video_output_dir, exist_ok=True)
original_video_output_path = os.path.join(video_output_dir, f"{selected_video_name}.mp4")
if not os.path.exists(original_video_output_path):
    shutil.copy2(selected_video_path, original_video_output_path)
    print(f"\n原始视频已复制到: {original_video_output_path}")
else:
    print(f"\n原始视频已存在: {original_video_output_path}，跳过复制")

# 读取视频
arr = mp4_to_grayscale_array(selected_video_path)

# ============================================================
# 第二步：按方法分流处理
# ============================================================
if METHOD == 'fft':
    # FFT 模式：3D 频域方向滤波
    print(f"\n{'=' * 70}")
    print(f"3D FFT 频域分析 | 分块大小: {FFT_TILE_SIZE} | 池化: {POOL_SIZE}x{POOL_SIZE}")
    print(f"{'=' * 70}")

    output_dir = f"output/{selected_video_name}/fft"
    os.makedirs(output_dir, exist_ok=True)

    process_motion_blurred_array(
        arr, output_dir, suffix="",
        pool_size=POOL_SIZE,
        enable_diff=True,
        method="fft",
        tile_size=FFT_TILE_SIZE,
    )

else:
    # Prewitt 模式：3D 卷积 (T, 3, 3) 边缘检测
    if cli_conv_min is not None and cli_conv_max is not None:
        conv_min = cli_conv_min
        conv_max = cli_conv_max
    else:
        conv_min = max(1, CONVOLUTION_SIZE - 1)
        conv_max = CONVOLUTION_SIZE

    temporal_sizes = list(range(conv_min, conv_max + 1))

    print(f"\n3D卷积核时间维度 T 区间: {conv_min} ~ {conv_max}")
    print(f"将依次处理 T = {temporal_sizes}")
    print(f"图像放大倍数: {SCALE_FACTOR}x")
    print(f"池化大小: {POOL_SIZE}x{POOL_SIZE}")

    # 遍历 3D 卷积核的时间维度大小 T
    for temporal_size in temporal_sizes:
        output_dir = f"output/{selected_video_name}/{temporal_size}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"3D卷积核时间维度 T={temporal_size} | 放大: {SCALE_FACTOR}x | 池化: {POOL_SIZE}x{POOL_SIZE}")
        print(f"{'=' * 70}")

        # 图像放大
        if SCALE_FACTOR > 1:
            print(f"\n>>> 开始图像放大 ({SCALE_FACTOR}x)")
            identity_kernel = np.ones((1, 1, 1), dtype=np.float64)
            processed_arr = scale_and_convolve(arr, SCALE_FACTOR, identity_kernel)
        else:
            processed_arr = arr.copy()

        # 3D 卷积边缘检测
        print(f"\n>>> 开始3D卷积边缘检测处理 (T={temporal_size})")
        process_motion_blurred_array(
            processed_arr, output_dir, suffix="",
            pool_size=POOL_SIZE,
            enable_diff=True,
            temporal_size=temporal_size,
            method="prewitt",
        )

        if SCALE_FACTOR > 1:
            del processed_arr
        gc.collect()

# ========== 处理完成 ==========
print(f"\n{'=' * 70}")
print("所有视频处理完成！")
print(f"{'=' * 70}")

# 自动打开 output 文件夹
output_path = os.path.abspath(f"output/{selected_video_name}")
print(f"\n正在打开输出文件夹: {output_path}")
os.startfile(output_path)
