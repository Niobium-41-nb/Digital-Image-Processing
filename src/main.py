"""
============================================================
主程序 —— 基于3D卷积的运动感知与边缘检测流水线
============================================================

本程序将边缘检测视为 **3D卷积核 (T, 3, 3) 对 3D矩阵 (frames, H, W) 的卷积** 操作。

核心设计：
  传统的"时间运动模糊 + 空间边缘检测"是分离的两步操作。
  本程序将时间平滑核与空间边缘检测核通过 **外积 (outer product)**
  组合为统一的 3D 卷积核 (T, 3, 3)，一次卷积同时完成时间平滑和空间边缘检测。

主要流程如下：

  1. 自动扫描 data/ 目录下的视频文件（或通过命令行参数指定）
  2. 处理阶段：
         a) 图像放大 —— 将每帧图像放大 SCALE_FACTOR 倍
         b) 3D卷积边缘检测 —— 使用 3D 卷积核 (T, 3, 3) 对视频做真正的3D卷积，
            同时完成时间平滑和空间方向边缘检测（8方向）
         c) 帧间差分 —— 相邻帧差分绝对值，反映运动强度
         d) 最大池化还原 —— 将图像大小还原到原始尺寸
         e) 输出灰度视频 —— 各方向边缘检测结果及帧间差分分别输出为独立灰度视频

配置方式：
  卷积核大小（时间维度T）、放大倍数等参数通过 .env 文件配置。
  视频文件通过命令行参数或自动扫描 data/ 目录选择。

依赖模块：
  - src/config.py           : 配置加载
  - src/video_selector.py   : 视频文件扫描
  - src/video_converter.py  : MP4 与 NumPy 数组转换
  - src/temporal_convolver.py : 图像放大
  - src/edge_process.py     : 3D卷积边缘检测核与流水线处理

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
        python src/main.py [video_path] [temporal_min] [temporal_max]

    返回:
        (video_path, temporal_min, temporal_max)
    """
    video_path: str | None = None
    temporal_min: int | None = None
    temporal_max: int | None = None

    if len(sys.argv) >= 2:
        video_path = sys.argv[1]
    if len(sys.argv) >= 3:
        temporal_min = int(sys.argv[2])
    if len(sys.argv) >= 4:
        temporal_max = int(sys.argv[3])

    return video_path, temporal_min, temporal_max


# ============================================================
# 加载配置
# ============================================================
config = get_config()
CONVOLUTION_SIZE = config['CONVOLUTION_SIZE']
SCALE_FACTOR = config['SCALE_FACTOR']
POOL_SIZE = config['POOL_SIZE']

# ============================================================
# 第一步：选择视频
# ============================================================
video_arg, cli_conv_min, cli_conv_max = parse_args()

selected_video_name, selected_video_path = select_video(video_arg)
if selected_video_name is None:
    exit(1)

print(f"\n已选择视频: {selected_video_name}")
print(f"视频路径: {selected_video_path}")

# ============================================================
# 第二步：确定 3D 卷积核时间维度 T
# ============================================================
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
print(f"帧间差分: 启用")

# 读取视频
arr = mp4_to_grayscale_array(selected_video_path)

# ========== 将原始视频复制到 output 目录 ==========
video_output_dir = f"output/{selected_video_name}"
os.makedirs(video_output_dir, exist_ok=True)
original_video_output_path = os.path.join(video_output_dir, f"{selected_video_name}.mp4")
if not os.path.exists(original_video_output_path):
    shutil.copy2(selected_video_path, original_video_output_path)
    print(f"\n原始视频已复制到: {original_video_output_path}")
else:
    print(f"\n原始视频已存在: {original_video_output_path}，跳过复制")

# 遍历3D卷积核的时间维度大小 T
for temporal_size in temporal_sizes:
    output_dir = f"output/{selected_video_name}/{temporal_size}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"3D卷积核时间维度 T={temporal_size} | 放大倍数: {SCALE_FACTOR}x | 池化大小: {POOL_SIZE}x{POOL_SIZE}")
    print(f"{'=' * 70}")

    # ========== 图像放大（如果需要） ==========
    if SCALE_FACTOR > 1:
        print(f"\n>>> 开始图像放大 ({SCALE_FACTOR}x)")
        # 放大时不需要时间卷积，只做放大
        # 创建一个单位时间核（长度为1，不做时间平滑，因为3D核会处理）
        identity_kernel = np.ones((1, 1, 1), dtype=np.float64)
        processed_arr = scale_and_convolve(arr, SCALE_FACTOR, identity_kernel)
    else:
        processed_arr = arr.copy()

    # ========== 3D卷积边缘检测处理 ==========
    # process_motion_blurred_array 内部使用 3D 卷积核 (T, 3, 3)
    # 对视频做真正的3D卷积，一次调用同时完成时间平滑和空间边缘检测
    print(f"\n>>> 开始3D卷积边缘检测处理 (T={temporal_size})")
    process_motion_blurred_array(
        processed_arr, output_dir, suffix="",
        pool_size=POOL_SIZE,
        enable_diff=True,
        temporal_size=temporal_size
    )

    # 释放内存
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
