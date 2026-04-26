"""
============================================================
主程序 —— 时间-空间联合卷积的运动感知与边缘检测流水线
============================================================

本程序实现了一套完整的视频运动分析与边缘检测流水线，主要流程如下：

  1. 生成阶段
     调用 generators/complex_motion.py 中的生成器，创建多种复杂运动模式
     的测试视频（斜向、圆周、螺旋、缩放脉冲、8字形运动）。

  2. 处理阶段
     对每个视频执行以下联合卷积处理：
       a) 时间维度运动模糊 —— 使用三维卷积核沿时间轴平滑，模拟运动拖影
       b) 空间边缘检测 —— 对模糊后的视频逐帧应用 0 度 / 60 度 / 120 度
          三个方向的 Prewitt 罗盘卷积核，检测不同方向的边缘
       c) 多通道融合 —— 将三方向边缘检测结果映射到 RGB 彩色通道

  3. 内存优化
     - 使用 float32 替代 float64，减少 50% 内存占用
     - 逐帧处理边缘检测，避免一次性加载所有结果
     - 主动释放中间结果并调用垃圾回收

依赖模块：
  - src/video_converter.py    : MP4 与 NumPy 数组的相互转换
  - src/temporal_convolver.py : 三维卷积核心算法
  - src/edge_process.py       : 方向边缘检测卷积核
  - generators/complex_motion.py : 复杂运动模式视频生成器

作者：数字图像处理课程项目
============================================================
"""

import os
import gc

import numpy as np
from scipy.ndimage import convolve

# ---------- 导入视频转换工具 ----------
# mp4_to_grayscale_array  : 读取 MP4 → 灰度三维数组 (frames, H, W)
# gray_array_to_mp4       : 灰度三维数组 → MP4
# two_gray_array_to_GB_mp4 : 两个灰度数组 → 彩色 MP4（G/B 通道）
# three_gray_array_to_RGB_mp4 : 三个灰度数组 → 彩色 MP4（R/G/B 通道）
from src.video_converter import mp4_to_grayscale_array, gray_array_to_mp4, two_gray_array_to_GB_mp4, three_gray_array_to_RGB_mp4

# ---------- 导入时间-空间卷积处理工具 ----------
# create_temporal_motion_blur : 创建时间维度的运动模糊卷积核
# apply_temporal_convolution  : 对三维数组执行三维卷积（时间+空间）
from src.temporal_convolver import create_temporal_motion_blur, apply_temporal_convolution

# ---------- 导入边缘处理专用工具 ----------
# CONVOLUTION_0DEG / _60DEG / _120DEG : 三方向 Prewitt 罗盘卷积核
from src.edge_process import (
    CONVOLUTION_0DEG,
    CONVOLUTION_60DEG,
    CONVOLUTION_120DEG,
)

# ============================================================
# 第一步：生成所有复杂运动模式的测试视频
# ============================================================

# 导入复杂运动模式视频生成器
from generators.complex_motion import (
    generate_diagonal_motion_video,      # 斜向运动
    generate_circular_motion_video,      # 圆周运动
    generate_spiral_motion_video,        # 螺旋运动
    generate_pulsing_zoom_video,         # 缩放脉冲运动
    generate_figure_eight_motion_video,  # 8字形运动
)

# 确保 data/ 目录存在，用于存放生成的测试视频
os.makedirs("data", exist_ok=True)

print("=" * 70)
print("开始生成复杂运动模式测试视频...")
print("=" * 70)

# 定义待生成的测试视频列表
# 键为运动模式名称，值为输出文件路径
test_videos = {
    "diagonal": "data/diagonal_motion.mp4",          # 斜向运动
    "circular": "data/circular_motion.mp4",          # 圆周运动
    "spiral": "data/spiral_motion.mp4",              # 螺旋运动
    "pulsing": "data/pulsing_zoom.mp4",              # 缩放脉冲
    "figure_eight": "data/figure_eight_motion.mp4",  # 8字形运动
}

# 遍历所有运动模式，若视频文件不存在则调用对应的生成器创建
for name, path in test_videos.items():
    if not os.path.exists(path):
        print(f"\n生成 {name} 运动视频...")
        if name == "diagonal":
            generate_diagonal_motion_video(path, square_size=4)
        elif name == "circular":
            generate_circular_motion_video(path, square_size=4)
        elif name == "spiral":
            generate_spiral_motion_video(path, square_size=4)
        elif name == "pulsing":
            generate_pulsing_zoom_video(path, square_size=4)
        elif name == "figure_eight":
            generate_figure_eight_motion_video(path, square_size=4)
    else:
        print(f"{path} 已存在，跳过生成")

# ============================================================
# 第二步：对所有视频进行时间-空间联合卷积处理
# ============================================================

# 所有待处理的视频列表（包括原有的双滚动视频）
# 双滚动视频由 generators/dual_scrolling.py 生成（需提前手动运行）
video_sources = {
    "dual_scroll": "data/dual_scroll_background_right_foreground_down.mp4",
    "diagonal": "data/diagonal_motion.mp4",
    "circular": "data/circular_motion.mp4",
    "spiral": "data/spiral_motion.mp4",
    "pulsing": "data/pulsing_zoom.mp4",
    "figure_eight": "data/figure_eight_motion.mp4",
}

# 显示可选视频列表
print("\n" + "=" * 50)
print("可选视频列表：")
print("=" * 50)
for idx, (name, path) in enumerate(video_sources.items(), 1):
    exists = "✓" if os.path.exists(path) else "✗"
    print(f"  {idx}. {name:15} {exists}")
print("=" * 50)

# 让用户选择要处理的视频
while True:
    try:
        video_choice = int(input("\n请选择要处理的视频编号 (1-6): "))
        if 1 <= video_choice <= len(video_sources):
            break
        print("输入无效，请输入 1-6 之间的数字")
    except ValueError:
        print("输入无效，请输入数字")

# 获取用户选择的视频
selected_video_name = list(video_sources.keys())[video_choice - 1]
selected_video_path = video_sources[selected_video_name]

# 检查视频是否存在
if not os.path.exists(selected_video_path):
    print(f"\n[错误] {selected_video_path} 不存在，请先生成该视频")
    exit(1)

print(f"\n已选择视频: {selected_video_name}")

# 让用户选择时间维度模糊核的帧数跨度
print("\n" + "=" * 50)
print("可选卷积核大小（时间维度模糊核的帧数跨度）：")
print("=" * 50)
print("  1. size=2  (轻度模糊)")
print("  2. size=3  (中度模糊)")
print("  3. size=4  (较强模糊)")
print("  4. size=5  (强模糊)")
print("  5. size=6  (极强模糊)")
print("=" * 50)

while True:
    try:
        size_choice = int(input("\n请选择卷积核大小编号 (1-6): "))
        if 1 <= size_choice <= 5:
            break
        print("输入无效，请输入 1-6 之间的数字")
    except ValueError:
        print("输入无效，请输入数字")

# 确定要使用的卷积核大小列表
if size_choice == 6:
    convolution_sizes = [2, 3, 4, 5, 6]
else:
    convolution_sizes = [size_choice + 1]  # 1->2, 2->3, etc.

print(f"\n已选择卷积核大小: {convolution_sizes}")
print(f"\n{'=' * 70}")
print(f"开始处理视频: {selected_video_name}")
print(f"{'=' * 70}")

# 读取视频为灰度三维数组
# arr 形状: (帧数, 高度, 宽度)，数据类型 uint8 (0~255)
arr = mp4_to_grayscale_array(selected_video_path)

# 遍历用户选择的卷积核大小
for convolution_size in convolution_sizes:
    # 创建输出目录：output/<视频名称>/<卷积核大小>/
    output_dir = f"output/{selected_video_name}/{convolution_size}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- 卷积核大小: {convolution_size} ---")

    # ========== (a) 创建时间维度运动模糊卷积核 ==========
    # 生成一个三维卷积核，形状为 (convolution_size, convolution_size, convolution_size)
    # 三个维度分别对应：时间帧数、空间高度、空间宽度
    # 由于 height=width=convolution_size，该核同时具有时间模糊和轻微空间模糊效果
    # 核内所有元素值相等，归一化后为 1/(convolution_size^3)
    convolution = create_temporal_motion_blur(convolution_size, convolution_size, convolution_size)

    # ========== (b) 执行时间维度卷积（运动模糊） ==========
    # 对原始视频沿时间轴做加权平均，模拟运动拖影效果
    # 运动速度越快/卷积核越大，模糊效果越明显
    processed_arr = apply_temporal_convolution(arr, convolution)

    # ========== (c) 三方向空间边缘检测 ==========
    # 对运动模糊后的视频逐帧应用 0 度 / 60 度 / 120 度边缘检测卷积核
    # 三个方向的结果将分别映射到 RGB 的 R、G、B 通道
    # 使用动态内存分配，逐帧处理以减少内存占用
    
    # 获取视频帧数
    num_frames = processed_arr.shape[0]
    height, width = processed_arr.shape[1], processed_arr.shape[2]
    
    # 预分配结果数组（使用 float32 减少内存）
    result_0deg = np.zeros((num_frames, height, width), dtype=np.float32)
    result_60deg = np.zeros((num_frames, height, width), dtype=np.float32)
    result_120deg = np.zeros((num_frames, height, width), dtype=np.float32)
    
    print(f"开始三方向边缘检测，共 {num_frames} 帧")
    for i in range(num_frames):
        frame = processed_arr[i].astype(np.float32)
        result_0deg[i] = convolve(frame, CONVOLUTION_0DEG, mode='constant', cval=0.0)
        result_60deg[i] = convolve(frame, CONVOLUTION_60DEG, mode='constant', cval=0.0)
        result_120deg[i] = convolve(frame, CONVOLUTION_120DEG, mode='constant', cval=0.0)
        
        if (i + 1) % 30 == 0:
            print(f"已处理帧: {i + 1}/{num_frames}")
    print("边缘检测完成")

    # ========== (d) 保存各处理结果为独立灰度视频 ==========
    # 保存运动模糊后的视频
    gray_array_to_mp4(processed_arr, f"{output_dir}/output.mp4")
    # 保存 0 度方向边缘检测结果
    gray_array_to_mp4(result_0deg, f"{output_dir}/output_edge_0deg.mp4")
    # 保存 60 度方向边缘检测结果
    gray_array_to_mp4(result_60deg, f"{output_dir}/output_edge_60deg.mp4")
    # 保存 120 度方向边缘检测结果
    gray_array_to_mp4(result_120deg, f"{output_dir}/output_edge_120deg.mp4")

    # ========== (e) RGB 三通道融合 ==========
    # 将三方向边缘检测结果直接映射到 RGB 彩色通道：
    #   R 通道 ← 0 度方向边缘检测
    #   G 通道 ← 60 度方向边缘检测
    #   B 通道 ← 120 度方向边缘检测
    three_gray_array_to_RGB_mp4(
        result_0deg, result_60deg, result_120deg,
        f"{output_dir}/output_RGB.mp4"
    )

    # 主动释放中间结果内存
    del result_0deg, result_60deg, result_120deg
    gc.collect()

# ========== 处理完成 ==========
print(f"\n{'=' * 70}")
print("所有视频处理完成！")
print(f"{'=' * 70}")
