"""
============================================================
全流程脚本 —— 从视频生成到形状检测的端到端管线
============================================================

流程:
  1. 检查/生成测试视频（如需要）
  2. 对每个视频运行形状检测
  3. 生成可视化:
     - 形状掩码叠加图 (overlay.png)
     - 运动方向彩色图 (direction_map.png)
     - 形状掩码 (shape_mask.png)
     - 掩码视频 (shape_mask_video.mp4)
     - 叠加视频 (overlay_video.mp4)
  4. 输出检测结果汇总

用法:
  python run_full_pipeline.py                          # 处理 data/ 下所有视频
  python run_full_pipeline.py dual_scroll              # 处理匹配的视频
  python run_full_pipeline.py --generate               # 先生成新视频再处理
  python run_full_pipeline.py -b 4                     # 指定块大小
============================================================
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.shape_detector import detect_shape, create_overlay_frame, create_direction_map
from src.video_converter import mp4_to_grayscale_array, gray_array_to_mp4


# ============================================================
# 单视频完整处理
# ============================================================

def process_one_video(video_path: str, output_base: str,
                      block_size: int = None) -> dict:
    """处理单个视频的完整管线。

    参数:
        video_path: 视频文件路径
        output_base: 输出根目录
        block_size: 手动指定块大小（None=自动检测）

    返回:
        检测结果字典，失败返回 None
    """
    video_name = Path(video_path).stem
    output_dir = os.path.join(output_base, video_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# 处理: {video_name}")
    print(f"{'#'*70}")

    # ── 形状检测 ──
    result = detect_shape(video_path, block_size=block_size,
                          output_dir=output_dir)
    if result is None:
        print(f"  [FAIL] 检测失败")
        return None

    mask = result['mask']
    block_labels = result['block_labels']
    bs = result['block_size']

    # Read video for visualizations
    video = mp4_to_grayscale_array(video_path, verbose=False)
    F, H, W = video.shape

    # ── 叠加视频 ──
    print(">>> 生成叠加视频...")
    _save_overlay_video(video, mask, os.path.join(output_dir, 'overlay_video.mp4'))

    # ── 方向图视频 ──
    dmap = create_direction_map(block_labels, bs, (H, W))
    dmap_gray = cv2.cvtColor(dmap, cv2.COLOR_BGR2GRAY)
    dmap_video = np.tile(dmap_gray[np.newaxis, :, :], (min(F, 30), 1, 1))
    gray_array_to_mp4(dmap_video,
                      os.path.join(output_dir, 'direction_map_video.mp4'),
                      fps=10, verbose=False)

    # ── 报告 ──
    _save_report(output_dir, video_name, result)

    print(f"  输出目录: {output_dir}")
    print(f"  [OK] Detection complete")

    return result


def _save_overlay_video(video, mask, output_path, max_frames=60):
    """保存叠加彩色视频。"""
    F, H, W = video.shape
    max_frames = min(F, max_frames)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 10, (W, H), isColor=True)

    for i in range(max_frames):
        frame = video[i]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        overlay = frame_bgr.copy()
        overlay[mask > 0] = (0, 255, 0)
        result = cv2.addWeighted(frame_bgr, 0.65, overlay, 0.35, 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        writer.write(result)

    writer.release()


def _save_report(output_dir, video_name, result):
    """保存检测报告。"""
    mask = result['mask']
    H, W = mask.shape
    shape_pixels = int(np.sum(mask > 0))
    shape_pct = 100 * shape_pixels / (H * W)

    report_path = os.path.join(output_dir, 'detection_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Shape Detection Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Video: {video_name}\n")
        f.write(f"Frame size: {W} x {H}\n")
        f.write(f"Block size: {result['block_size']} px\n")
        f.write(f"Background motion direction: {result['bg_direction']}\n")
        f.write(f"Shape motion direction: {result['shape_direction']}\n")
        f.write(f"Shape area: {shape_pixels} pixels ({shape_pct:.1f}%)\n")
        f.write(f"Number of contours: {len(result['contours'])}\n")
        if result['contours']:
            f.write(f"Largest contour vertices: {len(result['contours'][0])}\n")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mosaic Video Shape Detection Pipeline")
    parser.add_argument('video_filter', nargs='?', default=None,
                        help='Video name filter keyword')
    parser.add_argument('--generate', '-g', action='store_true',
                        help='Generate test videos first')
    parser.add_argument('--output', '-o', default='output/pipeline_results',
                        help='Output directory')
    parser.add_argument('--block-size', '-b', type=int, default=None,
                        help='Manual block size (default: auto-detect)')
    parser.add_argument('--max-videos', '-n', type=int, default=5,
                        help='Maximum number of videos to process')
    args = parser.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, 'data')
    output_base = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(output_base, exist_ok=True)

    # ── 生成视频 ──
    if args.generate:
        print(">>> 生成测试视频...")
        from generators.dual_scrolling import generate_dual_scrolling_texture_video
        out = os.path.join(data_dir, 'dual_scroll_test.mp4')
        if not os.path.exists(out):
            generate_dual_scrolling_texture_video(out, square_size=4,
                                                   width=640, height=640,
                                                   duration=5)
            print(f"  已生成: {out}")

    # ── 扫描已有视频 ──
    videos_to_process = []
    if os.path.isdir(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.mp4'):
                name = Path(f).stem
                if args.video_filter and args.video_filter.lower() not in name.lower():
                    continue
                vpath = os.path.join(data_dir, f)
                videos_to_process.append(vpath)

    if not videos_to_process:
        print("No video files found!")
        print("Please put .mp4 files in data/ or use --generate")
        sys.exit(1)

    # 只处理 dual_scroll / diagonal / waterfall / snowfall 类型的视频
    # （固定形状内纹理滚动，而非形状整体移动）
    target_keywords = ['dual_scroll', 'diagonal_motion', 'waterfall', 'snowfall_tiny']
    exclude_keywords = ['combo_', 'shape_', 'circular', 'spiral', 'pulsing',
                        'figure_eight', 'perfect_camouflage', 'snowfall.mp4']
    if args.video_filter is None:
        filtered = [v for v in videos_to_process
                    if any(kw in Path(v).stem.lower() for kw in target_keywords)]
        # 排除非目标视频
        filtered = [v for v in filtered
                    if not any(kw in Path(v).stem.lower() for kw in exclude_keywords)]
        if filtered:
            videos_to_process = filtered
        videos_to_process = videos_to_process[:args.max_videos]

    print(f"\n{'='*70}")
    print(f"Mosaic Video Shape Detection Pipeline")
    print(f"{'='*70}")
    print(f"Videos to process: {len(videos_to_process)}")
    for v in videos_to_process:
        print(f"  - {os.path.basename(v)}")
    print(f"Output: {os.path.abspath(output_base)}")
    print(f"Block size: {'auto-detect' if args.block_size is None else args.block_size}")

    # ── 逐个处理 ──
    results = {}
    for vpath in videos_to_process:
        r = process_one_video(vpath, output_base, block_size=args.block_size)
        if r:
            results[Path(vpath).stem] = r

    # ── 汇总 ──
    print(f"\n{'='*70}")
    print(f"Pipeline Complete! {len(results)}/{len(videos_to_process)} videos processed")
    print(f"{'='*70}")

    # 表格输出
    print(f"\n{'Video':<40} {'Shape Area':>12} {'BG Dir':>10} {'Shape Dir':>10} {'Status':>8}")
    print("-" * 85)
    for name, r in results.items():
        mask = r['mask']
        shape_pct = 100 * np.sum(mask > 0) / mask.size
        bg = f"({r['bg_direction'][0]:.0f},{r['bg_direction'][1]:.0f})"
        sd = f"({r['shape_direction'][0]:.0f},{r['shape_direction'][1]:.0f})"
        status = "OK" if shape_pct > 1 else "NONE"
        print(f"{name:<40} {shape_pct:>11.1f}% {bg:>10} {sd:>10} {status:>8}")

    # 汇总报告
    summary_path = os.path.join(output_base, 'pipeline_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Pipeline Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Videos processed: {len(results)}/{len(videos_to_process)}\n\n")
        for name, r in results.items():
            mask = r['mask']
            shape_pct = 100 * np.sum(mask > 0) / mask.size
            f.write(f"{name}:\n")
            f.write(f"  Block size: {r['block_size']} px\n")
            f.write(f"  BG direction: {r['bg_direction']}\n")
            f.write(f"  Shape direction: {r['shape_direction']}\n")
            f.write(f"  Shape area: {shape_pct:.1f}%\n")
            f.write(f"  Contours: {len(r['contours'])}\n\n")

    print(f"\nSummary saved to: {os.path.abspath(summary_path)}")
    print(f"All results in: {os.path.abspath(output_base)}")

    # 打开输出文件夹
    os.startfile(os.path.abspath(output_base))


if __name__ == '__main__':
    main()
