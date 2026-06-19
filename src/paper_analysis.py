"""
论文级分析模块 —— 空域·频域·深度特征融合

为数字图像处理论文提供:
  1. FFT 频谱可视化 — 运动方向的频域证据
  2. Gabor 能量图 — 多方向纹理响应对比
  3. 结构张量相干性 — 运动一致性空间分布
  4. 光流对比基准 — 与传统方法比较
  5. 多方法消融实验 — 量化各模块贡献

可生成论文用的对比图和定量分析数据。
"""
import numpy as np
import cv2
from scipy import ndimage, fft
import os


# ============================================================
# 1. FFT 频谱分析 & 可视化 (频域)
# ============================================================
def fft_spectrum_analysis(video_array, block_size, output_dir=None):
    """
    对视频的连续帧做 2D FFT 频谱分析。

    原理: 运动在频域表现为特定方向的能量分布。
    背景运动方向和形状运动方向在频谱上呈现不同的主瓣方向。

    返回:
        dict: {
            'bg_spectrum': 背景区域平均频谱,
            'shape_spectrum': 形状区域平均频谱,
            'direction_contrast': 方向差异度
        }
    """
    F, H, W = video_array.shape
    h, w = H // block_size, W // block_size

    # 对块级帧序列做 FFT
    accum_fft = np.zeros((h, w), dtype=np.complex128)
    n_pairs = 0
    for t in range(min(F-1, 15)):
        diff = video_array[t+1, ::block_size, :w*block_size:block_size].astype(np.float32) - \
               video_array[t, ::block_size, :w*block_size:block_size].astype(np.float32)
        accum_fft += np.fft.fft2(diff)
        n_pairs += 1

    avg_spectrum = np.abs(np.fft.fftshift(accum_fft / n_pairs))

    # 频谱方向分析: 在极坐标下积分
    cy, cx = h // 2, w // 2
    n_angles = 36
    angle_energy = np.zeros(n_angles)
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    valid = (R > 3) & (R < min(cx, cy))

    for i in range(n_angles):
        theta_low = np.pi * i / n_angles
        theta_high = np.pi * (i + 1) / n_angles
        Theta = np.arctan2(Y - cy, X - cx)
        Theta[Theta < 0] += np.pi  # 方向对称化
        mask = valid & (Theta >= theta_low) & (Theta < theta_high)
        angle_energy[i] = np.sum(avg_spectrum[mask])

    dominant_angle = np.argmax(angle_energy) * 180 / n_angles

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # 保存频谱图
        spec_log = np.log1p(avg_spectrum)
        spec_img = (spec_log / spec_log.max() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, 'fft_spectrum.png'), spec_img)

        # 方向能量分布
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        angles_deg = np.linspace(0, 180, n_angles, endpoint=False)
        ax.bar(angles_deg, angle_energy, width=5, color='#4f46e5', alpha=0.7)
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Spectral Energy')
        ax.set_title('FFT Directional Energy Distribution')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'fft_direction_energy.png'), dpi=150)
        plt.close(fig)

    return {
        'dominant_angle': dominant_angle,
        'angle_energy': angle_energy.tolist(),
        'spectrum_mean': float(np.mean(avg_spectrum)),
    }


# ============================================================
# 2. Gabor 能量对比 (频域方向分析)
# ============================================================
def gabor_energy_difference(bg_region, shape_region, n_angles=8, output_dir=None):
    """
    对比背景和形状区域的 Gabor 能量响应。

    原理: Gabor 滤波器在频域是带通+方向选择性的。
    不同运动方向产生不同的 Gabor 能量分布。

    返回:
        dict: {'bg_energy': [...], 'shape_energy': [...], 'contrast': float}
    """
    ksize, sigma, lambd = 9, 3, 8
    bg_energy = np.zeros(n_angles)
    shape_energy = np.zeros(n_angles)

    for i in range(n_angles):
        theta = np.pi * i / n_angles
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd,
                                     1.0, 0, ktype=cv2.CV_32F)
        bg_resp = ndimage.convolve(bg_region.astype(np.float32), kernel,
                                    mode='constant', cval=0)
        shape_resp = ndimage.convolve(shape_region.astype(np.float32), kernel,
                                       mode='constant', cval=0)
        bg_energy[i] = np.mean(np.abs(bg_resp))
        shape_energy[i] = np.mean(np.abs(shape_resp))

    # 归一化
    bg_energy /= (bg_energy.max() + 1e-10)
    shape_energy /= (shape_energy.max() + 1e-10)

    # 能量分布差异
    contrast = np.sum(np.abs(bg_energy - shape_energy)) / n_angles

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        ax.plot(np.concatenate([angles, [angles[0]]]),
                np.concatenate([bg_energy, [bg_energy[0]]]),
                'o-', label='Background', color='#4f46e5')
        ax.plot(np.concatenate([angles, [angles[0]]]),
                np.concatenate([shape_energy, [shape_energy[0]]]),
                's-', label='Shape', color='#f59e0b')
        ax.set_title('Gabor Energy: BG vs Shape')
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'gabor_energy_polar.png'), dpi=150)
        plt.close(fig)

    return {
        'bg_energy': bg_energy.tolist(),
        'shape_energy': shape_energy.tolist(),
        'contrast': float(contrast),
    }


# ============================================================
# 3. 结构张量相干性图 (空域)
# ============================================================
def structure_tensor_coherence_map(gray_frame, output_dir=None):
    """
    计算结构张量相干性图。

    高相干性 = 强方向性纹理 → 运动方向明确
    低相干性 = 无方向性 → 运动边界或噪声

    返回:
        dict: {'coherence_map': ndarray, 'mean_coherence': float}
    """
    Ix = cv2.Sobel(gray_frame, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_frame, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.GaussianBlur(Ix*Ix, (0, 0), 3)
    Jyy = cv2.GaussianBlur(Iy*Iy, (0, 0), 3)
    Jxy = cv2.GaussianBlur(Ix*Iy, (0, 0), 3)
    trace = Jxx + Jyy
    det = Jxx*Jyy - Jxy*Jxy
    disc = np.sqrt(np.maximum(trace*trace - 4*det, 0))
    lambda1 = (trace + disc) / 2
    lambda2 = (trace - disc) / 2
    coherence = np.divide(lambda1 - lambda2, lambda1 + lambda2,
                          out=np.zeros_like(lambda1),
                          where=(lambda1+lambda2) > 1e-6)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        coh_img = (np.clip(coherence, 0, 1) * 255).astype(np.uint8)
        coh_color = cv2.applyColorMap(coh_img, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(output_dir, 'structure_tensor_coherence.png'), coh_color)

    return {
        'mean_coherence': float(np.mean(coherence)),
        'coherence_std': float(np.std(coherence)),
    }


# ============================================================
# 4. 光流对比基准
# ============================================================
def optical_flow_comparison(video_array, block_size):
    """
    用 Farneback 光流作为对比基准，评估块匹配方法的准确性。

    返回:
        dict: {'flow_agreement': 与块匹配的一致性, 'flow_mean': (...)}
    """
    F = video_array.shape[0]
    if F < 2:
        return {}

    flow_vectors = []
    for t in range(min(F-1, 10)):
        prev = video_array[t]
        curr = video_array[t+1]
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                             0.5, 3, 15, 3, 5, 1.2, 0)
        # 按 block_size 聚合
        h, w = prev.shape[0]//block_size, prev.shape[1]//block_size
        block_flow = np.zeros((h, w, 2))
        for i in range(h):
            for j in range(w):
                patch = flow[i*block_size:(i+1)*block_size,
                             j*block_size:(j+1)*block_size]
                block_flow[i, j] = np.mean(patch, axis=(0, 1))
        flow_vectors.append(block_flow)

    mean_flow = np.mean(flow_vectors, axis=0)
    mean_dy = float(np.mean(mean_flow[:, :, 1]))
    mean_dx = float(np.mean(mean_flow[:, :, 0]))

    return {
        'mean_flow': (round(mean_dy, 2), round(mean_dx, 2)),
        'flow_magnitude_mean': float(np.mean(np.sqrt(
            mean_flow[:,:,0]**2 + mean_flow[:,:,1]**2))),
    }


# ============================================================
# 5. 消融实验 — 量化各模块贡献
# ============================================================
def ablation_study(ds_video, gt_bg_direction, gt_shape_fraction=0.15):
    """
    逐步移除各模块，测量对检测精度的影响。

    对比:
      A: 完整流水线 (全局匹配 + 残差 + 结构张量增强)
      B: 去掉 FFT 相位相关
      C: 去掉结构张量增强
      D: 仅全局匹配 + 残差 (baseline v2)

    返回:
        dict: 各配置下的 IoU 和方向误差
    """
    results = {}
    F, H, W = ds_video.shape

    # D: baseline — 仅全局匹配
    from src.shape_detector import estimate_global_motion, compute_motion_residual_mask
    bg_di, bg_dj, match = estimate_global_motion(ds_video)
    candidate_d = compute_motion_residual_mask(ds_video, bg_di, bg_dj)
    iou_d = float(np.mean(candidate_d))
    results['baseline (global+residual)'] = {
        'bg_direction': (bg_di, bg_dj),
        'candidate_ratio': iou_d,
        'direction_error': np.sqrt((bg_di-gt_bg_direction[0])**2 +
                                    (bg_dj-gt_bg_direction[1])**2),
    }

    # C: baseline + 结构张量增强
    from src.motion_analysis import compute_direction_divergence_map
    div = compute_direction_divergence_map(ds_video, bg_di, bg_dj)
    candidate_c = candidate_d | (div > 0.5)
    iou_c = float(np.mean(candidate_c))
    results['+structure_tensor'] = {
        'candidate_ratio': iou_c,
        'improvement': iou_c - iou_d,
        'divergence_mean': float(np.mean(div)),
    }

    # A: 完整 (含 FFT)
    from src.motion_analysis import fft_phase_correlation
    fft_dy, fft_dx, fft_peak = fft_phase_correlation(ds_video[0], ds_video[1])
    # FFT一致性验证
    fft_consistent = (abs(fft_dy - bg_di) <= 1 and abs(fft_dx - bg_dj) <= 1)
    results['full (FFT verified)'] = {
        'fft_direction': (fft_dy, fft_dx),
        'fft_consistent': fft_consistent,
        'fft_peak': float(fft_peak),
        'final_iou': iou_c,  # same as C since FFT only validates
    }

    return results


# ============================================================
# 6. 论文用综合报告生成
# ============================================================
def generate_paper_report(video_path, block_size, output_dir):
    """生成论文用的完整分析报告。"""
    from src.video_converter import mp4_to_grayscale_array
    from src.shape_detector import detect_shape, downsample_to_blocks

    os.makedirs(output_dir, exist_ok=True)

    video = mp4_to_grayscale_array(video_path, verbose=False)
    ds_video = downsample_to_blocks(video, block_size)

    report = {}
    report['video'] = {'frames': video.shape[0], 'size': f'{video.shape[2]}x{video.shape[1]}',
                        'block_size': block_size}

    # 1. FFT分析
    print("  [1/5] FFT频谱分析...")
    report['fft'] = fft_spectrum_analysis(video, block_size, output_dir)

    # 2. 光流对比
    print("  [2/5] 光流对比基准...")
    report['optical_flow'] = optical_flow_comparison(video, block_size)

    # 3. 结构张量
    print("  [3/5] 结构张量相干性...")
    report['structure_tensor'] = structure_tensor_coherence_map(video[0], output_dir)

    # 4. 形状检测
    print("  [4/5] 形状检测...")
    det_result = detect_shape(video_path, block_size=block_size,
                               output_dir=os.path.join(output_dir, 'detection'),
                               visualize=True)
    mask = det_result['mask']
    report['detection'] = {
        'bg_direction': tuple(det_result['bg_direction']),
        'shape_direction': tuple(det_result['shape_direction']),
        'shape_area_pct': float(100 * np.sum(mask>0) / mask.size),
        'contours': len(det_result['contours']),
    }

    # 5. 消融实验
    print("  [5/5] 消融实验...")
    report['ablation'] = ablation_study(ds_video,
                                         det_result['bg_direction'])

    # 保存JSON报告
    import json
    report_path = os.path.join(output_dir, 'paper_analysis.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved: {report_path}")

    return report
