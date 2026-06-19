"""
频域+空域运动分析模块

技术栈:
  - FFT 相位相关 (频域全局运动估计)
  - Gabor 滤波器组 (频域方向纹理分析)
  - 结构张量 (空域局部方向估计)
  - NCC/SSIM 块匹配 (空域鲁棒匹配)

用于提升马赛克视频中运动方向和形状检测的准确率。
"""
import numpy as np
from scipy import ndimage, signal, fft
import cv2


# ============================================================
# 1. FFT 相位相关 — 全局运动估计 (频域)
# ============================================================
def fft_phase_correlation(prev, curr):
    """
    用 FFT 相位相关计算两帧之间的全局位移。

    原理:
      - 对两帧做 2D FFT → 计算互功率谱 → 逆 FFT → 峰值位置 = 位移
      - 对平移、亮度变化不敏感，非常适合马赛克块运动估计

    返回:
        (dy, dx, peak_value) — 位移和峰值强度
    """
    # 加 Hanning 窗减少边界效应
    h, w = prev.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    win = wy * wx

    f1 = np.fft.fft2(prev.astype(np.float32) * win)
    f2 = np.fft.fft2(curr.astype(np.float32) * win)

    # 互功率谱归一化
    cross = f1 * np.conj(f2)
    cross_norm = cross / (np.abs(cross) + 1e-10)

    # 逆 FFT → 空间域互相关
    corr = np.fft.ifft2(cross_norm).real

    # 找峰值
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    peak_val = corr[peak_y, peak_x]

    # 转换为位移 (FFT 循环移位 → wrap-around)
    dy = peak_y if peak_y <= h // 2 else peak_y - h
    dx = peak_x if peak_x <= w // 2 else peak_x - w

    return dy, dx, float(peak_val)


# ============================================================
# 2. Gabor 滤波器组 — 方向纹理分析 (频域)
# ============================================================
def gabor_kernel(ksize, sigma, theta, lambd, gamma=1.0, psi=0):
    """生成 Gabor 滤波核。"""
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi,
                                 ktype=cv2.CV_32F)
    return kernel


def gabor_direction_response(gray_img, n_angles=8, ksize=9, sigma=3, lambd=8):
    """
    用 Gabor 滤波器组对图像做方向响应分析。

    返回:
        responses: (H, W, n_angles) — 每个方向上的响应强度
        best_angle: (H, W) — 每个像素的最佳响应方向 (弧度)
        best_response: (H, W) — 最佳响应强度
    """
    H, W = gray_img.shape
    img_f = gray_img.astype(np.float32)

    responses = np.zeros((H, W, n_angles), dtype=np.float32)

    for i in range(n_angles):
        theta = np.pi * i / n_angles
        kernel = gabor_kernel(ksize, sigma, theta, lambd)
        # 使用 scipy 的卷积
        resp = ndimage.convolve(img_f, kernel, mode='constant', cval=0)
        responses[:, :, i] = np.abs(resp)

    best_idx = np.argmax(responses, axis=2)
    best_response = np.max(responses, axis=2)
    best_angle = best_idx * np.pi / n_angles

    return responses, best_angle, best_response


def gabor_motion_direction_analysis(video_block, n_angles=8):
    """
    用 Gabor 滤波器分析视频块的运动方向。

    对连续帧做 Gabor 滤波，运动方向的改变反映在 Gabor 响应的变化上。
    在不同方向的响应中，运动方向的响应会随帧持续。

    返回:
        dominant_direction: 主要运动方向 (弧度)
        direction_confidence: 方向置信度
    """
    F = video_block.shape[0]
    if F < 3:
        return 0, 0

    # 对每帧做 Gabor 分解
    all_best_angles = []
    for t in range(F):
        _, angles, responses = gabor_direction_response(
            video_block[t], n_angles=n_angles)
        all_best_angles.append(angles.flatten())

    # 跨帧投票：最稳定的方向 = 运动方向
    all_angles = np.array(all_best_angles)
    # 量化到 n_angles 个 bin
    quantized = np.round(all_angles * n_angles / np.pi) % n_angles
    # 众数
    from scipy import stats
    mode_result = stats.mode(quantized, axis=0, keepdims=False)
    dominant_bin = mode_result.mode
    confidence = mode_result.count / F

    dominant_angle = dominant_bin * np.pi / n_angles
    mean_confidence = float(np.mean(confidence))

    return dominant_angle, mean_confidence


# ============================================================
# 3. 结构张量 — 局部方向估计 (空域)
# ============================================================
def structure_tensor(gray_img, sigma=2.0):
    """
    计算图像的结构张量。

    结构张量 = [[Ix^2, Ix*Iy], [Ix*Iy, Iy^2]] * G_sigma

    从张量可提取:
      - 主方向 (特征向量)
      - 方向一致性 / 相干性 (特征值比)

    返回:
        coherence: (H, W) — 方向一致性 (0=无方向, 1=强方向)
        orientation: (H, W) — 主方向角度 (弧度)
    """
    # 梯度
    Ix = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)

    # 张量分量
    Jxx = Ix * Ix
    Jyy = Iy * Iy
    Jxy = Ix * Iy

    # 高斯平滑
    Jxx = cv2.GaussianBlur(Jxx, (0, 0), sigma)
    Jyy = cv2.GaussianBlur(Jyy, (0, 0), sigma)
    Jxy = cv2.GaussianBlur(Jxy, (0, 0), sigma)

    # 特征值
    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    # λ1, λ2 = (trace ± sqrt(trace^2 - 4*det)) / 2
    discriminant = np.sqrt(np.maximum(trace*trace - 4*det, 0))

    lambda1 = (trace + discriminant) / 2
    lambda2 = (trace - discriminant) / 2

    # 相干性: (λ1-λ2)/(λ1+λ2) ∈ [0, 1]
    coherence = np.divide(lambda1 - lambda2, lambda1 + lambda2,
                          out=np.zeros_like(lambda1), where=(lambda1 + lambda2) > 1e-6)

    # 主方向: atan2(2*Jxy, Jxx-Jyy) / 2
    orientation = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)

    return coherence, orientation


def structure_tensor_motion_direction(frame_diff, n_frames=3):
    """
    对帧间差分图做结构张量分析。

    帧差分图在运动边界处有强梯度 → 结构张量反映运动方向。

    返回:
        dominant_orientation: 主导方向 (弧度)
        coherence: 方向一致性
    """
    coherence, orientation = structure_tensor(frame_diff, sigma=3.0)

    # 加权投票：方向按相干性加权
    valid = coherence > 0.15
    if not np.any(valid):
        return 0, 0

    # 构建方向直方图
    n_bins = 36
    hist = np.zeros(n_bins)
    ori_bins = np.floor((orientation[valid] % np.pi) * n_bins / np.pi).astype(int)
    ori_bins = np.clip(ori_bins, 0, n_bins - 1)

    for i in range(n_bins):
        hist[i] = np.sum(coherence[valid][ori_bins == i])

    dominant_bin = np.argmax(hist)
    dominant_ori = (dominant_bin + 0.5) * np.pi / n_bins
    conf = hist[dominant_bin] / (np.sum(hist) + 1e-6)

    return dominant_ori, float(conf)


# ============================================================
# 4. NCC 块匹配 — 鲁棒空域匹配
# ============================================================
def ncc_block_match(ds_prev, ds_curr, max_shift=1):
    """
    用归一化互相关 (NCC) 做块级运动估计。

    NCC 对比度不变，对彩色和灰度马赛克都鲁棒。

    返回:
        motion_dy, motion_dx: (H, W) — 每块的运动位移
        ncc_score: (H, W) — 匹配置信度
    """
    H, W = ds_prev.shape
    displacements = [(di, dj) for di in range(-max_shift, max_shift+1)
                     for dj in range(-max_shift, max_shift+1)]

    best_dy = np.zeros((H, W), dtype=np.int32)
    best_dx = np.zeros((H, W), dtype=np.int32)
    best_ncc = np.zeros((H, W), dtype=np.float32)

    # 对每个位移做 NCC
    for di, dj in displacements:
        shifted = np.roll(np.roll(ds_prev, di, axis=0), dj, axis=1)

        # 局部 NCC (3x3 窗口)
        prev_loc = shifted.astype(np.float32)
        curr_loc = ds_curr.astype(np.float32)

        # 3x3 均值
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        prev_mean = ndimage.convolve(prev_loc, kernel, mode='constant', cval=0)
        curr_mean = ndimage.convolve(curr_loc, kernel, mode='constant', cval=0)
        prev_diff = prev_loc - prev_mean
        curr_diff = curr_loc - curr_mean

        # NCC 分子: sum(prev_diff * curr_diff) in 3x3
        numer = ndimage.convolve(prev_diff * curr_diff, np.ones((3,3)),
                                  mode='constant', cval=0)
        # 分母
        prev_var = ndimage.convolve(prev_diff**2, np.ones((3,3)),
                                     mode='constant', cval=0)
        curr_var = ndimage.convolve(curr_diff**2, np.ones((3,3)),
                                     mode='constant', cval=0)
        denom = np.sqrt(np.maximum(prev_var * curr_var, 1e-10))

        ncc = numer / denom

        # 排除边界 (np.roll wrap-around)
        if di > 0: ncc[:di, :] = -1
        elif di < 0: ncc[di:, :] = -1
        if dj > 0: ncc[:, :dj] = -1
        elif dj < 0: ncc[:, dj:] = -1

        # 更新最佳匹配
        better = ncc > best_ncc
        best_dy[better] = di
        best_dx[better] = dj
        best_ncc[better] = ncc[better]

    return best_dy, best_dx, best_ncc


# ============================================================
# 5. 综合运动估计 — 融合频域+空域
# ============================================================
def fused_motion_estimate(ds_video, max_frames=20):
    """
    融合 FFT 相位相关 + NCC 块匹配的运动估计。

    流程:
      1. FFT 相位相关 → 全局运动方向 (频域)
      2. NCC 块匹配 → 每块运动 + 置信度 (空域)
      3. 结构张量 → 运动一致性验证 (空域)
      4. 融合: 全局方向投票 + 局部 NCC 细化

    返回:
        bg_di, bg_dj: 背景运动方向 (块坐标)
        confidence: 全局置信度
        block_labels: (H, W) — 0=背景, 1=形状候选
    """
    F, H, W = ds_video.shape
    max_frames = min(max_frames, F - 1)

    # 1. FFT 相位相关 (频域全局估计)
    fft_shifts = []
    for t in range(min(F-1, max_frames)):
        dy, dx, peak = fft_phase_correlation(ds_video[t], ds_video[t+1])
        if abs(dy) <= 1 and abs(dx) <= 1:  # 只取合理位移
            fft_shifts.append((dy, dx, peak))

    if fft_shifts:
        # 投票找最频繁的位移
        from collections import Counter
        shift_counts = Counter((s[0], s[1]) for s in fft_shifts)
        (bg_di, bg_dj), _ = shift_counts.most_common(1)[0]
    else:
        bg_di, bg_dj = 0, 0

    # 2. NCC 块匹配 (空域局部细化)
    ncc_dy = np.zeros((H, W), dtype=np.int32)
    ncc_dx = np.zeros((H, W), dtype=np.int32)
    ncc_conf = np.zeros((H, W), dtype=np.float32)
    n_frames_matched = 0

    for t in range(min(F-1, max_frames)):
        dy, dx, conf = ncc_block_match(ds_video[t], ds_video[t+1])
        ncc_dy += dy; ncc_dx += dx; ncc_conf += conf
        n_frames_matched += 1

    if n_frames_matched > 0:
        ncc_dy = np.round(ncc_dy / n_frames_matched).astype(np.int32)
        ncc_dx = np.round(ncc_dx / n_frames_matched).astype(np.int32)
        ncc_conf /= n_frames_matched

    # 3. 方向直方图 → 找两个主要方向 (背景 + 形状)
    # 只取 NCC 置信度较高的块
    reliable = ncc_conf > 0.25
    if np.sum(reliable) < 100:
        return bg_di, bg_dj, 0.0, np.zeros((H, W), dtype=np.int32)

    # 收集可靠块的运动向量
    vecs = np.column_stack([ncc_dy[reliable].flatten(),
                             ncc_dx[reliable].flatten()]).astype(np.float32)

    # 简易 2-means
    rng = np.random.RandomState(0)
    c0 = vecs[rng.randint(0, len(vecs))]
    dists = np.sum((vecs - c0)**2, axis=1)
    c1 = vecs[np.argmax(dists)]
    for _ in range(10):
        d0 = np.sum((vecs - c0)**2, axis=1)
        d1 = np.sum((vecs - c1)**2, axis=1)
        a = (d0 <= d1).astype(int)
        nc0 = vecs[a==0].mean(axis=0) if np.any(a==0) else c0
        nc1 = vecs[a==1].mean(axis=0) if np.any(a==1) else c1
        if np.allclose(nc0, c0) and np.allclose(nc1, c1): break
        c0, c1 = nc0, nc1

    # 更多块的 = 背景
    n0, n1 = np.sum(a==0), np.sum(a==1)
    if n0 >= n1: bg_c, sh_c = c0, c1
    else: bg_c, sh_c = c1, c0; n0, n1 = n1, n0

    bg_di, bg_dj = int(round(bg_c[0])), int(round(bg_c[1]))
    sh_di, sh_dj = int(round(sh_c[0])), int(round(sh_c[1]))
    confidence = float(max(n0, n1) / (n0 + n1))

    # 分配每个块
    block_labels = np.full((H, W), -1, dtype=np.int32)
    for i in range(H):
        for j in range(W):
            if not reliable[i, j]: continue
            v = np.array([ncc_dy[i,j], ncc_dx[i,j]], dtype=np.float32)
            db = np.sum((v - bg_c)**2)
            ds = np.sum((v - sh_c)**2)
            block_labels[i, j] = 0 if db <= ds else 1

    # 如果两个方向相同，说明只有一种运动
    if bg_di == sh_di and bg_dj == sh_dj:
        block_labels[:] = 0  # 全是背景

    print(f"  FFT+NCC 融合: BG=({bg_di},{bg_dj}), Shape=({sh_di},{sh_dj}), "
          f"置信度={confidence:.3f}, 候选块={np.sum(block_labels==1)}/{H*W}")

    return bg_di, bg_dj, confidence, block_labels


# ============================================================
# 6. 方向差异图 (用于可视化)
# ============================================================
def compute_direction_divergence_map(ds_video, bg_di, bg_dj, max_frames=10):
    """
    计算每个块的运动方向与背景方向的差异程度。

    使用结构张量对每块的帧间差分进行分析，
    得到局部主方向，然后与背景方向比较角度差。

    返回:
        divergence: (H, W) — 方向差异 (0=与背景一致, 大=不同)
    """
    F, H, W = ds_video.shape
    max_frames = min(max_frames, F - 1)

    # 累积结构张量
    acc_Jxx = np.zeros((H, W), dtype=np.float64)
    acc_Jyy = np.zeros((H, W), dtype=np.float64)
    acc_Jxy = np.zeros((H, W), dtype=np.float64)

    n = 0
    for t in range(F - 1):
        if n >= max_frames: break
        diff = (ds_video[t+1].astype(np.float32) - ds_video[t].astype(np.float32))
        Ix = cv2.Sobel(diff, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(diff, cv2.CV_32F, 0, 1, ksize=3)
        acc_Jxx += Ix * Ix
        acc_Jyy += Iy * Iy
        acc_Jxy += Ix * Iy
        n += 1

    if n == 0:
        return np.zeros((H, W))

    acc_Jxx /= n; acc_Jyy /= n; acc_Jxy /= n

    # 高斯平滑
    acc_Jxx = cv2.GaussianBlur(acc_Jxx.astype(np.float32), (0, 0), 2)
    acc_Jyy = cv2.GaussianBlur(acc_Jyy.astype(np.float32), (0, 0), 2)
    acc_Jxy = cv2.GaussianBlur(acc_Jxy.astype(np.float32), (0, 0), 2)

    # 主方向
    orientation = 0.5 * np.arctan2(2 * acc_Jxy, acc_Jxx - acc_Jyy)

    # 背景方向角
    bg_angle = np.arctan2(float(bg_dj), float(bg_di))

    # 角度差 (考虑方向对称性)
    angle_diff = np.abs(orientation - bg_angle)
    angle_diff = np.minimum(angle_diff, np.pi - angle_diff)

    return angle_diff.astype(np.float32)
