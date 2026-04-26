"""
复杂图形视频生成模块。

与 complex_motion 关注"运动轨迹"不同，本模块关注"前景图形的几何形状"。
现有 generators 中前景一律为矩形方块，本模块新增以下复杂图形支持：

    - 圆形 / 椭圆形
    - 任意边数的正多边形（三角形、五边形、六边形 ...）
    - 星形（可配置角数与内外半径比）
    - 多形状混合场景（多个不同图形同时运动）
    - 自身旋转的多边形 / 星形

所有图形均使用与背景一致的"随机黑白方块"纹理填充，因此保持了
项目原有的"方块纹理 + 伪装"风格，但显著增加了形状层面的复杂度，
适合用于测试边缘检测、形状识别、运动分析等算法在非矩形目标下的表现。
"""

import cv2
import numpy as np
import os
import math


# ---------------------------------------------------------------------------
# 通用工具函数
# ---------------------------------------------------------------------------

def _make_block_texture(rows, cols, square_size):
    """生成放大后的随机黑白方块纹理。

    参数:
        rows: 小图像行数（以方块为单位）
        cols: 小图像列数（以方块为单位）
        square_size: 单个方块的像素边长

    返回:
        (H, W) 的 uint8 灰度图，H = rows * square_size，W = cols * square_size
    """
    small = np.random.choice([0, 255], size=(rows, cols)).astype(np.uint8)
    return np.repeat(np.repeat(small, square_size, axis=0), square_size, axis=1)


def _snap(v, step):
    """把任意数值量化（snap）到 step 的整数倍。

    保证所有运动以"方块"为最小步长，使前景方块边界永远落在背景方块网格上，
    避免亚像素抖动并保持完美伪装效果（与 perfect_camouflage.py 风格一致）。
    """
    return int(round(v / step)) * step


def _polygon_vertices(cx, cy, radius, n_sides, rotation=0.0):
    """计算正 n 边形的顶点坐标。

    参数:
        cx, cy: 多边形中心
        radius: 外接圆半径
        n_sides: 边数（>=3）
        rotation: 旋转角度（弧度），0 表示第一个顶点指向正上方

    返回:
        np.int32 数组，形状 (n_sides, 2)，可直接用于 cv2.fillPoly
    """
    pts = []
    for k in range(n_sides):
        # -pi/2 让初始顶点朝上，更符合直觉
        theta = -math.pi / 2 + rotation + 2 * math.pi * k / n_sides
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)


def _star_vertices(cx, cy, outer_r, inner_r, n_points, rotation=0.0):
    """计算星形的顶点坐标（外顶点和内顶点交替排列）。

    参数:
        cx, cy: 星形中心
        outer_r: 外顶点到中心的距离
        inner_r: 内顶点到中心的距离
        n_points: 星形角数（5 表示五角星）
        rotation: 旋转角度（弧度）

    返回:
        np.int32 数组，形状 (2 * n_points, 2)
    """
    pts = []
    total = 2 * n_points
    for k in range(total):
        r = outer_r if k % 2 == 0 else inner_r
        theta = -math.pi / 2 + rotation + math.pi * k / n_points
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)


def _composite_with_mask(bg, fg, mask):
    """按 mask（0/255）将 fg 合成到 bg 上。

    参数:
        bg: 背景灰度图，形状 (H, W)
        fg: 前景灰度图，形状 (H, W)，与 bg 同尺寸
        mask: 0/255 掩码，形状 (H, W)

    返回:
        合成后的灰度图（不修改输入）
    """
    out = bg.copy()
    out[mask > 0] = fg[mask > 0]
    return out


def _quantize_mask_to_blocks(mask, square_size):
    """把任意 mask 量化到 square_size 网格上：每个方块要么全 0、要么全 255。

    这样前景的"边缘"也严格落在方块边界上，整个前景看起来由方块拼成，
    而不是有亚像素的连续曲线边缘——产生与项目原版 perfect_camouflage
    一致的、纯方块风格的视觉效果。

    实现：把每个方块内的均值与 127 比较，超过则该方块整体为前景。

    参数:
        mask: 任意 0/255 掩码，(H, W)，要求 H 与 W 为 square_size 整数倍
        square_size: 方块边长

    返回:
        与 mask 同尺寸的方块对齐 mask
    """
    if square_size <= 1:
        return mask
    H, W = mask.shape
    h = H // square_size
    w = W // square_size
    # 把 (H, W) 重塑为 (h, square_size, w, square_size) 然后对方块内求均值
    reshaped = mask[:h * square_size, :w * square_size].reshape(
        h, square_size, w, square_size)
    block = (reshaped.mean(axis=(1, 3)) > 127).astype(np.uint8) * 255
    return np.repeat(np.repeat(block, square_size, axis=0), square_size, axis=1)


def _write_video_and_optional_frames(filename, frame_iter, width, height, fps,
                                     output_frames_folder):
    """通用的视频写入辅助器。

    将 frame_iter 产生的每一帧（灰度图）写入 mp4，并可选地把帧另存为 PNG。

    参数:
        filename: 输出 mp4 路径
        frame_iter: 可迭代对象，每次产出一帧灰度图（uint8, H x W）
        width, height: 视频尺寸
        fps: 帧率
        output_frames_folder: 是否同时把每一帧另存为 PNG
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    frames_dir = None
    if output_frames_folder:
        frames_dir = filename.rsplit('.', 1)[0] + '_frames'
        os.makedirs(frames_dir, exist_ok=True)
        print(f"同时保存帧到: {frames_dir}")

    for i, frame_gray in enumerate(frame_iter):
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

        if frames_dir is not None:
            frame_path = os.path.join(frames_dir, f'frame_{i:05d}.png')
            cv2.imwrite(frame_path, frame_gray)

        if i % 30 == 0:
            print(f"已生成帧: {i}")

    out.release()
    print(f"视频生成完成: {filename}")


# ---------------------------------------------------------------------------
# 1. 圆形前景运动
# ---------------------------------------------------------------------------

def generate_circle_shape_video(filename, width=1280, height=720, square_size=4,
                                fps=30, duration=15, radius=180,
                                motion='horizontal', output_frames_folder=False):
    """生成圆形前景的运动视频。

    前景为一个填充了随机方块纹理的圆形（而非矩形），可选择多种运动模式。
    背景纹理静止，前景圆形在画面中按指定模式移动。

    参数:
        filename: 输出视频文件路径
        width: 视频宽度（默认1280）
        height: 视频高度（默认720）
        square_size: 方块大小，影响纹理颗粒度（默认4）
        fps: 帧率（默认30）
        duration: 视频时长，单位秒（默认15）
        radius: 圆形半径，单位像素（默认180）
        motion: 运动模式，可选 'horizontal' / 'circular' / 'static'（默认'horizontal'）
        output_frames_folder: 是否同时输出每一帧到同名文件夹（默认False）
    """
    num_frames = fps * duration
    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    bg_rows = height // square_size
    bg_cols = width // square_size
    bg_img = _make_block_texture(bg_rows, bg_cols, square_size)
    fg_img = _make_block_texture(bg_rows, bg_cols, square_size)

    print(f"开始渲染圆形前景视频: {filename}")
    print(f"圆形半径={radius}px，运动模式={motion}")

    s = square_size
    snapped_radius = _snap(radius, s)

    def frames():
        for i in range(num_frames):
            t = i / max(num_frames - 1, 1)

            if motion == 'horizontal':
                progress = 0.5 - 0.5 * math.cos(2 * math.pi * t)
                cx = _snap(snapped_radius + (width - 2 * snapped_radius) * progress, s)
                cy = _snap(height // 2, s)
            elif motion == 'circular':
                angle = 2 * math.pi * 2 * t  # 2 圈
                orbit_r = min(width, height) // 4
                cx = _snap(width // 2 + orbit_r * math.cos(angle), s)
                cy = _snap(height // 2 + orbit_r * math.sin(angle), s)
            else:  # static
                cx = _snap(width // 2, s)
                cy = _snap(height // 2, s)

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), snapped_radius, 255, thickness=-1)
            mask = _quantize_mask_to_blocks(mask, s)

            yield _composite_with_mask(bg_img, fg_img, mask)

    _write_video_and_optional_frames(filename, frames(), width, height, fps,
                                     output_frames_folder)


# ---------------------------------------------------------------------------
# 2. 正多边形前景运动
# ---------------------------------------------------------------------------

def generate_polygon_shape_video(filename, width=1280, height=720, square_size=4,
                                 fps=30, duration=15, n_sides=6, radius=180,
                                 spin=True, output_frames_folder=False):
    """生成正多边形前景的运动视频。

    前景为可配置边数的正多边形（三角形、五边形、六边形 ...），
    填充随机方块纹理，可选择是否让其自身旋转。前景沿水平方向往返移动。

    参数:
        filename: 输出视频文件路径
        width: 视频宽度（默认1280）
        height: 视频高度（默认720）
        square_size: 方块大小（默认4）
        fps: 帧率（默认30）
        duration: 视频时长，单位秒（默认15）
        n_sides: 多边形边数，>=3（默认6，即正六边形）
        radius: 多边形外接圆半径，单位像素（默认180）
        spin: 是否让多边形自身旋转（默认True）
        output_frames_folder: 是否同时输出帧（默认False）
    """
    if n_sides < 3:
        raise ValueError("n_sides 必须 >= 3")

    num_frames = fps * duration
    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    bg_rows = height // square_size
    bg_cols = width // square_size
    bg_img = _make_block_texture(bg_rows, bg_cols, square_size)
    fg_img = _make_block_texture(bg_rows, bg_cols, square_size)

    print(f"开始渲染正{n_sides}边形前景视频: {filename}")
    print(f"外接圆半径={radius}px，自旋={spin}")

    s = square_size
    snapped_radius = _snap(radius, s)

    def frames():
        for i in range(num_frames):
            t = i / max(num_frames - 1, 1)

            progress = 0.5 - 0.5 * math.cos(2 * math.pi * t)
            cx = _snap(snapped_radius + (width - 2 * snapped_radius) * progress, s)
            cy = _snap(height // 2, s)

            rotation = (math.pi / 2) * (i / fps) if spin else 0.0

            verts = _polygon_vertices(cx, cy, snapped_radius, n_sides, rotation=rotation)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [verts], 255)
            mask = _quantize_mask_to_blocks(mask, s)

            yield _composite_with_mask(bg_img, fg_img, mask)

    _write_video_and_optional_frames(filename, frames(), width, height, fps,
                                     output_frames_folder)


# ---------------------------------------------------------------------------
# 3. 星形前景运动
# ---------------------------------------------------------------------------

def generate_star_shape_video(filename, width=1280, height=720, square_size=4,
                              fps=30, duration=15, n_points=5,
                              outer_radius=180, inner_ratio=0.45,
                              motion='circular', spin=True,
                              output_frames_folder=False):
    """生成星形前景的运动视频。

    前景为可配置角数的星形（默认五角星），同时支持沿轨迹运动 + 自身旋转。
    星形比正多边形具有更多内外凹凸，能产生更复杂的边缘响应。

    参数:
        filename: 输出视频文件路径
        width: 视频宽度（默认1280）
        height: 视频高度（默认720）
        square_size: 方块大小（默认4）
        fps: 帧率（默认30）
        duration: 视频时长，单位秒（默认15）
        n_points: 星形角数（默认5）
        outer_radius: 外顶点半径（默认180）
        inner_ratio: 内顶点半径占外顶点的比例，范围(0,1)（默认0.45）
        motion: 运动模式，'circular' 或 'horizontal'（默认'circular'）
        spin: 是否让星形自身旋转（默认True）
        output_frames_folder: 是否同时输出帧（默认False）
    """
    num_frames = fps * duration
    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    bg_rows = height // square_size
    bg_cols = width // square_size
    bg_img = _make_block_texture(bg_rows, bg_cols, square_size)
    fg_img = _make_block_texture(bg_rows, bg_cols, square_size)

    inner_radius = int(outer_radius * inner_ratio)

    print(f"开始渲染{n_points}角星前景视频: {filename}")
    print(f"外径={outer_radius}px, 内径={inner_radius}px, 运动={motion}, 自旋={spin}")

    s = square_size
    snapped_outer = _snap(outer_radius, s)
    snapped_inner = _snap(inner_radius, s)

    def frames():
        for i in range(num_frames):
            t = i / max(num_frames - 1, 1)

            if motion == 'circular':
                angle = 2 * math.pi * 2 * t  # 2 圈
                orbit_r = min(width, height) // 4
                cx = _snap(width // 2 + orbit_r * math.cos(angle), s)
                cy = _snap(height // 2 + orbit_r * math.sin(angle), s)
            else:  # horizontal
                progress = 0.5 - 0.5 * math.cos(2 * math.pi * t)
                cx = _snap(snapped_outer + (width - 2 * snapped_outer) * progress, s)
                cy = _snap(height // 2, s)

            rotation = (math.pi / 2) * (i / fps) if spin else 0.0

            verts = _star_vertices(cx, cy, snapped_outer, snapped_inner,
                                   n_points, rotation=rotation)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [verts], 255)
            mask = _quantize_mask_to_blocks(mask, s)

            yield _composite_with_mask(bg_img, fg_img, mask)

    _write_video_and_optional_frames(filename, frames(), width, height, fps,
                                     output_frames_folder)


# ---------------------------------------------------------------------------
# 4. 多形状混合场景
# ---------------------------------------------------------------------------

def generate_multi_shapes_video(filename, width=1280, height=720, square_size=4,
                                fps=30, duration=15, output_frames_folder=False):
    """生成多形状混合运动视频。

    画面中同时存在 4 个不同形状的前景物体：
        - 圆形（左上）：水平往返
        - 三角形（右上）：垂直往返 + 自旋
        - 正方形（左下）：圆周运动
        - 五角星（右下）：8 字形运动 + 自旋

    所有前景共享同一份"随机方块纹理"，与背景纹理类似但内容独立。
    用于测试同一帧内多个不同形状目标的检测/分割表现。

    参数:
        filename: 输出视频文件路径
        width: 视频宽度（默认1280）
        height: 视频高度（默认720）
        square_size: 方块大小（默认4）
        fps: 帧率（默认30）
        duration: 视频时长，单位秒（默认15）
        output_frames_folder: 是否同时输出帧（默认False）
    """
    num_frames = fps * duration
    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    bg_rows = height // square_size
    bg_cols = width // square_size
    bg_img = _make_block_texture(bg_rows, bg_cols, square_size)
    fg_img = _make_block_texture(bg_rows, bg_cols, square_size)

    s = square_size
    qx_left = _snap(width // 4, s)
    qx_right = _snap(3 * width // 4, s)
    qy_top = _snap(height // 4, s)
    qy_bot = _snap(3 * height // 4, s)

    circle_r = _snap(90, s)
    tri_r = _snap(100, s)
    sq_half = _snap(80, s)
    star_outer = _snap(110, s)
    star_inner = _snap(star_outer * 0.45, s)

    print(f"开始渲染多形状混合视频: {filename}")
    print(f"包含: 圆形/三角形/正方形/五角星，共 4 个前景目标")

    def frames():
        for i in range(num_frames):
            t = i / max(num_frames - 1, 1)
            mask = np.zeros((height, width), dtype=np.uint8)

            # ---- 圆形（左上象限，水平往返）----
            cx1 = _snap(qx_left + 100 * math.sin(2 * math.pi * t), s)
            cy1 = qy_top
            cv2.circle(mask, (cx1, cy1), circle_r, 255, thickness=-1)

            # ---- 三角形（右上象限，垂直往返 + 自旋）----
            cx2 = qx_right
            cy2 = _snap(qy_top + 80 * math.sin(2 * math.pi * t * 1.5), s)
            rot2 = 2 * math.pi * t
            tri_pts = _polygon_vertices(cx2, cy2, tri_r, n_sides=3, rotation=rot2)
            cv2.fillPoly(mask, [tri_pts], 255)

            # ---- 正方形（左下象限，圆周运动）----
            angle3 = 2 * math.pi * 2 * t
            orbit3 = 80
            cx3 = _snap(qx_left + orbit3 * math.cos(angle3), s)
            cy3 = _snap(qy_bot + orbit3 * math.sin(angle3), s)
            cv2.rectangle(mask,
                          (cx3 - sq_half, cy3 - sq_half),
                          (cx3 + sq_half, cy3 + sq_half),
                          255, thickness=-1)

            # ---- 五角星（右下象限，8 字形运动 + 自旋）----
            tau = 2 * math.pi * 2 * t
            cx4 = _snap(qx_right + 100 * math.sin(tau), s)
            cy4 = _snap(qy_bot + 70 * math.sin(2 * tau), s)
            rot4 = 2 * math.pi * t * 2
            star_pts = _star_vertices(cx4, cy4, star_outer, star_inner,
                                      n_points=5, rotation=rot4)
            cv2.fillPoly(mask, [star_pts], 255)

            mask = _quantize_mask_to_blocks(mask, s)
            yield _composite_with_mask(bg_img, fg_img, mask)

    _write_video_and_optional_frames(filename, frames(), width, height, fps,
                                     output_frames_folder)


# ---------------------------------------------------------------------------
# 5. 自身旋转的图形（运动方向 = 纯旋转，无平移）
# ---------------------------------------------------------------------------

def generate_rotating_shape_video(filename, width=1280, height=720, square_size=4,
                                  fps=30, duration=15, shape='star',
                                  radius=220, n=5, total_turns=3,
                                  output_frames_folder=False):
    """生成纯自旋图形视频（图形位置固定，仅自身旋转）。

    用于隔离"旋转"这一单一运动分量，相比其他模块的"平移+旋转"复合运动，
    更便于研究旋转对边缘检测、纹理流估计等任务的单独影响。

    参数:
        filename: 输出视频文件路径
        width: 视频宽度（默认1280）
        height: 视频高度（默认720）
        square_size: 方块大小（默认4）
        fps: 帧率（默认30）
        duration: 视频时长，单位秒（默认15）
        shape: 图形类型，'polygon' 或 'star'（默认'star'）
        radius: 图形外接圆半径（默认220）
        n: 当 shape='polygon' 时表示边数；'star' 时表示角数（默认5）
        total_turns: 整个视频总共旋转的圈数（默认3）
        output_frames_folder: 是否同时输出帧（默认False）
    """
    num_frames = fps * duration
    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    bg_rows = height // square_size
    bg_cols = width // square_size
    bg_img = _make_block_texture(bg_rows, bg_cols, square_size)
    fg_img = _make_block_texture(bg_rows, bg_cols, square_size)

    s = square_size
    cx = _snap(width // 2, s)
    cy = _snap(height // 2, s)
    snapped_radius = _snap(radius, s)
    snapped_inner = _snap(radius * 0.45, s)
    angular_speed = (2 * math.pi * total_turns) / num_frames

    print(f"开始渲染自旋图形视频: {filename}")
    print(f"形状={shape}, n={n}, 半径={radius}px, 总圈数={total_turns}")

    def frames():
        for i in range(num_frames):
            rotation = angular_speed * i
            mask = np.zeros((height, width), dtype=np.uint8)

            if shape == 'polygon':
                verts = _polygon_vertices(cx, cy, snapped_radius, n, rotation=rotation)
            elif shape == 'star':
                verts = _star_vertices(cx, cy, snapped_radius, snapped_inner,
                                       n, rotation=rotation)
            else:
                raise ValueError(f"未知 shape: {shape}，应为 'polygon' 或 'star'")

            cv2.fillPoly(mask, [verts], 255)
            mask = _quantize_mask_to_blocks(mask, s)
            yield _composite_with_mask(bg_img, fg_img, mask)

    _write_video_and_optional_frames(filename, frames(), width, height, fps,
                                     output_frames_folder)


# ---------------------------------------------------------------------------
# 6. 统一接口：任意形状 × 任意运动 × 自旋 × 脉冲缩放
# ---------------------------------------------------------------------------

def _build_shape_mask(shape, n, cx, cy, radius, rotation, height, width,
                      square_size=1):
    """根据形状类型构建二值掩码。

    若 square_size > 1，最后会把 mask 量化到 square_size 网格上，
    让前景边缘严格由方块拼成（消除连续曲线边缘）。

    参数:
        shape: 'circle' / 'polygon' / 'star' / 'square'
        n: polygon 时为边数，star 时为角数（其他形状忽略）
        cx, cy: 形状中心
        radius: 形状的"等效半径"（外接圆半径）
        rotation: 当前自旋角度（弧度）
        height, width: 输出掩码尺寸
        square_size: 方块边长（用于把 mask 量化到方块网格）
    返回:
        (H, W) uint8 掩码，前景=255，背景=0
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if shape == 'circle':
        cv2.circle(mask, (cx, cy), int(radius), 255, thickness=-1)
    elif shape == 'square':
        # 把 radius 解释为半边长，正方形边长 = 2 * radius
        half = int(radius)
        x1, y1 = cx - half, cy - half
        x2, y2 = cx + half, cy + half
        # 自旋的正方形 = 旋转的 4 边形
        if abs(rotation) > 1e-6:
            verts = _polygon_vertices(cx, cy, int(radius * math.sqrt(2)),
                                      n_sides=4, rotation=rotation + math.pi / 4)
            cv2.fillPoly(mask, [verts], 255)
        else:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    elif shape == 'polygon':
        verts = _polygon_vertices(cx, cy, int(radius), n, rotation=rotation)
        cv2.fillPoly(mask, [verts], 255)
    elif shape == 'star':
        verts = _star_vertices(cx, cy, int(radius), int(radius * 0.45),
                               n, rotation=rotation)
        cv2.fillPoly(mask, [verts], 255)
    else:
        raise ValueError(f"未知 shape={shape}，应为 circle/polygon/star/square")

    return _quantize_mask_to_blocks(mask, square_size)


def _trajectory(motion, frame_idx, num_frames, width, height, radius,
                square_size, cycles=2):
    """根据运动模式返回当前帧的中心位置 (cx, cy)。

    所有返回坐标已对齐到 square_size 整数倍，保证前景方块与背景方块
    始终对齐到同一网格上（与 perfect_camouflage 风格一致）。

    参数:
        motion: 运动模式名（见下表）
        frame_idx: 当前帧索引
        num_frames: 总帧数
        width, height: 画面尺寸
        radius: 形状半径，用于边距保护
        square_size: 方块尺寸（运动最小步长）
        cycles: 周期性运动重复次数（圆/8字/对角/螺旋通用）

    支持的运动模式:
        'static'        - 静止居中
        'horizontal'    - 水平往返
        'vertical'      - 垂直往返
        'diagonal'      - 沿对角线往返
        'circular'      - 圆周运动
        'spiral'        - 螺旋（半径递增）
        'figure_eight'  - 8 字形 (Lissajous)
    """
    t = frame_idx / max(num_frames - 1, 1)
    cx0 = _snap(width // 2, square_size)
    cy0 = _snap(height // 2, square_size)
    s = square_size

    if motion == 'static':
        return cx0, cy0

    if motion == 'horizontal':
        progress = 0.5 - 0.5 * math.cos(2 * math.pi * t * cycles)
        cx = _snap(radius + (width - 2 * radius) * progress, s)
        return cx, cy0

    if motion == 'vertical':
        progress = 0.5 - 0.5 * math.cos(2 * math.pi * t * cycles)
        cy = _snap(radius + (height - 2 * radius) * progress, s)
        return cx0, cy

    if motion == 'diagonal':
        progress = 0.5 - 0.5 * math.cos(2 * math.pi * t * cycles)
        cx = _snap(radius + (width - 2 * radius) * progress, s)
        cy = _snap(radius + (height - 2 * radius) * progress, s)
        return cx, cy

    if motion == 'circular':
        angle = 2 * math.pi * cycles * t
        orbit = min(width, height) // 2 - radius - 20
        return (_snap(cx0 + orbit * math.cos(angle), s),
                _snap(cy0 + orbit * math.sin(angle), s))

    if motion == 'spiral':
        angle = 2 * math.pi * cycles * 2 * t
        max_orbit = min(width, height) // 2 - radius - 20
        orbit = max_orbit * t
        return (_snap(cx0 + orbit * math.cos(angle), s),
                _snap(cy0 + orbit * math.sin(angle), s))

    if motion == 'figure_eight':
        tau = 2 * math.pi * cycles * t
        amp_x = width // 2 - radius - 20
        amp_y = height // 2 - radius - 20
        return (_snap(cx0 + amp_x * math.sin(tau), s),
                _snap(cy0 + amp_y * math.sin(2 * tau), s))

    raise ValueError(f"未知 motion={motion}")


def generate_shape_motion_video(
    filename,
    shape='star',
    motion='figure_eight',
    width=1280, height=720, square_size=4,
    fps=30, duration=15,
    radius=160,
    n=5,
    spin=False,
    spin_turns=2,
    pulse=False,
    pulse_cycles=4,
    pulse_range=(0.6, 1.3),
    cycles=2,
    step_per_frame=None,
    output_frames_folder=False,
):
    """统一的"形状 × 运动"组合生成器（最强大的接口）。

    可以把任意形状和任意运动模式自由叠加，并且可同时启用：
        - 自旋（图形绕自身中心转）
        - 脉冲（图形周期性放大缩小）
        - 复杂轨迹（圆周/螺旋/8字/对角线...）
        - 方块步长锁定（step_per_frame）

    示例：旋转的五角星沿 8 字形轨迹运动，同时尺寸做正弦脉冲。

    参数:
        filename: 输出视频文件路径
        shape: 形状 - 'circle' / 'polygon' / 'star' / 'square'（默认'star'）
        motion: 运动 - 'static' / 'horizontal' / 'vertical' / 'diagonal'
                       / 'circular' / 'spiral' / 'figure_eight'（默认'figure_eight'）
        width, height: 画面尺寸
        square_size: 方块纹理颗粒度（默认4）
        fps: 帧率（默认30）
        duration: 时长（秒，默认15）
        radius: 形状基准半径（默认160）
        n: polygon 边数 / star 角数（默认5）
        spin: 是否自旋（默认False）
        spin_turns: 自旋总圈数（默认2）
        pulse: 是否启用脉冲缩放（默认False）
        pulse_cycles: 脉冲周期数（默认4）
        pulse_range: (最小缩放, 最大缩放) 二元组（默认 (0.6, 1.3)）
        cycles: 平移轨迹的重复周期数（默认2）
        step_per_frame: 每帧最多移动多少个方块（即 step_per_frame * square_size 像素）。
                        None 表示不限制（按原始轨迹速度）；
                        1 表示每帧最多移动 1 个方块（最像 perfect_camouflage 风格）；
                        2~3 也可以做出明显的"逐方块跳跃"视觉效果。
                        启用后前景会从上一帧位置出发、朝向理想轨迹位置最多走该步长。
                        注意：限速后前景可能跟不上原 cycles 设定的圈数，
                        如需走完更多圈请相应增大 duration。
        output_frames_folder: 是否同时输出帧（默认False）
    """
    num_frames = fps * duration
    width = (width // square_size) * square_size
    height = (height // square_size) * square_size

    bg_rows = height // square_size
    bg_cols = width // square_size
    bg_img = _make_block_texture(bg_rows, bg_cols, square_size)
    fg_img = _make_block_texture(bg_rows, bg_cols, square_size)

    print(f"开始渲染组合视频: {filename}")
    print(f"形状={shape}(n={n}), 运动={motion}, 自旋={spin}, 脉冲={pulse}, "
          f"步长锁定={step_per_frame}")

    min_s, max_s = pulse_range
    max_step_px = (step_per_frame * square_size) if step_per_frame else None

    def frames():
        prev_cx, prev_cy = None, None  # 追逐模式下记录上一帧实际位置

        for i in range(num_frames):
            # 自旋角度
            rotation = (2 * math.pi * spin_turns * i / num_frames) if spin else 0.0

            # 脉冲缩放（半径也量化到 square_size 的整数倍）
            if pulse:
                phase = 2 * math.pi * pulse_cycles * i / num_frames
                scale = min_s + (max_s - min_s) * (0.5 + 0.5 * math.sin(phase))
            else:
                scale = 1.0
            cur_radius = max(square_size, _snap(radius * scale, square_size))

            # 理想轨迹位置
            raw_cx, raw_cy = _trajectory(
                motion, i, num_frames, width, height,
                _snap(radius * max_s, square_size),
                square_size, cycles=cycles)

            # 步长锁定（追逐模式）：限制每帧最大位移
            if max_step_px is not None and prev_cx is not None:
                dx = raw_cx - prev_cx
                dy = raw_cy - prev_cy
                dist = math.hypot(dx, dy)
                if dist > max_step_px:
                    dx = dx * max_step_px / dist
                    dy = dy * max_step_px / dist
                cx = _snap(prev_cx + dx, square_size)
                cy = _snap(prev_cy + dy, square_size)
            else:
                cx, cy = raw_cx, raw_cy

            prev_cx, prev_cy = cx, cy

            mask = _build_shape_mask(shape, n, cx, cy, cur_radius,
                                     rotation, height, width,
                                     square_size=square_size)
            yield _composite_with_mask(bg_img, fg_img, mask)

    _write_video_and_optional_frames(filename, frames(), width, height, fps,
                                     output_frames_folder)


if __name__ == "__main__":
    # 始终把视频输出到"项目根目录/data/"，与 generators/ 同级，
    # 不受当前工作目录（在 generators/ 下还是项目根下运行）影响。
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"输出目录: {DATA_DIR}")

    def out(name):
        """把文件名拼到统一的输出目录上。"""
        return os.path.join(DATA_DIR, name)

    print("=" * 60)
    print("生成圆形前景视频...")
    generate_circle_shape_video(out('shape_circle.mp4'), motion='circular')

    print("=" * 60)
    print("生成正六边形前景视频...")
    generate_polygon_shape_video(out('shape_hexagon.mp4'), n_sides=6, spin=True)

    print("=" * 60)
    print("生成五角星前景视频...")
    generate_star_shape_video(out('shape_star.mp4'), n_points=5, motion='circular')

    print("=" * 60)
    print("生成多形状混合视频...")
    generate_multi_shapes_video(out('shape_multi.mp4'))

    print("=" * 60)
    print("生成自旋星形视频...")
    generate_rotating_shape_video(out('shape_rotating_star.mp4'),
                                  shape='star', n=6, total_turns=4)

    print("=" * 60)
    print("【组合 demo 1】自旋五角星 + 8 字轨迹 + 脉冲缩放（每帧 1 方块步长）")
    generate_shape_motion_video(
        out('combo_star_fig8_spin_pulse.mp4'),
        shape='star', n=5, motion='figure_eight',
        spin=True, pulse=True,
        step_per_frame=1,
    )

    print("=" * 60)
    print("【组合 demo 2】六边形 + 螺旋轨迹 + 自旋（每帧 1 方块步长）")
    generate_shape_motion_video(
        out('combo_hex_spiral_spin.mp4'),
        shape='polygon', n=6, motion='spiral',
        spin=True, spin_turns=4,
        step_per_frame=1,
    )

    print("=" * 60)
    print("【组合 demo 3】圆形 + 对角线运动 + 脉冲（每帧 2 方块步长）")
    generate_shape_motion_video(
        out('combo_circle_diagonal_pulse.mp4'),
        shape='circle', motion='diagonal',
        pulse=True, pulse_range=(0.4, 1.4),
        step_per_frame=2,
    )

    print("=" * 60)
    print("【组合 demo 4】对比演示：相同参数下 step_per_frame=None vs 1")
    print("  → 不限速版本（看起来连续滑动）...")
    generate_shape_motion_video(
        out('combo_compare_unlocked.mp4'),
        shape='star', n=5, motion='figure_eight',
        step_per_frame=None,
        duration=8,
    )
    print("  → 锁定 1 方块/帧版本（明显的方块跳跃）...")
    generate_shape_motion_video(
        out('combo_compare_locked.mp4'),
        shape='star', n=5, motion='figure_eight',
        step_per_frame=1,
        duration=8,
    )

    print("=" * 60)
    print("所有复杂图形视频生成完成！")
