"""
文字形状马赛克视频生成器。

支持字母(A-Z)、数字(0-9)和基本图形(square/circle/triangle/star/hexagon)
作为马赛克视频中的形状，背景和形状各自可配置运动方向。

依赖 cv2 + numpy，复用 complex_shapes 中的工具函数。
"""
import cv2, numpy as np, os, math

# ---- 通用工具（从 complex_shapes 提取，避免循环依赖）----
def _make_block_texture(rows, cols):
    small = np.random.choice([0, 255], size=(rows, cols)).astype(np.uint8)
    return np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)

def _text_to_mask(text, H, W):
    """用 cv2.putText 渲染文字到二值掩码。"""
    mask = np.zeros((H, W), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    # 更大更粗的字体，确保在马赛克块级别可见
    font_scale = min(W, H) / 60
    thickness = max(8, int(font_scale * 8))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cx, cy = (W - tw) // 2, (H + th) // 2
    cv2.putText(mask, text, (cx, cy), font, font_scale, 255, thickness)
    return mask

def _get_shape_mask(shape, H, W):
    """获取任意形状的二值掩码(float64, 0/1)，坐标对齐到 4px 网格。"""
    mask = np.zeros((H//4, W//4), dtype=np.float64)
    h, w = mask.shape
    cx, cy = w/2, h/2
    s = 4  # 对齐步长

    if shape in ('square', 'rectangle'):
        side = min(w, h) // 3
        x1 = int(_snap(cx - side/2, 1)); y1 = int(_snap(cy - side/2, 1))
        x2 = int(_snap(cx + side/2, 1)); y2 = int(_snap(cy + side/2, 1))
        mask[y1:y2, x1:x2] = 1.0

    elif shape == 'circle':
        r = min(w, h) // 4
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask[dist <= r] = 1.0

    elif shape == 'triangle':
        r = min(w, h) // 3
        pts = np.array([[cx, cy - r], [cx - r, cy + r//2], [cx + r, cy + r//2]])
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1.0)

    elif shape == 'star':
        outer_r = min(w, h) // 3
        inner_r = outer_r * 0.4
        pts = []
        for k in range(10):
            r = outer_r if k % 2 == 0 else inner_r
            theta = -math.pi/2 + math.pi * k / 5
            pts.append([cx + r*math.cos(theta), cy + r*math.sin(theta)])
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1.0)

    elif shape == 'hexagon':
        r = min(w, h) // 3
        pts = []
        for k in range(6):
            theta = -math.pi/2 + math.pi * k / 3
            pts.append([cx + r*math.cos(theta), cy + r*math.sin(theta)])
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1.0)

    elif shape == 'diamond':
        r = min(w, h) // 3
        pts = np.array([[cx, cy-r], [cx+r, cy], [cx, cy+r], [cx-r, cy]])
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1.0)

    elif shape == 'cross':
        arm = min(w, h) // 8
        mask[:, int(cx-arm):int(cx+arm)] = 1.0
        mask[int(cy-arm):int(cy+arm), :] = 1.0

    elif shape == 'heart':
        r = min(w, h) // 5
        for y in range(h):
            for x in range(w):
                dx = (x - cx) / r; dy = (y - cy) / r
                # 心形公式: (x^2 + y^2 - 1)^3 - x^2*y^3 < 0
                if (dx**2 + (dy+0.3)**2 - 1)**3 - dx**2 * (dy+0.3)**3 < 0:
                    if y < cy + r*0.8:
                        mask[y, x] = 1.0

    else:  # 字母/数字/文字
        # 渲染到小块级掩码
        char_mask = _text_to_mask(shape, H, W)
        # 降采样到块级
        small_mask = char_mask[::4, ::4].astype(np.float64) / 255.0
        mask = small_mask[:h, :w]

    return np.clip(mask, 0, 1)


def _snap(v, step):
    return int(round(v / step)) * step


def generate_text_shape_video(filename, shape='A', bg_direction='right',
                               shape_direction='down', block_size=4,
                               width=640, height=640, fps=30, duration=5):
    """生成文字/图形马赛克视频。

    参数:
        filename: 输出路径
        shape: 形状 — 'A'~'Z', '0'~'9', 'square','circle','triangle','star','hexagon','diamond','cross','heart'
        bg_direction: 背景运动 — 'right','left','up','down','upright','upleft','downright','downleft','static'
        shape_direction: 形状内部运动（同上）
        block_size: 马赛克块大小(默认4)
        width, height: 视频尺寸(默认640x640)
        fps: 帧率
        duration: 时长(秒)
    """
    # 方向映射
    DIR_MAP = {
        'right': (0, 1), 'left': (0, -1), 'up': (-1, 0), 'down': (1, 0),
        'upright': (-1, 1), 'upleft': (-1, -1),
        'downright': (1, 1), 'downleft': (1, -1),
        'static': (0, 0),
    }
    bg_dy, bg_dx = DIR_MAP.get(bg_direction, (0, 1))
    sh_dy, sh_dx = DIR_MAP.get(shape_direction, (1, 0))

    # 对齐尺寸
    width = (width // block_size) * block_size
    height = (height // block_size) * block_size
    num_frames = int(fps * duration)

    # 块级尺寸
    h_blocks = height // block_size
    w_blocks = width // block_size

    # 生成独立的背景和形状纹理
    bg_small = np.random.choice([0, 255], size=(h_blocks, w_blocks)).astype(np.uint8)
    sh_small = np.random.choice([0, 255], size=(h_blocks, w_blocks)).astype(np.uint8)

    # 形状掩码(块级, 0/1)
    shape_block_mask = _get_shape_mask(shape, height, width)

    # 视频写入
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(num_frames):
        # 展开背景
        bg_frame = np.repeat(np.repeat(bg_small, block_size, axis=0), block_size, axis=1)
        # 展开形状纹理
        sh_frame = np.repeat(np.repeat(sh_small, block_size, axis=0), block_size, axis=1)

        # 像素级掩码
        pixel_mask = np.repeat(np.repeat(
            shape_block_mask.astype(np.uint8), block_size, axis=0), block_size, axis=1)
        pixel_mask = pixel_mask[:height, :width]

        # 合成：掩码区域用形状纹理，其余用背景
        frame = bg_frame.copy()
        frame[pixel_mask > 0] = sh_frame[pixel_mask > 0]

        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

        # 滚动
        if bg_dy != 0:
            bg_small = np.roll(bg_small, bg_dy, axis=0)
        if bg_dx != 0:
            bg_small = np.roll(bg_small, bg_dx, axis=1)
        if sh_dy != 0:
            sh_small = np.roll(sh_small, sh_dy, axis=0)
        if sh_dx != 0:
            sh_small = np.roll(sh_small, sh_dx, axis=1)

    out.release()
    print(f"Video generated: {filename}")


# ---- 自测 ----
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    generate_text_shape_video('data/test_A.mp4', shape='A',
                               bg_direction='right', shape_direction='down')
    print('Done!')
