"""
马赛克视频生成器 v2 —— 彩色、任意角度、可调密度和粗细

特性:
  - 彩色 RGB 马赛克块（背景/形状各自独立调色板）
  - 任意运动角度（连续 0-360°，精度 1°）
  - 可调块密度 (2/3/4 px)
  - 可调字体粗细
  - 支持字母/数字/几何图形
"""
import cv2, numpy as np, os, math, colorsys

# ---- 方向解析 ----
def _angle_to_shift(angle_deg: float) -> tuple:
    """将角度转为块级位移 (dy, dx)，支持亚块精度累积。"""
    rad = math.radians(angle_deg)
    return math.sin(rad), math.cos(rad)  # dy, dx (可能为小数)


# ---- 文字/图形掩码 ----
def _get_shape_mask(shape, H, W, block_size, thickness_scale=1.0):
    """获取形状的块级二值掩码 (h_blocks, w_blocks)，0/1 float。"""
    h, w = H // block_size, W // block_size

    # 几何图形
    cx, cy = w/2, h/2
    mask = np.zeros((h, w), dtype=np.float64)

    if shape == 'square':
        side = min(w, h) // 3
        x1, y1 = int(cx - side/2), int(cy - side/2)
        mask[y1:y1+side, x1:x1+side] = 1.0

    elif shape == 'circle':
        r = min(w, h) // 4
        Y, X = np.ogrid[:h, :w]
        mask[np.sqrt((X-cx)**2 + (Y-cy)**2) <= r] = 1.0

    elif shape == 'triangle':
        r = min(w, h) // 3
        pts = np.array([[cx, cy-r], [cx-r, cy+r//2], [cx+r, cy+r//2]], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1.0)

    elif shape == 'star':
        outer_r, inner_r = min(w,h)//3, min(w,h)//9
        pts = []
        for k in range(10):
            r = outer_r if k%2==0 else inner_r
            theta = -math.pi/2 + math.pi*k/5
            pts.append([cx+r*math.cos(theta), cy+r*math.sin(theta)])
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1.0)

    elif shape == 'hexagon':
        r = min(w, h) // 3
        pts = [[cx+r*math.cos(-math.pi/2+math.pi*k/3),
                cy+r*math.sin(-math.pi/2+math.pi*k/3)] for k in range(6)]
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1.0)

    elif shape == 'diamond':
        r = min(w,h)//3
        pts = np.array([[cx,cy-r],[cx+r,cy],[cx,cy+r],[cx-r,cy]], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1.0)

    elif shape == 'heart':
        r = min(w,h)//5
        for y_ in range(h):
            for x_ in range(w):
                dx, dy = (x_-cx)/r, (y_-cy)/r
                if (dx**2+(dy+0.3)**2-1)**3 - dx**2*(dy+0.3)**3 < 0 and y_ < cy+r*0.8:
                    mask[y_,x_] = 1.0

    elif shape == 'cross':
        arm = min(w,h)//8
        mask[:, int(cx-arm):int(cx+arm)] = 1.0
        mask[int(cy-arm):int(cy+arm), :] = 1.0

    else:
        # 字母/数字：渲染到像素级再降采样
        char_mask = np.zeros((H, W), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_DUPLEX
        # 字体大小自适应 block_size：块越小字体越大
        div = {2: 42, 3: 48, 4: 55}.get(block_size, 50)
        font_scale = min(W, H) / div * thickness_scale
        thickness = max(3, int(font_scale * 1.2))
        (tw, th), _ = cv2.getTextSize(shape, font, font_scale, thickness)
        cx_px, cy_px = (W-tw)//2, (H+th)//2
        cv2.putText(char_mask, shape, (cx_px, cy_px), font, font_scale, 255, thickness)
        # 降采样：每个 block_size×block_size 块内取均值
        for by in range(h):
            for bx in range(w):
                block = char_mask[by*block_size:(by+1)*block_size,
                                  bx*block_size:(bx+1)*block_size]
                mask[by, bx] = np.mean(block) / 255.0

    return np.clip(mask, 0, 1)


# ---- 彩色调色板 ----
def _random_palette(blocks_h, blocks_w, hue_range, sat=0.7, val=0.8):
    """生成随机 HSV 调色板，返回 RGB uint8 数组 (blocks_h, blocks_w, 3)。"""
    H = np.random.uniform(hue_range[0], hue_range[1], size=(blocks_h, blocks_w))
    S = np.full((blocks_h, blocks_w), sat)
    V = np.random.uniform(val-0.2, val+0.2, size=(blocks_h, blocks_w))
    V = np.clip(V, 0.2, 1.0)
    rgb = np.zeros((blocks_h, blocks_w, 3), dtype=np.float32)
    for y in range(blocks_h):
        for x in range(blocks_w):
            r, g, b = colorsys.hsv_to_rgb(H[y,x], S[y,x], V[y,x])
            rgb[y,x] = [r, g, b]
    return (rgb * 255).astype(np.uint8)


def _palette_to_frame(palette, block_size):
    """将 block 级调色板 (H,W,3) 展开为像素级帧 (H*bs, W*bs, 3)。"""
    h, w = palette.shape[:2]
    return np.repeat(np.repeat(palette, block_size, axis=0), block_size, axis=1)


# ---- 主函数 ----
def generate_text_shape_video(filename, shape='A',
                               bg_angle=0, shape_angle=90,
                               block_size=2, color=True,
                               thickness_scale=1.0,
                               width=640, height=640, fps=25, duration=4):
    """生成彩色马赛克形状视频。

    参数:
        filename:       输出路径 (.mp4)
        shape:          形状/字母
        bg_angle:       背景运动角度（0=右, 90=下, 180=左, 270=上）
        shape_angle:    形状内部运动角度
        block_size:     马赛克块像素大小 (2=细密, 4=标准)
        color:          是否彩色
        thickness_scale: 字体粗细缩放 (1.0=默认)
        width, height:  视频尺寸
        fps:            帧率
        duration:       时长(秒)
    """
    width = (width // block_size) * block_size
    height = (height // block_size) * block_size
    num_frames = int(fps * duration)

    h_blocks = height // block_size
    w_blocks = width // block_size

    # 形状掩码
    shape_mask = _get_shape_mask(shape, height, width, block_size, thickness_scale)

    # 背景和形状共用同一个调色板（保证伪装效果，只能靠运动差异分辨）
    if color:
        shared_pal = _random_palette(h_blocks, w_blocks, hue_range=(0.0, 1.0))
        bg_pal = shared_pal.copy()
        sh_pal = shared_pal.copy()
    else:
        rng = np.random.RandomState(42)
        shared_arr = rng.choice([0, 255], size=(h_blocks, w_blocks)).astype(np.uint8)
        bg_pal = np.stack([shared_arr]*3, axis=-1)
        sh_pal = np.stack([shared_arr]*3, axis=-1)

    # 运动累积器（亚块精度）
    bg_acc_y, bg_acc_x = 0.0, 0.0
    sh_acc_y, sh_acc_x = 0.0, 0.0
    bg_dy, bg_dx = _angle_to_shift(bg_angle)
    sh_dy, sh_dx = _angle_to_shift(shape_angle)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=True)

    for i in range(num_frames):
        # 累积位移 → 取整滚动
        bg_acc_y += bg_dy; bg_acc_x += bg_dx
        sh_acc_y += sh_dy; sh_acc_x += sh_dx
        bg_roll_y = int(round(bg_acc_y)); bg_roll_x = int(round(bg_acc_x))
        sh_roll_y = int(round(sh_acc_y)); sh_roll_x = int(round(sh_acc_x))
        bg_acc_y -= bg_roll_y; bg_acc_x -= bg_roll_x
        sh_acc_y -= sh_roll_y; sh_acc_x -= sh_roll_x

        # 滚动调色板
        bg_rolled = np.roll(np.roll(bg_pal, bg_roll_y, axis=0), bg_roll_x, axis=1)
        sh_rolled = np.roll(np.roll(sh_pal, sh_roll_y, axis=0), sh_roll_x, axis=1)

        # 展开到像素级
        bg_frame = _palette_to_frame(bg_rolled, block_size)
        sh_frame = _palette_to_frame(sh_rolled, block_size)

        # 像素级掩码
        pixel_mask = np.repeat(np.repeat(
            shape_mask.astype(np.uint8), block_size, axis=0), block_size, axis=1)
        pixel_mask = pixel_mask[:height, :width, np.newaxis]

        # 合成
        frame = (bg_frame * (1 - pixel_mask) + sh_frame * pixel_mask).astype(np.uint8)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # 更新调色板
        bg_pal = bg_rolled
        sh_pal = sh_rolled

    out.release()
    print(f"Video: {filename}  ({width}x{height}, block={block_size}px, {num_frames}frames)")


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    generate_text_shape_video('data/v2_demo.mp4', shape='A',
                               bg_angle=0, shape_angle=90,
                               block_size=2, color=True, duration=2)
    print('Done')
