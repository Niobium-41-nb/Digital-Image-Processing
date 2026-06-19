"""
Flask 后端 — 马赛克视频形状识别 + 生成 + CAPTCHA

启动:  python web_app.py
访问:  http://localhost:5000
"""
import os, sys, uuid, io, json, time, glob

# 确保项目根在 sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, request, jsonify, send_file, session
import cv2, numpy as np

app = Flask(__name__)
app.secret_key = 'mosaic_shape_captcha_2024_secret'
app.config['JSON_AS_ASCII'] = False  # 允许 JSON 返回中文

# ---- 工具: 进度条静默 ----
import src.progress_bar as pb
_pb_update = pb.update_progress
_pb_finish = pb.finish_progress
def _quiet():
    pb.update_progress = lambda *a, **kw: None
    pb.finish_progress = lambda *a, **kw: None
def _loud():
    pb.update_progress = _pb_update
    pb.finish_progress = _pb_finish

_quiet()


# ============================================
# 视频转换 (mp4v → 浏览器兼容 H.264)
# ============================================
def _to_browser_video(src_path, dst_path=None):
    """用 ffmpeg 将 mp4v 视频转为浏览器兼容的 H.264。"""
    if dst_path is None:
        dst_path = src_path.replace('.mp4', '_h264.mp4')
    import subprocess
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', src_path,
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            '-an',  # 无音频
            dst_path
        ], check=True, capture_output=True, timeout=60)
        return dst_path
    except Exception as e:
        print(f"  ffmpeg conversion failed: {e}, falling back to original")
        return src_path


# ============================================
# 清理旧文件
# ============================================
def _cleanup_old(dir_path, max_age_s=1800):
    """删除超过 max_age_s 秒的旧文件。"""
    now = time.time()
    if os.path.isdir(dir_path):
        for f in os.listdir(dir_path):
            fp = os.path.join(dir_path, f)
            try:
                if os.path.isfile(fp) and now - os.path.getmtime(fp) > max_age_s:
                    os.remove(fp)
            except:
                pass


# ============================================
# 功能 1: 视频生成 API
# ============================================
@app.route('/api/generate', methods=['POST'])
def api_generate():
    """生成马赛克形状视频。

    POST JSON: {shape, bg_angle, shape_angle, block_size, color, thickness_scale, duration}
    返回: video/mp4
    """
    try:
        data = request.get_json()
        shape = data.get('shape', 'A')
        bg_angle = float(data.get('bg_angle', 0))
        sh_angle = float(data.get('shape_angle', 90))
        block_size = int(data.get('block_size', 2))
        color = data.get('color', True)
        thickness_scale = float(data.get('thickness_scale', 1.0))
        duration = int(data.get('duration', 4))

        # 生成到临时文件
        os.makedirs('output/web_generated', exist_ok=True)
        _cleanup_old('output/web_generated')

        vid_id = uuid.uuid4().hex[:8]
        fname = f'output/web_generated/{vid_id}.mp4'

        from generators.text_shape import generate_text_shape_video
        _loud()  # 控制台看到进度
        generate_text_shape_video(fname, shape=shape,
                                   bg_angle=bg_angle,
                                   shape_angle=sh_angle,
                                   block_size=block_size,
                                   color=color,
                                   thickness_scale=thickness_scale,
                                   width=640, height=640,
                                   fps=25, duration=duration)
        _quiet()

        # 转为浏览器兼容格式
        fname_h264 = _to_browser_video(fname)
        return send_file(os.path.abspath(fname_h264), mimetype='video/mp4')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================
# 功能 2: 形状检测 API
# ============================================
@app.route('/api/detect', methods=['POST'])
def api_detect():
    """上传视频，检测其中的形状。

    POST: multipart/form-data, field 'video'
    返回 JSON: {shape_type, position, center, area_pct, width, height,
                vertices, bg_direction, shape_direction, description}
    """
    try:
        f = request.files.get('video')
        if not f:
            return jsonify({'error': 'No video file'}), 400

        # 保存上传视频
        os.makedirs('output/web_uploads', exist_ok=True)
        _cleanup_old('output/web_uploads')
        vid_id = uuid.uuid4().hex[:8]
        up_path = f'output/web_uploads/{vid_id}.mp4'
        f.save(up_path)

        # 检测
        from src.shape_detector import detect_shape, describe_shape
        _loud()
        result = detect_shape(up_path, block_size=None,  # 自动检测块大小
                              output_dir=f'output/web_uploads/{vid_id}_result',
                              visualize=True)
        _quiet()

        mask = result['mask']
        shape_px = int(np.sum(mask > 0))
        if shape_px == 0:
            return jsonify({
                'found': False,
                'message': '视频中所有区域的运动方向一致，未发现与背景运动不同的形状区域。',
                'bg_direction': list(result['bg_direction']),
                'shape_direction': list(result['shape_direction']),
                'block_size': result['block_size'],
                'area_pct': 0,
            })

        # 文字描述 + 结构化数据（直接从 describe_shape 返回）
        desc, stats = describe_shape(mask, block_size=result['block_size'])

        return jsonify({
            'found': True,
            'description': desc,
            'stats': stats,
            'block_size': result['block_size'],
            'bg_direction': list(result['bg_direction']),
            'shape_direction': list(result['shape_direction']),
            'area_pct': round(100 * np.sum(mask > 0) / mask.size, 1),
            'result_dir': f'output/web_uploads/{vid_id}_result',
            'contours': len(result['contours']),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================
# 功能 3: CAPTCHA API
# ============================================
_captcha_store = {}  # {token: {'shape': 'A', 'created': timestamp}}

@app.route('/api/captcha/generate', methods=['GET'])
def api_captcha_generate():
    """生成 CAPTCHA 视频。

    随机选一个形状，生成短视频，返回 {token, video_url, options?}
    """
    _cleanup_old('output/web_captcha')
    os.makedirs('output/web_captcha', exist_ok=True)

    # 随机形状池（已去 I/O/0/1，并按易混淆对分组确保不会同时出现）
    _CONFUSING_GROUPS = [
        ['8', 'B'],       # 双圈结构
        ['5', 'S'],       # 曲线形
        ['2', 'Z'],       # 折线形
        ['6', 'G'],       # 圈+尾
        ['C', 'G'],       # 开口圈
        ['D', 'O', '0'],  # 圆形（O/0 已排除，仅 D）
        ['E', 'F'],       # 三横
        ['P', 'R'],       # 圈+腿
        ['U', 'V'],       # 开口向上 vs 尖底
        ['M', 'W'],       # 对称峰谷
        ['H', 'N'],       # 横杠数不同但在马赛克中易混
    ]
    _all_chars = list('ABCDEFGHJKLMNPQRSTUVWXYZ23456789')
    # 构建一个安全的候选池：从每组最多选一个
    _used_in_groups = set()
    for grp in _CONFUSING_GROUPS:
        for c in grp:
            _used_in_groups.add(c)
    pool = [c for c in _all_chars if c not in _used_in_groups]  # 非混淆字符
    # 从每个混淆组随机选一个加入
    import random as _random
    for grp in _CONFUSING_GROUPS:
        available = [c for c in grp if c in _all_chars]
        if available:
            pool.append(_random.choice(available))
    shape_pool = ['square', 'circle', 'triangle', 'star', 'hexagon']
    # 让 shapes 占 ~40%: 把 shape_pool 重复加入
    pool = pool + shape_pool * max(1, len(pool) // len(shape_pool))
    shape = np.random.choice(pool)

    # 清晰可辨的角度组合，两方向差距 >= 30°
    ANGLE_PAIRS = [
        (0, 90), (0, 135), (0, 180), (0, 225), (0, 270), (0, 315),
        (45, 135), (45, 180), (45, 225), (45, 270), (45, 315),
        (90, 180), (90, 225), (90, 270), (90, 315), (90, 0),
        (135, 225), (135, 270), (135, 315), (135, 0), (135, 45),
        (180, 270), (180, 315), (180, 0), (180, 45), (180, 90),
        (225, 315), (225, 0), (225, 45), (225, 90), (225, 135),
        (270, 0), (270, 45), (270, 90), (270, 135), (270, 180),
        (315, 45), (315, 90), (315, 135), (315, 180), (315, 225),
    ]
    bg_angle, sh_angle = ANGLE_PAIRS[np.random.randint(0, len(ANGLE_PAIRS))]

    token = uuid.uuid4().hex[:12]
    vid_path = f'output/web_captcha/{token}.mp4'

    from generators.text_shape import generate_text_shape_video
    _loud()
    generate_text_shape_video(vid_path, shape=shape,
                               bg_angle=bg_angle,
                               shape_angle=sh_angle,
                               block_size=2, color=True,
                               thickness_scale=1.0,
                               width=400, height=400,
                               fps=20, duration=2.5)
    _quiet()

    # 转为浏览器兼容格式
    vid_path = _to_browser_video(vid_path)

    _captcha_store[token] = {
        'shape': shape,
        'created': time.time(),
    }

    # 清理过期(超过5分钟)
    now = time.time()
    for k in list(_captcha_store.keys()):
        if now - _captcha_store[k]['created'] > 300:
            del _captcha_store[k]

    return jsonify({
        'token': token,
        'video_url': f'/api/captcha/video/{token}',
        'shape': shape,  # 前端据此决定显示按钮还是输入框
    })


@app.route('/api/captcha/video/<token>')
def api_captcha_video(token):
    """返回 CAPTCHA 视频文件。"""
    # 优先用 H.264 版本
    for path in [f'output/web_captcha/{token}_h264.mp4',
                 f'output/web_captcha/{token}.mp4']:
        if os.path.exists(path):
            return send_file(os.path.abspath(path), mimetype='video/mp4')
    return 'Video not found or expired', 404


@app.route('/api/captcha/verify', methods=['POST'])
def api_captcha_verify():
    """验证 CAPTCHA 答案。

    POST JSON: {token, answer}
    返回: {success, message}
    """
    data = request.get_json()
    token = data.get('token', '')
    answer = data.get('answer', '').strip().upper()

    entry = _captcha_store.get(token)
    if not entry:
        return jsonify({'success': False, 'message': 'CAPTCHA expired, please refresh'})

    correct = entry['shape'].upper()
    if answer == correct:
        del _captcha_store[token]
        return jsonify({'success': True, 'message': f'Correct! The shape is {correct}'})
    else:
        return jsonify({'success': False, 'message': f'Incorrect. Expected {correct}, got {answer}'})


@app.route('/api/captcha/hint', methods=['POST'])
def api_captcha_hint():
    """获取 CAPTCHA 提示（显示方向信息）。"""
    data = request.get_json()
    token = data.get('token', '')
    entry = _captcha_store.get(token)
    if not entry:
        return jsonify({'hint': 'Expired'})
    # 不做任何提示，这会让 CAPTCHA 失去意义
    return jsonify({'hint': 'Watch the video carefully — the shape moves differently from the background'})


# ============================================
# 静态文件服务（检测结果图片）
# ============================================
@app.route('/output/<path:filepath>')
def serve_output(filepath):
    """服务 output 目录下的静态文件。"""
    return send_file(os.path.abspath(f'output/{filepath}'))


# ============================================
# 首页
# ============================================
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    # 检查模板是否存在
    if not os.path.exists('templates/index.html'):
        print("Warning: templates/index.html not found!")
    app.run(host='0.0.0.0', port=5000, debug=True)
