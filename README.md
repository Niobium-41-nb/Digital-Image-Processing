# 马赛克视频形状识别 — Web 应用

基于**运动残差分析**的马赛克视频形状识别 Web 应用。支持在线生成马赛克形状视频、上传视频检测隐藏形状，以及基于运动差异的 CAPTCHA 验证。

---

## 快速开始

### 方式一：本地运行

```bash
pip install flask opencv-python numpy scipy pillow
python web_app.py
```

访问 `http://localhost:5000` 即可使用。

> **注意**：视频转码需要安装 [FFmpeg](https://ffmpeg.org/)，若未安装则回退使用原始 mp4v 编码（部分浏览器可能不兼容）。

### 方式二：Docker 部署（推荐）

使用 Docker 可免去本地 Python 环境配置，一键启动。

**前提条件**：安装 [Docker](https://docker.com/) 和 [Docker Compose](https://docs.docker.com/compose/)。

```bash
# 构建并启动容器
docker compose up -d

# 查看运行日志
docker compose logs -f

# 访问 http://localhost:5000
```

**单独使用 Docker（不使用 Compose）：**

```bash
# 构建镜像
docker build -t mosaic-vision .

# 运行容器
docker run -d -p 5000:5000 --name mosaic-vision mosaic-vision
```

**常用命令：**

| 命令 | 说明 |
|------|------|
| `docker compose up -d` | 后台启动服务 |
| `docker compose down` | 停止并移除容器 |
| `docker compose logs -f` | 实时查看日志 |
| `docker compose restart` | 重启服务 |
| `docker images mosaic-vision` | 查看镜像信息 |

> **Docker 镜像内置 FFmpeg**，无需额外安装，视频编码兼容性更好。

---

## 功能

### 🎬 视频生成（`/api/generate`）
生成包含隐藏形状的马赛克视频。背景和形状区域使用**完全相同的调色板**，形状在静态帧中不可见，仅能通过运动差异感知。

| 参数 | 说明 |
|------|------|
| `shape` | 字母（A–Z）、数字（2–9）或几何图形（square/circle/triangle/star/hexagon） |
| `bg_angle` / `shape_angle` | 背景与形状的运动方向（0–360°），建议差值 ≥30° |
| `block_size` | 马赛克块大小（2/3/4 px），越小纹理越细密 |
| `color` | 彩色模式 / 灰度二值 |
| `thickness_scale` | 字体粗细缩放 |
| `duration` | 视频时长（秒） |

### 🔍 形状检测（`/api/detect`）
上传马赛克视频，自动检测其中的隐藏形状。返回形状类型、位置、大小、运动方向等结构化信息，并生成可视化叠加图片。

### 🤖 CAPTCHA（`/api/captcha/*`）
基于运动感知的验证码系统。用户观看一段短视频后，输入或选择其中隐藏的形状。人类能轻松感知运动差异中的形状，而自动化程序需完成运动估计、分割和识别的复杂流水线。

---

## 核心算法

### 运动残差分析法（6 步）

```
输入视频 → ①自动检测块大小 → ②块级降采样 → ③全局运动估计 → ④运动残差分析 → ⑤精炼掩码 → ⑥后处理 → 形状掩码
```

**① 自动检测马赛克块大小**
遍历候选块大小（2–32 px），统计块内像素全部相同（标准差 < 0.5）的均匀块比例，选择均匀比例高且块尽可能小的尺寸。

**② 块级降采样**
以检测到的块大小对视频降采样，每块取第一个像素，将 `(F, H, W)` 压缩为 `(F, H/bs, W/bs)`。

**③ 全局运动估计**
尝试 9 种块级位移 `(dy, dx) ∈ {-1,0,1}²`，将前一帧平移后与后一帧逐块比较，**匹配块数最多的位移**即为背景运动方向。同时使用 FFT 相位相关进行频域验证。

**④ 运动残差分析**
将前一帧按背景方向平移，与后一帧比较——不匹配的块即为形状候选。跨帧累积不匹配率，超过阈值（默认 40%）标记为候选。结构张量分析进一步增强候选区域。

**⑤ 精炼掩码**
在候选区域内估计第二运动方向（形状内部），对每个候选块比较两种方向的匹配率，**形状方向匹配更好则确认为形状块**。

**⑥ 后处理**
块级掩码上采样 → 闭运算（填补孔洞）→ 开运算（去除噪点）→ 提取最大连通域。

### 辅助技术

| 技术 | 用途 |
|------|------|
| **FFT 相位相关** | 频域全局运动方向验证 |
| **结构张量** | 局部运动一致性分析，增强形状候选区域 |
| **归一化模板匹配** | 多尺度滑动窗口字符识别（OCR） |

---

## API 文档

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 首页 |
| `/api/generate` | POST | 生成马赛克形状视频，返回 video/mp4 |
| `/api/detect` | POST | 上传视频检测形状，返回 JSON |
| `/api/captcha/generate` | GET | 生成 CAPTCHA 视频，返回 {token, video_url} |
| `/api/captcha/video/<token>` | GET | 获取 CAPTCHA 视频文件 |
| `/api/captcha/verify` | POST | 验证 CAPTCHA 答案 |
| `/api/captcha/hint` | POST | 获取 CAPTCHA 提示 |
| `/output/<path>` | GET | 静态文件服务（检测结果图片） |

---

## 项目结构

```
├── web_app.py              # Flask 应用入口
├── templates/
│   └── index.html          # Web 前端界面
├── generators/
│   └── text_shape.py       # 马赛克形状视频生成器
├── src/
│   ├── shape_detector.py   # 形状检测核心算法
│   ├── motion_analysis.py  # 频域+空域运动分析工具
│   ├── video_converter.py  # 视频读写转换
│   └── progress_bar.py     # 终端进度条工具
├── output/                 # 输出目录（生成视频、检测结果等）
└── requirements.txt        # Python 依赖
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| **后端框架** | Flask |
| **图像/视频处理** | OpenCV, NumPy, SciPy, Pillow |
| **视频转码** | FFmpeg |
| **前端** | HTML5 + CSS3（深色主题）, 原生 JavaScript |
| **频域分析** | NumPy FFT, SciPy ndimage |

