# 数字图像处理项目

这是一个用于生成和处理纹理视频的数字图像处理项目，提供了多种视频生成和处理功能，包括双滚动纹理、完美伪装效果、滚动纹理等。

## 项目结构

```
数字图像处理/
├── data/                # 输入视频数据目录
├── generators/          # 视频生成模块
│   ├── dual_scrolling.py        # 双滚动纹理视频生成
│   ├── perfect_camouflage.py    # 完美伪装效果视频生成
│   └── scrolling_texture.py     # 滚动纹理视频生成
├── output/              # 输出视频目录
├── src/                 # 核心处理模块
│   ├── video_converter.py       # 视频与数组转换功能
│   └── temporal_convolver.py    # 时间卷积处理功能
├── run.py               # 主程序文件
└── README.md            # 项目说明文档
```

## 功能特性

- ✅ 纹理视频生成
  - 双滚动纹理视频（背景和前景沿不同方向滚动）
  - 完美伪装效果视频（前景块与背景融合）
  - 滚动纹理视频（固定区域内纹理滚动）
- ✅ 视频与 NumPy 数组之间的相互转换
- ✅ 时间维度卷积处理
- ✅ 逐帧卷积处理
- ✅ 视频差分处理
- ✅ 最大池化处理
- ✅ 多通道视频合成（RGB、GB通道）

## 安装依赖

在使用本项目之前，需要安装以下依赖：

```bash
pip install opencv-python numpy scipy
```

## 核心模块说明

### 1. 视频生成模块 (generators/)

#### dual_scrolling.py
- `generate_dual_scrolling_texture_video()`: 生成双滚动纹理视频，背景向右滚动，前景方块内部向下滚动

#### perfect_camouflage.py
- `generate_perfect_camouflage_video()`: 生成完美伪装效果视频，前景块在背景中水平移动，利用随机纹理实现伪装效果

#### scrolling_texture.py
- `generate_scrolling_texture_video()`: 生成滚动纹理视频，固定区域内纹理向下循环滚动

### 2. 核心处理模块 (src/)

#### video_converter.py
- `mp4_to_grayscale_array()`: 读取MP4文件并返回灰度三维NumPy数组
- `gray_array_to_mp4()`: 将灰度三维NumPy数组写入MP4文件
- `two_gray_array_to_GB_mp4()`: 将两个灰度数组作为G和B通道写入彩色视频
- `three_gray_array_to_RGB_mp4()`: 将三个灰度数组作为RGB通道写入彩色视频

#### temporal_convolver.py
- `create_temporal_motion_blur()`: 创建时间维度的运动模糊卷积核
- `apply_temporal_convolution()`: 对三维数组执行时间卷积
- `apply_frame_convolution()`: 对每一帧图像执行二维卷积
- `apply_diff()`: 对视频做差分处理，计算相邻帧的差异
- `apply_peel_max()`: 对每一帧图像执行最大池化

## 使用示例

### 1. 生成纹理视频

```python
# 生成双滚动纹理视频
from generators.dual_scrolling import generate_dual_scrolling_texture_video
generate_dual_scrolling_texture_video('data/dual_scroll.mp4', square_size=4)

# 生成完美伪装效果视频
from generators.perfect_camouflage import generate_perfect_camouflage_video
generate_perfect_camouflage_video('data/camouflage.mp4', square_size=4)

# 生成滚动纹理视频
from generators.scrolling_texture import generate_scrolling_texture_video
generate_scrolling_texture_video('data/scrolling.mp4', square_size=4)
```

### 2. 处理视频

```python
from src.video_converter import mp4_to_grayscale_array, gray_array_to_mp4
from src.temporal_convolver import create_temporal_motion_blur, apply_temporal_convolution

# 读取视频为数组
arr = mp4_to_grayscale_array('data/input.mp4')

# 创建时间卷积核
convolution = create_temporal_motion_blur(5, 5, 5)

# 应用时间卷积
processed_arr = apply_temporal_convolution(arr, convolution)

# 保存处理后的视频
gray_array_to_mp4(processed_arr, 'output/output.mp4')
```

### 3. 运行主程序

```bash
python run.py
```

主程序会读取 `data/dual_scroll_background_right_foreground_down.mp4` 视频，进行以下处理：
- 时间卷积模糊
- 垂直和水平方向的边缘检测
- 视频差分处理
- 生成多个输出视频文件到 `output/{convolution_size}/` 目录

## 输出文件

运行主程序后，会在 `output/{convolution_size}/` 目录生成以下文件：
- `output.mp4`: 时间卷积处理后的视频
- `output_1.mp4`: 垂直边缘检测结果
- `output_2.mp4`: 水平边缘检测结果
- `output_3.mp4`: 视频差分结果
- `output_RGB.mp4`: RGB通道合成视频（R=垂直边缘, G=水平边缘, B=差分）
- `output_GB.mp4`: GB通道合成视频（G=垂直边缘, B=水平边缘）

## 技术说明

- **纹理生成**：使用随机黑白方块纹理，通过 `np.roll()` 实现纹理滚动效果
- **视频处理**：使用 OpenCV 进行视频读写，NumPy 进行数组操作
- **卷积处理**：使用 SciPy 的 `ndimage.convolve()` 进行高效卷积计算
- **进度显示**：所有处理函数都添加了详细的进度显示，方便用户了解处理状态

## 注意事项

- 生成视频时，默认分辨率为 1280x720（dual_scrolling 模块为 2048x2048）
- 处理大型视频时，请注意内存消耗
- 输出目录会自动创建，无需手动创建

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。