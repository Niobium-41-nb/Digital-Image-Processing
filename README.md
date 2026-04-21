# SnowVideo 包

SnowVideo 是一个用于生成和处理雪花视频的 Python 包，提供了多种功能，包括生成具有不同移动方向的雪花视频、视频与数组之间的转换、视频帧转 PNG 图片等。

## 功能特性

- ✅ 生成雪花视频，支持多种移动方向设置
  - 预设方向（左、右、上、下、左上、右上、左下、右下）
  - 自定义角度方向（0-360度）
  - 直接设置 x 和 y 方向移动分量
- ✅ 视频与 numpy 数组之间的相互转换
- ✅ 数组轴转置（将 y 轴转置为 z 轴）
- ✅ 视频帧转 PNG 图片

## 安装依赖

在使用 SnowVideo 包之前，需要安装以下依赖：

```bash
pip install opencv-python numpy
```

## 包结构

```
snowvideo/
├── __init__.py          # 包初始化文件
├── video.py             # 雪花视频生成功能
├── converter.py         # 视频与数组转换功能
└── utils.py             # 工具函数（轴转置、视频转PNG）
```

## 使用示例

### 1. 生成雪花视频

#### 使用预设方向

```python
from snowvideo import generate_snow_video

# 生成向右移动的雪花视频
generate_snow_video(direction='right', output_file='snow_right.avi', duration=5)

# 生成向上移动的雪花视频
generate_snow_video(direction='up', output_file='snow_up.avi', duration=5)
```

#### 使用角度方向

```python
from snowvideo import generate_snow_video

# 生成45度方向移动的雪花视频
generate_snow_video(direction=45, output_file='snow_45.avi', duration=5)

# 生成135度方向移动的雪花视频
generate_snow_video(direction=135, output_file='snow_135.avi', duration=5)
```

#### 使用自定义移动分量

```python
from snowvideo import generate_snow_video

# 生成自定义方向移动的雪花视频（向右上移动）
generate_snow_video(dx=0.7, dy=-0.7, output_file='snow_custom.avi', duration=5)
```

### 2. 视频与数组转换

```python
from snowvideo import avi_to_3d_array, array_to_avi

# 将视频转换为数组
video_array = avi_to_3d_array('snow_right.avi')
print(f"视频数组形状: {video_array.shape}")

# 将数组转换为视频
array_to_avi(video_array, 'output_video.avi')
```

### 3. 数组轴转置

```python
from snowvideo import avi_to_3d_array, transpose_y_to_z, array_to_avi

# 读取视频并转置数组
video_array = avi_to_3d_array('snow_right.avi')
transposed_array = transpose_y_to_z(video_array)
print(f"转置后数组形状: {transposed_array.shape}")

# 将转置后的数组转换为视频
array_to_avi(transposed_array, 'output_transposed.avi')
```

### 4. 视频转 PNG 图片

```python
from snowvideo import video_to_pngs

# 将视频的每一帧转换为PNG图片
png_count = video_to_pngs('snow_right.avi', 'output_frames')
print(f"已生成 {png_count} 张PNG图片")
```

## API 参考

### generate_snow_video

```python
generate_snow_video(direction='right', dx=None, dy=None, output_file='snow_video.avi', duration=10, fps=30)
```

- **direction**: 移动方向
  - 字符串: 'left', 'right', 'up', 'down', 'upleft', 'upright', 'downleft', 'downright'
  - 浮点数: 角度值（0-360度），0度为向右，顺时针增加
- **dx**: x方向移动分量（正值向右），如果提供则忽略direction参数
- **dy**: y方向移动分量（正值向下），如果提供则忽略direction参数
- **output_file**: 输出视频文件名
- **duration**: 视频时长（秒）
- **fps**: 帧率

### avi_to_3d_array

```python
avi_to_3d_array(video_path)
```

- **video_path**: 视频文件的路径
- **返回**: 形状为(帧数, 高度, 宽度, 通道数)的numpy数组

### array_to_avi

```python
array_to_avi(video_array, output_path, fps=30)
```

- **video_array**: 形状为(帧数, 高度, 宽度, 通道数)的numpy数组
- **output_path**: 输出.avi文件的路径
- **fps**: 视频帧率
- **返回**: 操作是否成功（布尔值）

### transpose_y_to_z

```python
transpose_y_to_z(arr)
```

- **arr**: 输入的三维或四维数组
- **返回**: 转置后的数组

### video_to_pngs

```python
video_to_pngs(video_path, output_dir)
```

- **video_path**: 视频文件路径
- **output_dir**: 输出图片的目录
- **返回**: 保存的图片数量

## 示例输出

运行示例代码后，你将看到类似以下输出：

```
雪花视频生成完成：snow_right.avi
移动方向：right
视频数组形状: (150, 480, 640, 3)
转置后数组形状: (480, 150, 640, 3)
转置视频生成成功
已生成 150 张PNG图片
```

## 注意事项

- 生成视频时，默认分辨率为 640x480
- 生成的视频为灰度视频
- 处理大型视频时，请注意内存消耗

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。
