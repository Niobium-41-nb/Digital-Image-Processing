"""
主程序文件，用于处理视频的时间卷积模糊。

该脚本加载一个MP4视频文件，将其转换为灰度数组，
应用时间维度的运动模糊卷积，然后将处理后的数组保存为新的MP4视频。
"""

from mp4_to_gray_array import mp4_to_grayscale_array
from gray_array_to_mp4 import gray_array_to_mp4
from f import f, create_temporal_motion_blur

# 加载视频为灰度数组
arr = mp4_to_grayscale_array("kinetic_boundary_waterfall.mp4")

print("arr.shape : \n" , arr.shape)

# 创建时间维度的模糊卷积核
convolution = create_temporal_motion_blur(2, 2, 2)

print("con.shape : \n",convolution.shape)

# 执行卷积操作
processed_arr = f(arr, convolution)

# 将处理后的数组保存为视频
gray_array_to_mp4(processed_arr, "output_video.mp4")