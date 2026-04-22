"""
主程序文件
"""

import numpy as np
from src.video_converter import mp4_to_grayscale_array, gray_array_to_mp4,two_gray_array_to_GB_mp4
from src.temporal_convolver import create_temporal_motion_blur, apply_temporal_convolution,apply_frame_convolution

arr = mp4_to_grayscale_array("data\\dual_scroll_background_right_foreground_down.mp4")

convolution = create_temporal_motion_blur(5, 5, 5)
convolution1 = np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])
convolution2 = np.array([
    [1,1,1],
    [0,0,0],
    [-1,-1,-1]
])
processed_arr = apply_temporal_convolution(arr, convolution)

result1 = apply_frame_convolution(processed_arr, convolution1)
result2 = apply_frame_convolution(processed_arr, convolution2)

gray_array_to_mp4(processed_arr, "output/output_video.mp4")
gray_array_to_mp4(result1, "output/output_video1.mp4")
gray_array_to_mp4(result2, "output/output_video2.mp4")
two_gray_array_to_GB_mp4(result1,result2,"output/output_video_GB.mp4")
