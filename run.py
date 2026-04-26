"""
主程序文件
"""

import os
import numpy as np
from src.video_converter import mp4_to_grayscale_array, gray_array_to_mp4,two_gray_array_to_GB_mp4,three_gray_array_to_RGB_mp4
from src.temporal_convolver import create_temporal_motion_blur, apply_temporal_convolution,apply_frame_convolution,apply_peel_max,apply_diff

arr = mp4_to_grayscale_array("data\\dual_scroll_background_right_foreground_down.mp4")

for convolution_size in range(1,7):

    os.makedirs(f"output/{convolution_size}", exist_ok=True)

    # 卷积核
    convolution = create_temporal_motion_blur(convolution_size, convolution_size, convolution_size)
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
    convolution_sharpen = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]
    ])

    # 模糊化
    processed_arr = apply_temporal_convolution(arr, convolution)

    # 垂直与水平核卷积
    result1=apply_frame_convolution(processed_arr, convolution1)
    result2=apply_frame_convolution(processed_arr, convolution2)
    result3=apply_diff(arr)

    # 池化
    # peel1 = apply_peel_max(tmp1,5)
    # peel2 = apply_peel_max(tmp2,5)

    # 锐化
    # result1 = apply_frame_convolution(result1, convolution_sharpen)
    # result2 = apply_frame_convolution(result2, convolution_sharpen)

    gray_array_to_mp4(processed_arr, f"output/{convolution_size}/output.mp4")
    gray_array_to_mp4(result1, f"output/{convolution_size}/output_1.mp4")
    gray_array_to_mp4(result2, f"output/{convolution_size}/output_2.mp4")
    gray_array_to_mp4(result3, f"output/{convolution_size}/output_3.mp4")
    three_gray_array_to_RGB_mp4(result1,result2,result3,f"output/{convolution_size}/output_RGB.mp4")
    two_gray_array_to_GB_mp4(result1,result2,f"output/{convolution_size}/output_GB.mp4",output_frames_folder=True)
