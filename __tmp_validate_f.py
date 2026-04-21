"""
验证卷积函数正确性脚本。

使用随机数据测试f函数的FFT卷积实现与直接卷积的差异，
确保卷积操作的准确性。
"""

import numpy as np
import sys
sys.path.insert(0, r'd:\\数字图像处理')
from f import f

np.random.seed(0)
array = np.random.randint(0, 256, size=(4, 5, 6), dtype=np.uint8)
kernel = np.random.rand(3, 3, 3)

p = (kernel.shape[0] // 2, kernel.shape[1] // 2, kernel.shape[2] // 2)
padded = np.pad(array, ((p[0], p[0]), (p[1], p[1]), (p[2], p[2])), mode='constant', constant_values=0)
direct = np.zeros_like(array, dtype=np.float64)
for z in range(array.shape[0]):
    for y in range(array.shape[1]):
        for x in range(array.shape[2]):
            direct[z, y, x] = np.sum(padded[z:z + kernel.shape[0], y:y + kernel.shape[1], x:x + kernel.shape[2]] * kernel)

fft_res = f(array, kernel).astype(np.float64)
print('max diff', np.max(np.abs(direct - fft_res)))
print('all close', np.allclose(direct, fft_res, atol=1e-6))
