"""图像预处理：**训练与推理必须走同一个函数**。

为什么单独成文件
------------------------------------------------------------------
``image_size`` 只在两处出现：训练时由 ``EpisodeDataset`` 读帧、推理时由 ``BCPolicy``
读观测。两处各写一份，早晚会分叉；分叉的后果不是报错，而是**训练看到的图和
部署看到的图不是一回事**，指标好看、闭环失灵 —— 历史上最贵的一类 bug。

所以这里只放一个函数，两边都调它。

为什么不是 ``img[::step, ::step][:S, :S]``
------------------------------------------------------------------
那个写法在 160 → 96 时 ``step = max(1, 160 // 96) = 1``，于是 ``[:96, :96]``
**把右下角裁掉了** —— 不是缩放，是取左上角。当时的目标碰巧都在左上窗口内
（实测全局相机 u∈[27,55] v∈[51,87]），指标因此正常，但这是靠运气：滑动窗口
再远一点就会整体丢出视野，而且丢的时候没有任何提示。

现在统一成 ``cv2.resize``（``INTER_AREA``，降采样时是最正确的插值）。代价是
物体在缩略图里变小（96² 下垃圾只剩 4–5 px），这一点无法用预处理弥补，
只能靠提高 ``image_size`` 或相机分辨率 —— 见 ``docs/roadmap.md``。
"""

from __future__ import annotations

import numpy as np


def resize_frame(img: np.ndarray, size: int) -> np.ndarray:
    """把 ``(H, W, 3)`` 的 uint8 图缩放到 ``size × size``（不裁剪、不丢视野）。

    尺寸已经对得上时原样返回（零拷贝），避免无意义的插值开销。
    """
    if img.ndim != 3:
        raise ValueError(f"期望 (H,W,3), 收到 {img.shape}")
    if img.shape[0] == size and img.shape[1] == size:
        return img
    import cv2  # 延迟导入: 只跑仿真/采集时不装 opencv 也能用
    interp = cv2.INTER_AREA if size < img.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(img, (int(size), int(size)), interpolation=interp)
