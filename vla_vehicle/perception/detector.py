"""颜色检测器：从 160×160 的 RGB 图里找出目标垃圾。

为什么用颜色而不是学习检测器
------------------------------------------------------------------
这不是"偷懒"，而是要和后面的视觉伺服/学习策略形成**同条件对照**：传统 CV 基线就应该
用传统 CV 的手段。场景里 4 类垃圾的颜色在 HSV 上互不重叠，且语言指令里同时给了
颜色和形状 —— 检测器解析指令取颜色，分割取掩码，这已经是这条流水线需要的全部感知。

工程上只有两个坑，都必须用实测数据定，不能拍：

1. **背景色和物体色在色度上会撞**。甲板棕 (0.60,0.47,0.32) 的归一化色度离黄很近，
   水面 (0.13,0.28,0.38) 离蓝也不远。所以不能只靠"最近色 + 距离阈值"，
   必须叠**连通域面积**：舷外见到的水面是上千像素的大块，垃圾只有几十到几百像素。
2. **光照会改变亮度**。判据取"归一化色度（去掉亮度）× 面积", 不直接比原始 RGB。

检测器输出的是**像素级**结果（质心、面积、包围盒），3D 定位交给
`perception/camera.py` 的反投影 —— 两件事分开，才能在下游做不同的误差分析。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ..sim.objects import COLORS

#: 颜色参考：归一化色度（r,g,b 各自除以和），去掉了亮度这一维。
_CHROMA = {k: np.asarray(v.rgba[:3], float) / float(np.sum(v.rgba[:3]))
           for k, v in COLORS.items()}


@dataclass
class Detection:
    color: str
    centroid: np.ndarray        # (u, v) 像素
    area: int
    bbox: tuple                 # (u0, v0, u1, v1) 含端点
    mask: np.ndarray | None = None

    @property
    def size_px(self) -> float:
        u0, v0, u1, v1 = self.bbox
        return float(max(u1 - u0 + 1, v1 - v0 + 1))


def _chroma_image(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    s = x.sum(axis=2, keepdims=True)
    return x / np.maximum(s, 1e-6)


class ColorDetector:
    """按颜色分割 + 面积过滤，返回候选检测。

    参数都是**实测标定**出来的（见 `scripts/probe_perception.py`），不是拍脑袋：

    * ``chroma_tol=0.13``：水面 (0.13,0.28,0.38) 的归一化色度离蓝只有 0.16，
      阈值放到 0.16 时整片水面会被判成蓝色，蓝物体与水面连成一块后被面积上限滤掉
      —— 实测漏检，且偶尔冒出 1.2 m 的定位误差（质心跑到画面中心）。0.10~0.14
      都能做到 6/6 不漏，取中间值。
    * ``max_area=900``：垃圾在 160² 图里最多 300 多像素（直径 20 px），
      水面碎块动辄上千 —— 面积是挡住背景的**主判据**，色度只是粗筛。
    """

    def __init__(self, chroma_tol: float = 0.13, min_area: int = 18,
                 max_area: int = 900, min_saturation: float = 0.22):
        self.chroma_tol = float(chroma_tol)
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self.min_saturation = float(min_saturation)

    # ------------------------------------------------------------------ #
    def _label_map(self, rgb: np.ndarray) -> np.ndarray:
        """每个像素的最近颜色索引；无匹配为 -1。"""
        ch = _chroma_image(rgb)
        keys = list(_CHROMA)
        ref = np.stack([_CHROMA[k] for k in keys])            # (K,3)
        d = np.linalg.norm(ch[:, :, None, :] - ref[None, None, :, :], axis=3)
        idx = np.argmin(d, axis=2)
        best = np.take_along_axis(d, idx[:, :, None], axis=2)[:, :, 0]
        x = rgb.astype(np.float32) / 255.0
        sat = 1.0 - x.min(axis=2) / np.maximum(x.max(axis=2), 1e-6)
        ok = (best < self.chroma_tol) & (sat > self.min_saturation) & (x.max(axis=2) > 0.12)
        return np.where(ok, idx, -1).astype(np.int16)

    def detect_all(self, rgb: np.ndarray) -> List[Detection]:
        import cv2

        lab = self._label_map(rgb)
        keys = list(_CHROMA)
        out: List[Detection] = []
        for i, key in enumerate(keys):
            m = (lab == i).astype(np.uint8)
            if m.sum() < self.min_area:
                continue
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, cc, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
            for j in range(1, n):
                area = int(stats[j, cv2.CC_STAT_AREA])
                if not (self.min_area <= area <= self.max_area):
                    continue
                u0, v0 = int(stats[j, cv2.CC_STAT_LEFT]), int(stats[j, cv2.CC_STAT_TOP])
                w, h = int(stats[j, cv2.CC_STAT_WIDTH]), int(stats[j, cv2.CC_STAT_HEIGHT])
                # 长宽比上限: 细长的多半是舷墙边缘/反光条纹, 不是物体
                if max(w, h) > 3.2 * max(1, min(w, h)):
                    continue
                out.append(Detection(color=key, centroid=np.asarray(cents[j], float),
                                     area=area, bbox=(u0, v0, u0 + w - 1, v0 + h - 1),
                                     mask=(cc[j] == j)))
        out.sort(key=lambda d: -d.area)
        return out

    def find(self, rgb: np.ndarray, color: str) -> Detection | None:
        cands = [d for d in self.detect_all(rgb) if d.color == color]
        return cands[0] if cands else None


# --------------------------------------------------------------------------- #
# 从语言指令里取目标属性
# --------------------------------------------------------------------------- #
def parse_target(instruction: str) -> Dict[str, str]:
    """把"把红色易拉罐丢进桶里"解析成 ``{color: red, shape: can}``。

    这不是"用特权信息"：指令本来就写在 ``obs.instruction`` 里，任何一条真实流水线
    也必须先把语言落到属性上。传统 CV 语言理解能力有限，恰好只能做到这一层 ——
    这也正是它和 VLA 的差别所在，值得在对比里体现出来。
    """
    zh2key = {v.zh: k for k, v in COLORS.items()}
    from ..sim.objects import SHAPES
    sh2key = {v.zh: k for k, v in SHAPES.items()}
    out: Dict[str, str] = {}
    for zh, k in zh2key.items():
        if zh in instruction:
            out["color"] = k
            break
    for zh, k in sh2key.items():
        if zh in instruction:
            out["shape"] = k
            break
    return out
