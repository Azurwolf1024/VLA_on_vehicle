"""传统 CV 基线：检测 → 反投影 → 位置控制（detect-then-grasp）。

它和另外两条路线的本质区别
------------------------------------------------------------------
控制律里**没有图像**。整条链路是::

    一张图 ──► 目标颜色 ──► 连通域质心 (u,v) ──► 反投影到平面 ──► 甲板系 (x,y)
                                                                    │
                            位置控制（每步限幅 2 cm）◄───────────────┘

每个阶段**只看一次图**，之后进入开环执行。这也正是工业流水线的常规做法：
视觉贵、控制便宜，所以把视觉用在前端定位上。

必然的误差来源（不是实现 bug）
------------------------------------------------------------------
1. **平面高度写死**。反投影用"水面 z = −0.12 + 物体半高"这个标定值。船一横摇，
   真实水面在甲板系里就不是这个高度了 —— 误差直接进 (x,y)。这是开环定位的性质，
   闭环（图像伺服）能把这一项补回来，对比实验要测的就是它。
2. **单目深度靠先验**。一个像素的横向误差 ≈ 3.2 mm（0.68 m 深度、f=150 px），
   检测质心差半像素就是 1.6 mm；而物体是 3D 的，可见轮廓质心 ≠ 几何中心投影。
3. **只看一次**。等机械臂移过去（0.5 s 以上），物体已经跟着浪走了几厘米。

目标属性从**指令**解析（颜色 + 形状），不是从真值读的 —— 任何真实流水线都得先
把语言落到属性上，这一步没有偷懒。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..control.expert import ScriptedExpert
from ..perception.camera import bow_camera
from ..perception.detector import ColorDetector, parse_target
from ..sim.objects import SHAPES

#: 每个阶段会不会重新看一眼。``False`` = 沿用进入该阶段时那次检测的结果。
_STALE_OK = {"hover", "descend", "close", "lift", "to_bin", "descend_bin",
             "open", "retreat", "settle", "done"}


class CVGraspPolicy(ScriptedExpert):
    """检测式抓取。阶段机复用 `ScriptedExpert`，只把"目标在哪"换成视觉。"""

    name = "cv"

    def __init__(self, env, detector: Optional[ColorDetector] = None):
        super().__init__(env)
        sc = env.cfg.scene
        self.scene = sc
        self.det = detector or ColorDetector()
        self.cam = bow_camera(sc)
        #: 标定的支撑面高度（水面/地面在甲板系的固定值）—— 就是误差来源 ①
        self.plane_z = float(sc.surface_z)
        self._cached: Optional[Tuple[np.ndarray, float, float]] = None
        self._cached_phase: Optional[str] = None
        self._color: Optional[str] = None
        self._half: float = 0.03
        self.hits = 0
        self.misses = 0
        self.look_log: list[dict] = []

    # ------------------------------------------------------------------ #
    def reset(self, instruction: str = "") -> None:
        super().reset(instruction)
        attrs = parse_target(instruction)
        self._color = attrs.get("color")
        shape = attrs.get("shape")
        self._half = float(SHAPES[shape].half_height) if shape in SHAPES else 0.03
        self._cached = None
        self._cached_phase = None
        self.hits = 0
        self.misses = 0
        self.look_log = []

    # ------------------------------------------------------------------ #
    def _detect(self, obs) -> Optional[np.ndarray]:
        """看一眼全局相机，返回甲板系目标中心；失败返回 None。"""
        if self._color is None:
            return None
        d = self.det.find(obs.global_rgb, self._color)
        if d is None:
            return None
        # 反投影到"水面 + 物体半高"——物体躺在水面上，所以这就是它的中心高度。
        # 漏掉半高会偏 10–50 mm（视线穿过中心后还要延伸到水面）。
        p = self.cam.backproject(float(d.centroid[0]), float(d.centroid[1]),
                                 self.plane_z + self._half)
        if p is None:
            return None
        self.look_log.append(dict(u=float(d.centroid[0]), v=float(d.centroid[1]),
                                  area=int(d.area), x=float(p[0]), y=float(p[1]),
                                  phase=self.phase))
        return p

    def _look(self, obs=None):
        """(物体中心, 半高, 支撑面高度)。每个阶段**只看一次图**，阶段内沿用。"""
        if self._cached_phase == self.phase and self._cached is not None:
            return self._cached

        p = self._detect(obs) if obs is not None else None
        if p is not None:
            self.hits += 1
            self._cached = (p, self._half, self.plane_z)
            self._cached_phase = self.phase
            return self._cached
        self.misses += 1
        if self._cached is not None:
            # 这一阶段没看到（被夹爪/船体挡住是常事），沿用上一次的位置。
            self._cached_phase = self.phase
            return self._cached
        # 从来没看到过: 把"目标"设成当前 TCP，动作归零，让阶段自然超时失败。
        anchor = np.asarray(obs.tcp, float) if obs is not None else np.zeros(3)
        self._cached = (anchor.copy(), self._half, self.plane_z)
        self._cached_phase = self.phase
        return self._cached

    # ------------------------------------------------------------------ #
    def info(self) -> dict:
        d = super().info()
        d.update(hits=self.hits, misses=self.misses, color=self._color,
                 looks=len(self.look_log))
        return d
