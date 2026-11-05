"""针孔相机几何：像素 ↔ 甲板系。

为什么只要甲板系就够
------------------------------------------------------------------
全局相机固定在船上（`vessel` 体里），所以它在**甲板系**的位姿是常数（出厂标定值）。
策略要的是"目标在甲板系哪里"，那就完全不需要知道船在世界哪里 ——
反过来，如果去算世界系坐标，就必须知道甲板位姿，而那是策略看不到的特权量。

所以这里的反投影是"像素 → 甲板系射线 → 与已知平面求交"，一个纯粹的标定问题。
真实系统也是这么做的：相机标定 + 已知工作平面高度。

**这也是传统定位法的固有弱点**：平面高度写死在标定值上（水面 z = -0.12），
船一横摇，真实水面在甲板系里就不是这个高度了，反投影立刻偏。
这个偏差不是实现 bug，而是"开环定位"这件事本身的性质 —— 对比实验要测的就是它。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _quat_to_rot(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class PinholeCamera:
    """甲板系下的针孔相机。``fovy`` 是**垂直**视场角（MuJoCo 约定）。"""

    pos: np.ndarray          # (3,) 甲板系
    quat: np.ndarray         # (4,) wxyz，甲板系；MuJoCo 约定: +x 右, +y 上, -z 视线
    fovy: float              # deg
    width: int
    height: int

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, float)
        self.quat = np.asarray(self.quat, float)
        self.R = _quat_to_rot(self.quat)
        self.f = self.height / (2.0 * np.tan(np.deg2rad(self.fovy) / 2.0))

    # ------------------------------------------------------------------ #
    def ray(self, u: float, v: float) -> np.ndarray:
        """像素点对应的甲板系**单位**方向。"""
        x_n = (float(u) + 0.5 - 0.5 * self.width) / self.f
        y_n = (0.5 * self.height - float(v) - 0.5) / self.f
        d = np.array([x_n, y_n, -1.0])          # 相机系；视线沿 -z
        d = self.R @ d
        return d / np.linalg.norm(d)

    def backproject(self, u: float, v: float, z_plane: float) -> np.ndarray | None:
        """像素 → 甲板系上高度为 ``z_plane`` 的平面交点。射线朝上/平行则返回 None。"""
        d = self.ray(u, v)
        if abs(d[2]) < 1e-9:
            return None
        t = (float(z_plane) - self.pos[2]) / d[2]
        if t <= 0.0:
            return None
        return self.pos + t * d

    def project(self, p) -> tuple[float, float, float] | None:
        """甲板系点 → (u, v, 沿视线深度)。在相机背后返回 None。"""
        d = self.R.T @ (np.asarray(p, float) - self.pos)
        depth = -d[2]
        if depth <= 1e-9:
            return None
        u = self.f * d[0] / depth + 0.5 * self.width - 0.5
        v = 0.5 * self.height - self.f * d[1] / depth - 0.5
        return float(u), float(v), float(depth)

    def pixels_per_meter(self, depth: float) -> float:
        """给定沿视线深度处，一个像素对应多少米（用于把像素误差换算成位移）。"""
        return float(depth) / self.f


def bow_camera(cfg) -> PinholeCamera:
    """按场景配置重建船头全局相机（与 `sim/scene.py` 用的是同一组参数）。"""
    from ..robot.mjcf import look_at_quat

    quat = look_at_quat(cfg.bow_cam_pos, cfg.bow_cam_target)
    return PinholeCamera(pos=np.asarray(cfg.bow_cam_pos, float), quat=quat,
                         fovy=float(cfg.bow_cam_fovy),
                         width=int(cfg.img_width), height=int(cfg.img_height))
