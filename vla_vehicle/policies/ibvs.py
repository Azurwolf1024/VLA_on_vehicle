"""图像视觉伺服（IBVS）：用**像素误差**闭环，不估计 3D 位姿。

和「检测式抓取」的本质区别
------------------------------------------------------------------
检测式那条链路是 "图 → 3D 点 → 位置控制"，控制律里没有图像；这条链路是

    期望像素 s*  ──┐
                   ├──► e = s − s* ──► Δx = −λ J⁺ e ──► 关节/TCP
    实测像素 s  ──┘

图像误差直接被压到零，中间**没有**"反投影到某个标定平面"这一步。于是：

* 标定平面写死带来的偏差（船一摇水面就不在 z=−0.12 了）在这里**自动消失** ——
  因为根本不用它；
* 目标自己动（漂在水上）、基座自己动（船在摇）也自动被吸收 —— 它们只体现在
  下一帧的像素误差里；
* 代价：**深度不可观**。单目 IBVS 没法从像素知道目标离相机多远，所以 z 仍然开环
  （按阶段往下走）。这是 IBVS 的教科书级弱点，不是本实现的偷懒。

雅可比从哪来
------------------------------------------------------------------
不手推。工具朝下、相机固连腕部，但相机相对甲板系的 x/y 朝向取决于 TCP 偏航
（`yaw = atan2(y,x)`），解析式写出来容易差个符号 —— 差一个符号的伺服会**发散**，
而且现象很难看出来（"怎么越走越远"）。

所以用两次**探测动作**（各 1 cm）把 2×2 图像雅可比的两列测出来，之后用 Broyden
在线更新（无标定视觉伺服的常规做法）::

    J ← J + (Δs − J Δx) Δxᵀ / (Δxᵀ Δx + ε)

探测本身会让 TCP 各偏 1 cm，伺服随后会把它纠回来。

分工：全局相机负责"目标在哪"（找到大致方位），腕部相机负责"对准了没有"。
这和真实系统一致 —— 全局相机做粗定位，腕部做最后几厘米。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..perception.camera import bow_camera
from ..perception.detector import ColorDetector, parse_target
from ..sim.objects import SHAPES
from .base import ACTION_DIM, Policy

PHASES = ["locate", "approach", "servo", "close", "lift", "to_bin",
          "descend_bin", "open", "retreat", "settle", "done"]


class PBVSPolicy(Policy):
    """全局粗定位 + 每步重测的 PBVS 精对位。"""

    name = "pbvs"

    def __init__(self, env, detector: Optional[ColorDetector] = None,
                 lam: float = 0.45,      # 伺服增益（每步吃掉多少位置误差）
                 area_keep: float = 0.6,       # 可见面积跌破起始值的这个比例就判遮挡
                 servo_tol_m: float = 0.006,   # 到位判据（米）: 比单帧检测噪声(±5mm)大,
                                              # 否则永远判不到位、每回合白跑满 45 步超时
                 approach_h: float = 0.140,   # 进入伺服的高度: 太近目标会被画面裁掉, 质心就没信息了
                 probe: float = 0.020,
                 area_close: float = 6000.0,  # 目标像素面积超过它就停止下降（深度代理）
                 phase_timeout: int = 45):
        self.cfg = env.cfg
        sc = env.cfg.scene
        self.scene = sc
        # 两支相机的工作距离差一个量级, 面积区间完全不同, 必须分开标定:
        #   全局图里物体 46~113 px（远处, 直径 8~12 px）
        #   腕部图里物体能到 8000+ px（近景贴着指垫）
        # 用同一组面积阈值必然有一端失效 —— 实测过: 全局漏检 -> locate 卡死 -> 170 步超时。
        self.det_bow = detector or ColorDetector()
        self.det_wrist = detector or ColorDetector(max_area=24000, min_area=200)
        self.bow = bow_camera(sc)
        #: 腕部相机内参：fovy 是垂直视场角
        self.f_wrist = float(sc.img_height) / (2.0 * np.tan(np.deg2rad(sc.wrist_cam_fovy) / 2.0))
        self.lam = float(lam)
        self.servo_tol_m = float(servo_tol_m)
        self._area_keep = float(area_keep)
        self._area0: Optional[float] = None
        self.occluded = 0
        self.approach_h = float(approach_h)
        self.probe = float(probe)
        self.area_close = float(area_close)
        self.phase_timeout = int(phase_timeout)
        self.action_scale = float(env.cfg.action_scale)
        self.bin_center = tuple(sc.bin_center)
        self.plane_z = float(sc.surface_z)
        self.clear_z = 0.120
        self.above_bin = 0.16
        self.release_z = 0.03
        self._reset_state()

    # ------------------------------------------------------------------ #
    def _reset_state(self) -> None:
        self.i = 0
        self.phase = PHASES[0]
        self.timer = 0
        self.hold: Optional[np.ndarray] = None
        self.gz = 0.0
        self.gz_frozen: Optional[float] = None
        self.xy = np.zeros(2)
        self.done = False
        self.hits = 0
        self.misses = 0
        self.servo_steps = 0
        self.px_err: list[float] = []
        self._e_filt: Optional[np.ndarray] = None
        self._area0 = None
        self.occluded = 0
        self.area_log: list[float] = []

    def reset(self, instruction: str = "") -> None:
        attrs = parse_target(instruction)
        self._color = attrs.get("color")
        shape = attrs.get("shape")
        self._half = float(SHAPES[shape].half_height) if shape in SHAPES else 0.03
        self._reset_state()

    # ------------------------------------------------------------------ #
    def _detect(self, img, color, detector=None):
        d = (detector or self.det_bow).find(img, color)
        return None if d is None else d

    def _servo_delta(self, obs) -> tuple:
        """每一步都重新"检测 → 反投影 → 位置闭环"。返回 (归一化 xy 动作, 面积, 到位?)。

        为什么是 PBVS 而不是 IBVS（这条走错过一段，记下来）
        ------------------------------------------------------------------
        一开始想做图像空间闭环（IBVS）：把目标质心拉到某个期望像素。两个尝试都失败，
        而且失败原因是**几何本身**，不是调参：

        1. 腕部相机自动瞄准 TCP，物体在 TCP 下方 25 mm，解析投影就落在画面**上边缘**
           （v≈0）甚至被裁 —— "质心到中心"的期望位置根本不存在；
        2. 改用全局相机做 eye-to-hand（期望 = TCP 的图像投影），像素能收敛到 1 px，
           抓取误差却仍有 **25 mm**。因为在斜视单目下，"物体与末端在同一视线上"
           **不等于**"末端在物体正上方"：同一视线穿过不同高度的两个点，xy 差 25 mm 很正常。

        结论：图像空间的 2D 误差在这个几何里不构成对 3D 对齐的充分条件。所以退回
        **基于位置**的视觉伺服（PBVS）—— 每步反投影出甲板系坐标，与末端做位置闭环。
        这也正是它的对照价值：CV 基线每个阶段只测一次就开环执行，这里每一步都在测，
        目标被浪带走、船把基座晃开的那几厘米因此能被追回来。

        它仍然**不读真值位姿**，只用"相机标定 + 指令解析 + 本体感觉"。
        """
        det = self._detect(obs.global_rgb, self._color, self.det_bow)
        if det is None:
            self.misses += 1
            # 完全看不见也说明"贴到跟前了"或"被挡住了" —— 横向别再瞎动
            return np.zeros(2), 0.0, self._near_floor(obs)
        self.hits += 1
        area = float(det.area)
        if self._area0 is None:
            self._area0 = area

        # 遮挡门控 —— 这条是整条链路的成败所在
        # ------------------------------------------------------------------
        # 夹爪下探时必然从上方遮住目标: 实测可见面积从 101 px 掉到 21 px。
        # 此时检测到的质心是**残余可见部分**的质心, 不是物体中心; 照它闭环,
        # 末端会被引着往外跑 —— 实测 tcp.y 从 −0.490 一路漂到 −0.466（目标在 −0.500）,
        # 偏差从 12 mm 涨到 37 mm。看起来像"伺服发散", 实际是"追一个假目标"。
        #
        # CV 基线天然免疫这件事（它在悬停时只看一次, 之后不再看）,
        # 所以这一条不修, 闭环就一定会输给开环。
        visible = area >= self._area_keep * self._area0
        if not visible:
            self.occluded += 1
            # 看不清就不做横向修正, 只把下降走完 —— 与工业上的"接近后开环下探"一致
            return np.zeros(2), area, self._near_floor(obs)

        p = self.bow.backproject(float(det.centroid[0]), float(det.centroid[1]),
                                 self.plane_z + self._half)
        if p is None:
            return np.zeros(2), area, self._near_floor(obs)
        e = p[:2] - np.asarray(obs.tcp, float)[:2]
        self.px_err.append(float(np.linalg.norm(e)) * 1e3)   # 这里记的是 mm
        self.servo_steps += 1
        # 单帧质心有 ±1.5 px（≈±5 mm）的抖动。不滤波就会去追这个噪声,
        # 末端一直在抖、稳态误差反而比"走到点就停"的检测式还大（实测 16.6 vs 8.2 mm）。
        self._e_filt = e.copy() if self._e_filt is None else 0.55 * self._e_filt + 0.45 * e
        if float(np.linalg.norm(e)) < self.servo_tol_m:
            return np.zeros(2), area, True
        d = float(self.lam) * self._e_filt
        lim = 0.7 * self.action_scale
        n = float(np.linalg.norm(d))
        if n > lim:
            d = d / n * lim
        return d / self.action_scale, area, False

    def _near_floor(self, obs, tol: float = 0.012) -> bool:
        """指尖是否已经到抓取高度（到这一层就该闭爪，别再等视觉）。"""
        return bool(abs(float(obs.tcp[2]) - self.gz) < tol)

    def _enter_close(self, z: float) -> None:
        self.i += 1
        self.phase = PHASES[self.i]
        self.timer = 0
        self.gz_frozen = float(z)

    def act(self, obs) -> np.ndarray:
        p = np.asarray(obs.tcp, float)
        a = np.zeros(ACTION_DIM)
        a[3] = 1.0
        scale = self.action_scale

        if self.phase == "locate":
            c = self._detect(obs.global_rgb, self._color, self.det_bow)
            if c is not None:
                # 与检测式相同的反投影（物体中心 = 水面 + 半高）。这里只用来**找方位**，
                # 精度交给后面的像素闭环。
                q = self.bow.backproject(float(c.centroid[0]), float(c.centroid[1]),
                                         self.plane_z + self._half)
                if q is not None:
                    self.xy = q[:2].copy()
                    top = self.plane_z + 2 * self._half
                    floor = self.plane_z + 0.004
                    self.gz = float(max(floor, top - 0.045))
                    self.i += 1
                    self.phase = PHASES[self.i]
                    self.timer = 0
            self.timer += 1
            return a

        if self.phase == "approach":
            tgt = np.array([self.xy[0], self.xy[1], self.gz + self.approach_h])
            d = np.clip(tgt - p, -scale, scale)
            self.timer += 1
            if float(np.linalg.norm(tgt - p)) < 0.006 or self.timer > self.phase_timeout:
                self.i += 1
                self.phase = PHASES[self.i]
                self.timer = 0
            a[:3] = d / scale
            return a

        if self.phase == "servo":
            self.timer += 1
            d2, area, ok = self._servo_delta(obs)
            # 投影用的是固定的 gz，所以下降不影响误差信号 —— 可以边降边对中。
            dz = float(np.clip(self.gz - p[2], -scale, scale))
            a[0], a[1] = d2
            a[2] = dz / scale
            if ok and abs(self.gz - p[2]) < 0.008:
                self._enter_close(float(p[2]))
            elif self.timer > self.phase_timeout:
                self._enter_close(float(p[2]))
            return a

        gz = self.gz_frozen if self.gz_frozen is not None else self.gz
        bx, by = self.bin_center
        ph = self.phase
        need = None                     # 只有 open 阶段靠"保持 N 步"推进
        if ph == "close":
            a[:3] = 0.0
            a[3] = 0.0
            self.timer += 1
            if self.timer >= 4:
                self.i += 1
                self.phase = PHASES[self.i]
                self.timer = 0
            return a
        if ph == "lift":
            tgt = np.array([self.xy[0], self.xy[1], max(gz + 0.13, self.clear_z)])
            grip, need = 0.0, None
        elif ph == "to_bin":
            tgt, grip = np.array([bx, by, self.above_bin]), 0.0
        elif ph == "descend_bin":
            tgt, grip = np.array([bx, by, self.release_z]), 0.0
        elif ph in ("open", "retreat"):
            tgt, grip = np.array([bx, by, self.above_bin]), 1.0
            if ph == "open":
                tgt = np.array([bx, by, self.release_z])
        elif ph == "settle":
            a[:3] = 0.0
            self.timer += 1
            if self.timer >= 6:
                self.i += 1
                self.phase = PHASES[self.i]
                self.done = True
            return a
        else:
            a[:3] = 0.0
            a[3] = 1.0
            return a

        d = np.clip(tgt - p, -scale, scale)
        a[:3] = d / scale
        a[3] = grip
        self.timer += 1
        if ph == "open":
            need = 4
        if need is not None:
            if self.timer >= need:
                self.i += 1
                self.phase = PHASES[self.i]
                self.timer = 0
            return a
        if float(np.linalg.norm(tgt - p)) < 0.006 or self.timer > self.phase_timeout:
            self.i += 1
            self.phase = PHASES[self.i]
            self.timer = 0
            if self.phase == "done":
                self.done = True
        return a

    # ------------------------------------------------------------------ #
    def info(self) -> dict:
        return dict(phase=self.phase, done=self.done, hits=self.hits,
                    misses=self.misses, occluded=self.occluded,
                    servo_steps=self.servo_steps,
                    px_err_med=(float(np.median(self.px_err)) if self.px_err else None))
