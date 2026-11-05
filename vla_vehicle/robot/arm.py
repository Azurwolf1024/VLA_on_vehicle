"""机械臂模型: 直接复用上游 ``veh_arm`` 的 NERO 7-DOF 模型。

这里只做三件事:
1. 从上游注册表拿到 ``ArmSpec`` / ``Robot``（厂商 URDF 解析、限位、惯量全部沿用）;
2. 定义**工具坐标系**(TCP)与夹爪几何 —— 上游模型只到法兰, 抓取需要工具;
3. 给出一组工程上稳定可用的标称位形（ready / 收起）。

上游模型是"关节链 + 法兰"，下游所有量（FK / 雅可比 / 逆解）都在**基座系**表述，
这正是载具作业需要的口径: 目标写在甲板系里, 基座随船运动, 内环只负责局部跟踪。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..deps import ensure_upstream

ensure_upstream()

from veh_arm.robot.model import ArmSpec, Robot  # noqa: E402
from veh_arm.robot.registry import get_robot_profile  # noqa: E402

ARM_NAME = "nero"

# 法兰 -> TCP (夹爪指尖) 的距离, 沿末端连杆 +z (法兰法向) 伸出。
TCP_OFFSET = 0.105

# 抓取时的工具朝向: 工具 z 轴指向甲板 (-z), 保证"自上而下"夹取。
def topdown_rotation(yaw: float = 0.0) -> np.ndarray:
    """返回工具朝下、绕竖直轴偏航 ``yaw`` 的姿态（基座系）。

    列向量为工具三轴: **y = 指爪开合方向**, x = 与之正交的进给方向, z = 指向下方。
    ``yaw=0`` 时 x=(1,0,0)、y=(0,-1,0)、z=(0,0,-1) —— ``READY_Q`` 就是按这个姿态解出来的,
    两者必须一致, 否则内环第一步就会出现 ~90° 的姿态误差。

    注意 y 轴是 ``(sin yaw, -cos yaw)`` 而不是 ``(cos yaw, sin yaw)``:
    为了让 z 朝下同时保持右手系, 这里对 y 取了负号。写"物体朝向"时要按
    ``local_y ∥ y_t`` 反解 (得 ψ = yaw), 按 (cos, sin) 去套会正好差 90°
    —— 后果是长方体被宽边塞进夹爪, 指垫压进物体 8 mm, 下探直接卡死。
    """
    c, s = np.cos(yaw), np.sin(yaw)
    x_t = np.array([c, s, 0.0])
    y_t = np.array([s, -c, 0.0])
    z_t = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x_t, y_t, z_t])


#: TCP 位于 (0.28, 0, 0.20)、工具朝下的 IK 解 —— 场景里的"准备位姿"。
#: 由 ``scripts/calibrate_ready.py`` 复核（可操作度 0.0204, 各关节离限位 >= 0.23 rad）。
READY_Q = np.array([1.1484, -0.5525, 1.5619, 1.9012, -0.9940, 0.2752, -0.6078])

#: TCP 抬到 0.35 m 高、靠近基座时的收起位形（不挡视野, 便于回合开始时的观测）。
STOW_Q = np.array([1.35, -0.75, 1.20, 1.95, -0.75, 0.30, -0.55])


@dataclass(frozen=True)
class GripperSpec:
    """二指平行夹爪的几何/驱动参数 (米 / 牛)。

    夹爪不在厂商 URDF 里, 是本仓库补的**仿真执行器**; 真机换成对应夹爪的 URDF 即可,
    只需保持 ``tcp`` 与 ``finger_*`` 语义一致。

    几何自洽约束: ``palm_z + finger_len == tcp``,
    即"指尖(TCP) = 法兰法向伸出 ``tcp``" 与 IK 用的 ``TCP_OFFSET`` 完全一致。
    """

    palm_z: float = 0.020             # 掌部中心相对法兰的高度
    palm_half: tuple = (0.030, 0.026, 0.012)
    finger_len: float = 0.073         # 指垫长度 (自掌部下沿到指尖)
    finger_open: float = 0.032        # 单指最大行程 (最大开口 84 mm)
    finger_gap0: float = 0.010        # q=0 时指垫内侧面到工具轴的距离
    pad_half: tuple = (0.012, 0.006, 0.0365)
    force: float = 30.0               # 单指最大夹持力
    kp: float = 500.0
    kv: float = 8.0
    tcp: float = TCP_OFFSET

    def __post_init__(self) -> None:
        expect = self.palm_z + 0.012 + self.finger_len
        if abs(expect - self.tcp) > 1e-9:
            raise ValueError(f"夹爪几何不自洽: palm_z+0.012+finger_len={expect:.4f} != tcp={self.tcp:.4f}")

    @property
    def max_width(self) -> float:
        """最大开口宽度 (两指内侧间距)。"""
        return 2.0 * (self.finger_gap0 + self.finger_open)

    @property
    def min_width(self) -> float:
        return 2.0 * self.finger_gap0

    def ctrl_from_norm(self, g) -> np.ndarray:
        """归一化开合指令 [0(闭合), 1(张开)] -> 单指行程。"""
        g = np.clip(np.asarray(g, dtype=float), 0.0, 1.0)
        return g * self.finger_open

    def norm_from_width(self, width: float) -> float:
        """由目标开口宽度反解归一化指令（专家/标注用）。"""
        q = 0.5 * float(width) - self.finger_gap0
        return float(np.clip(q / self.finger_open, 0.0, 1.0))


def load_arm(name: str = ARM_NAME) -> tuple[Robot, ArmSpec, dict]:
    """加载上游机械臂模型。

    返回 ``(Robot, ArmSpec, profile_meta)``。``profile_meta`` 里带 home 位形等元数据,
    避免下游再各自硬编码。

    NERO 走 :func:`~vla_vehicle.robot.urdf_fix.load_nero_spec`: 运动学/限位/质量与上游
    完全一致, 只是把上游漏读的**转动惯量**按 URDF 属性补回来 (见该模块文档)。
    """
    profile = get_robot_profile(name)
    if name == "nero":
        from veh_arm.robot.nero import DEFAULT_URDF, NERO_DQ_MAX, NERO_TAU_MAX
        from .urdf_fix import load_nero_spec
        spec = load_nero_spec(DEFAULT_URDF, ee_link="link7", tool_mass=0.5,
                              dq_max=NERO_DQ_MAX, tau_max=NERO_TAU_MAX)
    else:
        spec = profile.build_spec()
    robot = Robot(spec)
    meta = dict(
        name=name,
        display=profile.name,
        dof=robot.n,
        control_mode=profile.control_mode,
        q_min=robot.q_min,
        q_max=robot.q_max,
        dq_max=robot.dq_max,
        tau_max=robot.tau_max,
    )
    return robot, spec, meta
