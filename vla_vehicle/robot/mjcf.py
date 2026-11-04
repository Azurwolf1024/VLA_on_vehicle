"""把上游 ``ArmSpec`` 编译成 MuJoCo (MJCF) 片段。

为什么生成而不手写 XML
------------------------------------------------------------------
手写 XML 就等于把厂商 URDF 里的连杆偏置/关节轴/惯量**抄第二遍**, 一旦上游更新,
仿真与控制器就会默默错开。这里从 ``ArmSpec`` 直接生成, 两者在构造上就同源;
``tests/test_kinematics_match.py`` 再用数值 FK 逐点比对兜底。

坐标系约定与 ``veh_arm.robot.model`` 完全一致::

    T_child = Transl(offset) * Rot(rot0) * Rot(axis, q)

对应 MJCF::

    <body pos="offset" quat="quat(rot0)"> <joint axis="axis" .../> </body>

所以同一个 q 在两边的末端位姿应当逐位相同（测试容差 1e-9）。

视觉/碰撞
------------------------------------------------------------------
上游没有网格文件, 这里用"骨架胶囊": 在参考位形上取值, 把第 i 个连杆画成
``o_i -> o_{i+1}`` 的胶囊。除腕部与手指外**关闭碰撞** (contype=0), 避免臂体
自碰撞在狭窄位形下产生虚假接触; 抓取只依赖指垫与物体的接触。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..deps import ensure_upstream

ensure_upstream()

from veh_arm.robot.model import ArmSpec, Robot  # noqa: E402

from .arm import GripperSpec  # noqa: E402


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _f(v, nd: int = 6) -> str:
    arr = np.asarray(v, dtype=float).ravel()
    return " ".join(f"{x:.{nd}f}" for x in arr)


def quat_from_rot(R: np.ndarray) -> np.ndarray:
    """3x3 旋转 -> 四元数 (w, x, y, z), Shepperd 方法, 对任意旋转稳定。"""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    return np.where(q[0] < 0, -q, q)


def look_at_quat(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """相机放在 ``pos`` 看向 ``target`` 的四元数 (MuJoCo 约定: 视线 = -z, 上 = +y)。"""
    pos = np.asarray(pos, float)
    target = np.asarray(target, float)
    z = pos - target                       # 相机 +z 指向目标的反方向
    z /= np.linalg.norm(z) + 1e-12
    x = np.cross(np.asarray(up, float), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return quat_from_rot(np.column_stack([x, y, z]))


class _Xml:
    """极简缩进式 XML 拼接器（只够本项目用, 不追求通用性）。"""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.depth = 0

    def open(self, tag: str, **attrs) -> "_Xml":
        self.lines.append("  " * self.depth + f"<{tag}{self._attrs(attrs)}>")
        self.depth += 1
        return self

    def close(self, tag: str) -> "_Xml":
        self.depth -= 1
        self.lines.append("  " * self.depth + f"</{tag}>")
        return self

    def leaf(self, tag: str, **attrs) -> "_Xml":
        self.lines.append("  " * self.depth + f"<{tag}{self._attrs(attrs)}/>")
        return self

    def raw(self, text: str) -> "_Xml":
        for ln in text.splitlines():
            self.lines.append("  " * self.depth + ln)
        return self

    @staticmethod
    def _attrs(attrs: dict) -> str:
        out = []
        for k, v in attrs.items():
            if v is None:
                continue
            if isinstance(v, (float, np.floating)):
                s = f"{v:.6g}"
            elif isinstance(v, (np.ndarray, list, tuple)):
                s = _f(v)
            else:
                s = str(v)
            out.append(f'{k}="{s}"')
        return (" " + " ".join(out)) if out else ""

    def text(self) -> str:
        return "\n".join(self.lines)


# --------------------------------------------------------------------------- #
# 结果容器
# --------------------------------------------------------------------------- #
@dataclass
class ArmMJCF:
    xml: str                     # <body name="arm_base"> ... </body> 整段
    joint_names: List[str]
    actuator_names: List[str]    # 与 joint_names 一一对应
    finger_joints: List[str]
    finger_actuators: List[str]
    tcp_site: str
    wrist_camera: str
    tool_body: str


def _skeleton(spec: ArmSpec, q_ref: np.ndarray) -> List[tuple]:
    """参考位形下每个连杆的骨架线段 (a, b, radius), 均在该连杆自身坐标系。"""
    robot = Robot(spec)
    fr = robot.frames(np.eye(3), np.zeros(3), np.asarray(q_ref, float))
    n = spec.n
    segs = []
    for i in range(1, n + 1):
        a_world = fr.o[i]
        b_world = fr.o[i + 1] if i < n else fr.p_ee
        R = fr.R[i]
        a = np.zeros(3)
        b = R.T @ (b_world - a_world)
        L = float(np.linalg.norm(b))
        r = float(np.clip(0.22 * L, 0.018, 0.042))
        segs.append((a, b, r, L))
    return segs


def build_arm(spec: ArmSpec, gripper: GripperSpec | None = None,
              q_ref: np.ndarray | None = None, name: str = "arm",
              mount_pos=(0.0, 0.0, 0.0), mount_euler=(0.0, 0.0, 0.0),
              joint_kp: float = 2400.0, joint_kv: float = 60.0,
              wrist_cam_pos=(0.0, -0.042, 0.008), wrist_cam_fovy: float = 80.0,
              wrist_cam_tilt: float = 18.0
              ) -> ArmMJCF:
    """生成机械臂 + 夹爪 + 腕部相机的 MJCF 片段（不含 actuator）。"""
    gripper = gripper or GripperSpec()
    q_ref = np.asarray(q_ref if q_ref is not None else np.zeros(spec.n), float)
    segs = _skeleton(spec, q_ref)

    x = _Xml()
    x.open("body", name=f"{name}_base", pos=mount_pos, euler=mount_euler)

    def emit(i: int) -> None:
        """递归展开第 i 个连杆 —— MJCF 的子 body 必须**嵌套**在父连杆里。"""
        L = spec.links[i]
        tag = f"{name}_link{i + 1}"
        x.open("body", name=tag, pos=L.offset, quat=quat_from_rot(L.rot0))
        x.leaf("joint", name=f"{name}_j{i + 1}", type="hinge", axis=L.axis,
               range=[float(spec.q_min[i]), float(spec.q_max[i])],
               armature=0.02, damping=0.08, frictionloss=0.05)
        # 惯性: 上游 Link 给的质心/惯量是"关于质心、在连杆系"
        I = np.asarray(L.inertia, float)
        full = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
        x.leaf("inertial", pos=L.com, mass=float(L.mass), fullinertia=full)
        a, b, r, Lb = segs[i]
        if Lb < 5e-3:
            x.leaf("geom", name=f"{tag}_g", type="sphere", pos=a, size=[max(r, 0.025)],
                   rgba="0.42 0.45 0.50 1", contype=0, conaffinity=0)
        else:
            x.leaf("geom", name=f"{tag}_g", type="capsule", fromto=[*a, *b], size=[r],
                   rgba="0.42 0.45 0.50 1", contype=0, conaffinity=0)
        if i == spec.n - 1:
            _tool(x, gripper, name=name, cam_pos=wrist_cam_pos, cam_fovy=wrist_cam_fovy,
                  cam_tilt=wrist_cam_tilt)
        else:
            emit(i + 1)
        x.close("body")

    emit(0)
    x.close("body")

    joints = [f"{name}_j{i + 1}" for i in range(spec.n)]
    fingers = [f"{name}_finger_l", f"{name}_finger_r"]
    return ArmMJCF(
        xml=x.text(),
        joint_names=joints,
        actuator_names=[f"act_{j}" for j in joints],
        finger_joints=fingers,
        finger_actuators=[f"act_{j}" for j in fingers],
        tcp_site=f"{name}_tcp",
        wrist_camera=f"{name}_wrist_cam",
        tool_body=f"{name}_tool",
    )


def _tool(x: _Xml, g: GripperSpec, name: str, cam_pos, cam_fovy: float,
          cam_tilt: float = 18.0) -> None:
    """在法兰上挂平行夹爪 + TCP site + 腕部相机。

    工具系原点 = 法兰 + ``palm_z``; 指垫自掌部下沿 (工具 z=0.012) 向下一段 ``finger_len``,
    因此指尖恰好在法兰 + ``tcp`` 处, 与 IK 用的 TCP_OFFSET 同源。
    """
    x.open("body", name=f"{name}_tool", pos=[0.0, 0.0, g.palm_z])
    x.leaf("inertial", pos=[0.0, 0.0, 0.0], mass=0.02,
           diaginertia=[2e-5, 2e-5, 2e-5])
    # 掌部（参与碰撞: 放件时可能碰到桶壁）
    x.leaf("geom", name=f"{name}_palm", type="box", size=list(g.palm_half),
           rgba="0.22 0.24 0.28 1", contype=1, conaffinity=1, friction="1 0.02 0.001")
    pad_center_z = 0.012 + 0.5 * g.finger_len
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        x.open("body", name=f"{name}_finger_{side}", pos=[0.0, 0.0, 0.012])
        x.leaf("joint", name=f"{name}_finger_{side}", type="slide",
               axis=[0.0, sgn, 0.0], range=[0.0, g.finger_open],
               armature=0.002, damping=0.5)
        x.leaf("inertial", pos=[0.0, 0.0, 0.0], mass=0.01,
               diaginertia=[1e-5, 1e-5, 1e-5])
        x.leaf("geom", name=f"{name}_pad_{side}", type="box",
               pos=[0.0, sgn * (g.finger_gap0 + g.pad_half[1]),
                    pad_center_z - 0.012],
               size=list(g.pad_half),
               rgba="0.95 0.75 0.15 1", contype=1, conaffinity=1,
               friction="2.0 0.05 0.001", condim=4)
        x.close("body")
    # 只作测量用: 尺寸取到几乎为零, 避免挡住腕部相机
    x.leaf("site", name=f"{name}_tcp", pos=[0.0, 0.0, g.tcp - g.palm_z],
           size=[1e-5], rgba="0 0 0 0")
    # 腕部相机: 装在**掌部侧边**并朝工具轴内倾 cam_tilt 度。
    # 同轴装在两指之间时, 一旦夹住物体, 画面就被物体糊满 (物体距镜头只有 2 cm),
    # 策略会丢掉"接下来往哪走"的视野; 侧装前倾则夹爪始终在画面下缘, 前方仍可见。
    t = np.deg2rad(cam_tilt)
    z_c = np.array([0.0, np.sin(t), -np.cos(t)])        # 相机 +z (视线反方向)
    x_c = np.array([1.0, 0.0, 0.0])
    y_c = np.cross(z_c, x_c)
    R_cam = np.column_stack([x_c, y_c, z_c])
    x.leaf("camera", name=f"{name}_wrist_cam", pos=list(cam_pos),
           quat=quat_from_rot(R_cam), fovy=cam_fovy)
    x.close("body")


def actuator_block(arm: ArmMJCF, spec: ArmSpec, gripper: GripperSpec,
                   joint_kp: float = 2400.0, joint_kv: float = 60.0,
                   tau_scale: float = 1.0) -> str:
    """位置伺服执行器（关节力矩上限取上游 ``spec.tau_max``）。"""
    x = _Xml()
    x.open("actuator")
    for i, jn in enumerate(arm.joint_names):
        fmax = float(spec.tau_max[i]) * tau_scale
        x.leaf("position", name=f"act_{jn}", joint=jn, kp=joint_kp, kv=joint_kv,
               forcerange=[-fmax, fmax],
               ctrlrange=[float(spec.q_min[i]), float(spec.q_max[i])])
    for jn in arm.finger_joints:
        x.leaf("position", name=f"act_{jn}", joint=jn, kp=gripper.kp, kv=gripper.kv,
               forcerange=[-gripper.force, gripper.force],
               ctrlrange=[0.0, gripper.finger_open])
    x.close("actuator")
    return x.text()


def camera_xml(name: str, pos, quat, fovy: float) -> str:
    x = _Xml()
    x.leaf("camera", name=name, pos=list(pos), quat=list(quat), fovy=float(fovy))
    return x.text()
