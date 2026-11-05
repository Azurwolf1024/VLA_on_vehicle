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
**视觉分两档, 由 ``visual`` 选择:**

``"mesh"``    用上游 ``models/nero_stl/*.stl`` —— 厂商 SolidWorks 导出的真实外壳
              (agilex agx_arm_urdf, 就是 Blender 渲染用的那一套)。STL 顶点**就在
              URDF link 系里**(已核对: link2 沿 +y 伸 261 mm、link3 沿 -z 206 mm,
              与连杆偏置吻合), 所以以 identity 挂到对应 body 即可, 不需要任何补偿变换。
``"capsule"`` 骨架胶囊: 在参考位形上把第 i 个连杆画成 ``o_i -> o_{i+1}``。
              没有网格文件时的回落, 也是训练/评测要压渲染开销时的快档。

两档都**关闭碰撞** (contype=0), 避免臂体自碰撞在狭窄位形下产生虚假接触;
抓取只依赖指垫与物体的接触 —— 换视觉不会改变任何物理量。

夹爪与法兰**不用网格**: 厂商 flange_link 相对 link7 有 90° 相位差
(``rpy=-1.5708 0 -1.5708``), 真实夹爪行程是 100 mm/侧, 而本仿真的指垫只有
32 mm 行程且形状/摩擦是标定的接触几何。硬套网格会让"看到的"和"碰到的"分家,
比省这几个三角面划算得多。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np

from ..deps import ensure_upstream, veh_arm_root

ensure_upstream()

from veh_arm.robot.model import ArmSpec, Robot  # noqa: E402

from .arm import GripperSpec  # noqa: E402


# --------------------------------------------------------------------------- #
# 真实网格（上游资产, 运行期解析, 不复制）
# --------------------------------------------------------------------------- #
#: 上游仓库里 STL 所在子目录
MESH_SUBDIR = ("models", "nero_stl")

#: 连杆 -> STL 文件名。只有这 8 件进仿真（见模块 docstring: 法兰/夹爪不用网格）。
_MESH_FILES = {
    "base_link": "base_link.stl",
    "link1": "link1.stl",
    "link2": "link2.stl",
    "link3": "link3.stl",
    "link4": "link4.stl",
    "link5": "link5.stl",
    "link6": "link6.stl",
    "link7": "link7.stl",
}

#: 按连杆给色。"银灰铝合金 + 深灰关节盖"——STL 不带材质, 只能靠分色拉出层次,
#: 否则 27 万面渲染出来是一坨同色的灰。
_MESH_RGBA = {
    "base_link": "0.25 0.27 0.30 1",
    "link1": "0.28 0.30 0.33 1",
    "link2": "0.74 0.75 0.77 1",
    "link3": "0.74 0.75 0.77 1",
    "link4": "0.30 0.32 0.35 1",
    "link5": "0.74 0.75 0.77 1",
    "link6": "0.30 0.32 0.35 1",
    "link7": "0.55 0.57 0.60 1",
}


@lru_cache(maxsize=1)
def mesh_dir() -> str | None:
    """上游 STL 目录的绝对路径 (正斜杠); 找不到返回 ``None``。

    刻意**不复制**到本仓库: 网格是上游 ``agx_arm_urdf`` 的资产, 复制一份就等于
    承诺"我们的副本永远和上游一致", 这条承诺迟早会被打破。
    """
    try:
        root = Path(veh_arm_root())
    except Exception:
        return None
    d = root.joinpath(*MESH_SUBDIR)
    if not all((d / f).is_file() for f in _MESH_FILES.values()):
        return None
    return d.as_posix()


@lru_cache(maxsize=1)
def mesh_assets() -> str:
    """``<mesh>`` 声明行, 由 scene 放进了 ``<asset>``。"""
    d = mesh_dir()
    if d is None:
        return ""
    x = _Xml()
    for key, fname in _MESH_FILES.items():
        x.leaf("mesh", name=f"nero_{key}", file=fname,
               scale=[1.0, 1.0, 1.0])
    return x.text()


@lru_cache(maxsize=1)
def base_inertial() -> tuple | None:
    """上游 URDF 里 ``base_link`` 的 ``(质量, 质心, 惯量矩阵)``。

    为什么必须显式写出来: ``arm_base`` 这个 body 上只有视觉 geom。**没有**
    ``<inertial>`` 时 MuJoCo 会按 geom 的体积当**实心体**估质量 —— 换成真实外壳
    后基座凭空多出 0.73 kg (总量 4.32 → 5.05 kg)。基座焊在载具上、没有自由度,
    对臂的 ``M``/``bias`` 影响实测为 0 (1e-16), 但"整臂质量"是要跟厂商参数对账的
    数字, 不能让它随视觉档漂移。
    """
    try:
        from veh_arm.robot.nero import DEFAULT_URDF
        from .urdf_fix import read_link_inertials
        com, I, mass = read_link_inertials(DEFAULT_URDF).get(
            "base_link", (np.zeros(3), np.zeros((3, 3)), 0.0))
    except Exception:
        return None
    if mass <= 0.0:
        return None
    return float(mass), np.asarray(com, float), np.asarray(I, float)


def _resolve_visual(visual: str) -> str:
    """``"auto"`` 有网格就用网格, 否则回落胶囊。"""
    if visual == "auto":
        return "mesh" if mesh_dir() is not None else "capsule"
    if visual not in ("mesh", "capsule"):
        raise ValueError(f"未知 visual={visual!r}; 可选 auto/mesh/capsule")
    if visual == "mesh" and mesh_dir() is None:
        raise FileNotFoundError(
            "visual='mesh' 但找不到上游 STL。\n"
            f"期望路径: <upstream>/{'/'.join(MESH_SUBDIR)}/*.stl\n"
            "改用 visual='capsule' 或确认 VLA_VEHICLE_UPSTREAM 指向的上游仓库带 models/nero_stl。")
    return visual


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
    visual: str = "capsule"      # 实际生效的视觉档 ("mesh" | "capsule")
    mesh_assets: str = ""        # <mesh> 声明, 供 <asset> 段
    mesh_dir: str = ""           # <compiler meshdir=...>


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
              wrist_cam_pos=(0.0, -0.075, 0.010), wrist_cam_fovy: float = 80.0,
              wrist_cam_tilt: float | None = None, wrist_cam_ahead: float = 0.0,
              with_tool: bool = True, visual: str = "auto"
              ) -> ArmMJCF:
    """生成机械臂 + 夹爪 + 腕部相机的 MJCF 片段（不含 actuator）。

    ``visual``: ``"mesh"`` 用上游真实 STL, ``"capsule"`` 用骨架胶囊, ``"auto"``
    有网格就用网格。视觉档**只影响渲染**, 不影响任何物理量。
    """
    gripper = gripper or GripperSpec()
    q_ref = np.asarray(q_ref if q_ref is not None else np.zeros(spec.n), float)
    vis = _resolve_visual(visual)
    segs = _skeleton(spec, q_ref)
    # 网格名以 spec 里 base_link 的层数为准: 厂商给 7 个转动 link, 编号 1..7
    if vis == "mesh" and spec.n != len(_MESH_FILES) - 1:
        raise ValueError(
            f"网格按 {len(_MESH_FILES) - 1} 个转动 link 准备, 但 ArmSpec.n={spec.n}; "
            "换臂型时请同时更新 _MESH_FILES")

    def vis_geom(tag: str, key: str, a, b, r, Lb: float) -> None:
        """给第 i 个连杆画视觉 geom（网格或胶囊, 二选一, 不重叠）。"""
        if vis == "mesh":
            x.leaf("geom", name=f"{tag}_vis", type="mesh", mesh=f"nero_{key}",
                   rgba=_MESH_RGBA[key], contype=0, conaffinity=0)
            return
        if Lb < 5e-3:
            x.leaf("geom", name=f"{tag}_g", type="sphere", pos=a, size=[max(r, 0.025)],
                   rgba="0.42 0.45 0.50 1", contype=0, conaffinity=0)
        else:
            x.leaf("geom", name=f"{tag}_g", type="capsule", fromto=[*a, *b], size=[r],
                   rgba="0.42 0.45 0.50 1", contype=0, conaffinity=0)

    x = _Xml()
    x.open("body", name=f"{name}_base", pos=mount_pos, euler=mount_euler)
    bi = base_inertial()
    if bi is not None:
        m_b, com_b, I_b = bi
        x.leaf("inertial", pos=list(com_b), mass=m_b,
               fullinertia=[I_b[0, 0], I_b[1, 1], I_b[2, 2],
                            I_b[0, 1], I_b[0, 2], I_b[1, 2]])
    vis_geom(f"{name}_base", "base_link", np.zeros(3), np.zeros(3), 0.04, 0.0)

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
        vis_geom(tag, f"link{i + 1}", a, b, r, Lb)
        if i == spec.n - 1:
            if with_tool:
                _tool(x, gripper, name=name, cam_pos=wrist_cam_pos,
                      cam_fovy=wrist_cam_fovy, cam_tilt=wrist_cam_tilt,
                      cam_ahead=wrist_cam_ahead)
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
        visual=vis,
        mesh_assets=mesh_assets() if vis == "mesh" else "",
        mesh_dir=(mesh_dir() or "") if vis == "mesh" else "",
    )


def _tool(x: _Xml, g: GripperSpec, name: str, cam_pos, cam_fovy: float,
          cam_tilt: float | None = None, cam_ahead: float = 0.0) -> None:
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
    # 腕部相机: 装在掌部**侧前方**, 自动瞄准 TCP。
    #
    # 两个装法都被实测否掉了:
    #  1. 同轴装在两指之间 —— 一夹住物体, 物体距镜头 2 cm, 画面被它糊满;
    #  2. 只往 -y(指爪开合方向) 侧移 —— 相机落在**指垫背面**。指垫在工具系
    #     z∈[0.029,0.061]、相机在 z=0.020, 镜头顺着 +z 看过去先撞上垫子,
    #     画面中间是一根巨大的黄色"柱子", 只有边缘能漏出一点物体。
    # 现在沿 **x(与开合方向正交)** 侧移 + 前移到指垫前方, 从侧前方斜看夹持点:
    # 两指的缝隙是敞开的, 掌部在 x 方向只有 30 mm, 挡不住。
    cam_pos = np.asarray(cam_pos, float)
    # 瞄点 = TCP 沿工具轴再往前 cam_ahead。指垫 73 mm 长、离镜头只有几厘米,
    # 瞄准 TCP 时它必然占掉大半个画面; 往前瞄就能把夹爪压到画面下缘,
    # 保留"接下来往哪走"的视野 (真实腕部相机也是这么装的)。
    target = np.array([0.0, 0.0, g.tcp - g.palm_z + cam_ahead])
    d = target - cam_pos
    dist = float(np.linalg.norm(d))
    z_c = -d / max(dist, 1e-6)                          # 相机 +z = 视线反方向
    if cam_tilt is not None:                            # 给定额外俯仰: 绕 y 轴再抬一点
        t = np.deg2rad(cam_tilt)
        Ry = np.array([[np.cos(t), 0.0, np.sin(t)],
                       [0.0, 1.0, 0.0],
                       [-np.sin(t), 0.0, np.cos(t)]])
        z_c = Ry @ z_c
    up = np.array([0.0, 0.0, 1.0])                      # 工具轴, 用作"上"参考
    x_c = np.cross(z_c, up)
    x_c /= max(float(np.linalg.norm(x_c)), 1e-6)        # 视线与工具轴平行时退化, 有 y 偏移即可
    y_c = np.cross(z_c, x_c)
    R_cam = np.column_stack([x_c, y_c, z_c])
    x.leaf("camera", name=f"{name}_wrist_cam", pos=list(cam_pos),
           quat=quat_from_rot(R_cam), fovy=cam_fovy)
    x.close("body")


def actuator_block(arm: ArmMJCF, spec: ArmSpec, gripper: GripperSpec,
                   joint_kp: float = 2400.0, joint_kv: float = 60.0,
                   tau_scale: float = 1.0, mode: str = "position") -> str:
    """执行器组。

    ``mode="position"``: 关节位置伺服, ``ctrl`` = 目标角度（默认, 训练/采集用）。
    ``mode="torque"``  : 纯力矩电机, ``ctrl`` = 关节力矩（牛顿·米）, 供上游 TSID 力矩环用。
        夹爪始终保持位置伺服 —— 开合是二值指令, 没必要也不该进 QP。
    """
    x = _Xml()
    x.open("actuator")
    for i, jn in enumerate(arm.joint_names):
        fmax = float(spec.tau_max[i]) * tau_scale
        if mode == "torque":
            x.leaf("motor", name=f"act_{jn}", joint=jn, gear=1.0,
                   forcerange=[-fmax, fmax], ctrlrange=[-fmax, fmax])
            continue
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


def solo_xml(arm: ArmMJCF, spec: ArmSpec, gripper: GripperSpec | None = None,
             model_name: str = "rig", with_actuators: bool = True) -> str:
    """只含机械臂的完整 MJCF（测试 / 单臂调试用）。

    单独抽出来是因为"带网格的臂"必须把 ``<compiler meshdir>`` 与 ``<asset>``
    一起带上; 手工拼 ``<worldbody>{arm.xml}</worldbody>`` 在换成 mesh 视觉后
    会以 ``mesh not found`` 报错 —— 让它只有一种正确写法。

    ``with_actuators=False`` 给"纯被动臂"的动力学对照用（没有工具时也不该有指爪执行器）。
    """
    x = _Xml()
    x.raw(f'<mujoco model="{model_name}">')
    x.leaf("compiler", angle="radian", autolimits="true", meshdir=(arm.mesh_dir or None))
    x.leaf("option", timestep=0.002, integrator="implicitfast")
    if arm.mesh_assets:
        x.open("asset")
        x.raw(arm.mesh_assets)
        x.close("asset")
    x.open("worldbody")
    x.raw(arm.xml)
    x.close("worldbody")
    if with_actuators:
        x.raw(actuator_block(arm, spec, gripper or GripperSpec()))
    x.raw("</mujoco>")
    return x.text()
