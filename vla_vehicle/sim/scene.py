"""MuJoCo 场景装配: 载具甲板 + 内置垃圾舱 + NERO + 双相机 + 垃圾对象。

坐标与坐标系
------------------------------------------------------------------
``vessel`` 体 (带 freejoint) 的原点定义在**甲板面、机械臂安装点正下方**,
+ x 指向船头, + z 向上。于是:

* 机械臂的 q 全部是"相对甲板"的关节角 —— 与真机一致;
* 垃圾/垃圾舱也焊在甲板系里 —— 这就是"丢进载具**内置**垃圾桶"的任务描述;
* 载具 6-DOF 运动由上游 ``veh_arm.platform`` 的引擎逐帧写进 ``vessel`` 的 qpos/qvel
  (运动学平台, 不受接触力反作用), 与船体质量无关。

垃圾对象**不挂在 vessel 下**, 而是独立的 world 自由体, 靠摩擦被甲板带着走。
这样"甲板在动"是真的在动, 而不是把所有东西焊死在船上 —— 也顺带给出滑动/翻倒这类
真实失效模式。

双相机
------------------------------------------------------------------
``bow_cam``    船头桅杆位, 视野覆盖"臂 + 前方垃圾 + 右舷垃圾舱" —— 全局语义。
``arm_wrist_cam`` 装在夹爪掌部两指之间, 沿工具轴向下 —— 局部毫米级操作语义。
两支都是 `vessel`/机械臂链上的刚体相机, 因此会随船摇动, 这正是我们要的数据分布。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ..robot.arm import GripperSpec, READY_Q
from ..robot.mjcf import _Xml, actuator_block, build_arm, look_at_quat
from .objects import TrashItem


@dataclass
class SceneConfig:
    """场景几何/物理参数（米 / 秒 / 千克）。"""

    deck_half: tuple = (0.48, 0.36)
    deck_thick: float = 0.030
    deck_height: float = 0.30          # 甲板面到水面的高度（vessel 初始 z）
    hull_half: tuple = (0.62, 0.38, 0.150)

    bin_center: tuple = (0.26, 0.26)   # 右舷前方内置垃圾舱（甲板系 xy）
    bin_inner_half: tuple = (0.075, 0.065)
    bin_wall: float = 0.012
    bin_depth: float = 0.120

    bow_cam_pos: tuple = (0.56, 0.02, 0.80)
    bow_cam_target: tuple = (0.20, 0.10, 0.02)
    bow_cam_fovy: float = 50.0
    wrist_cam_fovy: float = 80.0
    wrist_cam_pos: tuple = (0.0, -0.042, 0.008)   # 工具系: 掌侧外挂
    wrist_cam_tilt: float = 18.0                  # 朝工具轴内倾 (度)

    img_width: int = 160
    img_height: int = 160

    dt_physics: float = 0.002
    gravity: float = 9.81

    # 垃圾初始撒布区域（甲板系极坐标）: 工具朝下时可达域是**环形**的,
    # 半径太小肘关节折不下（见 docs/sim.md 的可达域表）, 所以内径不能小于 ~0.30 m。
    spawn_r: tuple = (0.310, 0.390)
    spawn_angle_deg: tuple = (-55.0, 55.0)
    spawn_bin_clearance: float = 0.170   # 离垃圾舱中心的最小距离
    spawn_margin: float = 0.060          # 离甲板边缘的最小距离
    spawn_min_sep: float = 0.085
    spawn_settle: float = 0.6          # 回合开始前的静置时间 (s)

    # 相机/物体命名
    vessel_body: str = "vessel"
    bow_cam: str = "bow_cam"

    @property
    def bin_hole_half(self) -> tuple:
        return (self.bin_inner_half[0] + self.bin_wall,
                self.bin_inner_half[1] + self.bin_wall)

    @property
    def bin_floor_z(self) -> float:
        return -self.bin_depth


@dataclass
class SceneBuild:
    xml: str
    arm_joints: List[str]
    finger_joints: List[str]
    object_names: List[str]
    tcp_site: str
    wrist_cam: str
    bow_cam: str
    vessel_body: str
    tool_body: str


# --------------------------------------------------------------------------- #
# 部件
# --------------------------------------------------------------------------- #
def _slabs(x_half, y_half, hole_x, hole_y, cx, cy):
    """把一块矩形板按中间的矩形孔切成 4 块 (返回 ((xa,xb),(ya,yb)) 列表)。"""
    x0, x1 = cx - hole_x, cx + hole_x
    y0, y1 = cy - hole_y, cy + hole_y
    return [
        ((-x_half, x0), (-y_half, y_half)),      # 左
        ((x1, x_half), (-y_half, y_half)),       # 右
        ((x0, x1), (y1, y_half)),                # 前
        ((x0, x1), (-y_half, y0)),               # 后
    ]


def _plate_with_hole(x: "_Xml", name: str, z0: float, z1: float,
                     half, hole_half, center, rgba: str, friction: str = "1.0 0.02 0.001",
                     material: str = "") -> None:
    """一块带矩形孔的板（甲板/船体共用）—— 保证垃圾舱是真正的通孔。"""
    for i, ((xa, xb), (ya, yb)) in enumerate(
            _slabs(half[0], half[1], hole_half[0], hole_half[1], center[0], center[1])):
        if xb - xa <= 1e-6 or yb - ya <= 1e-6:
            continue
        x.leaf("geom", name=f"{name}_{i}", type="box",
               pos=[(xa + xb) / 2, (ya + yb) / 2, (z0 + z1) / 2],
               size=[(xb - xa) / 2, (yb - ya) / 2, (z1 - z0) / 2],
               rgba=rgba, friction=friction, condim=4)


def _object_xml(cfg: SceneConfig, item: TrashItem, x: _Xml) -> None:
    s = item.shape
    if s.geom == "cylinder":
        size = [float(s.size[0]), float(s.size[1])]
    elif s.geom == "sphere":
        size = [float(s.size[0])]
    else:
        size = [float(v) for v in s.size]
    x.open("body", name=item.name, pos=[float(v) for v in item.pos])
    x.leaf("freejoint", name=f"{item.name}_free")
    x.leaf("geom", name=f"{item.name}_g", type=s.geom, size=size, mass=float(s.mass),
           rgba=list(item.color.rgba), friction="1.4 0.03 0.001",
           condim=4, solref="0.004 1", solimp="0.9 0.95 0.001")
    x.close("body")


def build_scene(cfg: SceneConfig, items: Sequence[TrashItem], spec,
                gripper: GripperSpec | None = None,
                q_ref: np.ndarray | None = None,
                joint_kp: float = 2400.0, joint_kv: float = 60.0,
                max_objects: int | None = None) -> SceneBuild:
    """装配完整场景 XML。"""
    gripper = gripper or GripperSpec()
    arm = build_arm(spec, gripper=gripper, q_ref=q_ref if q_ref is not None else READY_Q,
                    name="nero", wrist_cam_fovy=cfg.wrist_cam_fovy,
                    wrist_cam_pos=cfg.wrist_cam_pos, wrist_cam_tilt=cfg.wrist_cam_tilt)
    q = look_at_quat(cfg.bow_cam_pos, cfg.bow_cam_target)

    x = _Xml()
    x.raw('<?xml version="1.0" encoding="utf-8"?>')
    x.open("mujoco", model="vla_vehicle_pickplace")
    x.leaf("compiler", angle="radian", autolimits="true", inertiafromgeom="auto",
           balanceinertia="true")
    x.leaf("option", timestep=cfg.dt_physics, integrator="implicitfast", cone="elliptic",
           impratio=10.0, gravity=[0, 0, -cfg.gravity], iterations=60)
    x.open("visual")
    x.leaf("global", offwidth=cfg.img_width * 2, offheight=cfg.img_height * 2, fovy=45)
    x.leaf("quality", shadowsize=2048)
    x.close("visual")

    x.open("worldbody")
    x.leaf("light", name="sun", pos=[1.2, -1.0, 2.4], dir=[-0.35, 0.3, -1.0],
           directional="true", diffuse=[0.6, 0.6, 0.62], specular=[0.08, 0.08, 0.08])
    x.leaf("light", name="fill", pos=[-1.0, 1.2, 1.6], dir=[0.4, -0.3, -1.0],
           directional="true", diffuse=[0.35, 0.35, 0.38])
    # 水面: **纯视觉**, 必须关掉碰撞。船体吃水线以下本来就"穿"在水里,
    # 一旦让 plane 参与接触, 求解器每步都会给 vessel 一个向上的巨大速度冲量,
    # 甲板于是在水下"抖", 再通过摩擦把垃圾拖着走 (表现为回合一开始物体就漂几厘米)。
    x.leaf("geom", name="water", type="plane", pos=[0, 0, 0], size=[12, 12, 0.02],
           rgba="0.13 0.28 0.38 1", contype=0, conaffinity=0)

    x.open("body", name=cfg.vessel_body, pos=[0, 0, cfg.deck_height])
    x.leaf("freejoint", name="vessel_free")
    x.leaf("inertial", pos=[0, 0, -0.12], mass=120.0,
           diaginertia=[9.0, 26.0, 26.0])
    # 船体（吃水线刚好在水面）: 同样在垃圾舱位置开孔, 否则舱只有 3 cm 深,
    # 垃圾会停在船体顶面上, 表现为"丢进去了却浮在甲板高度"。
    _plate_with_hole(x, "hull", -cfg.deck_height, -cfg.deck_thick, cfg.hull_half,
                     cfg.bin_hole_half, cfg.bin_center, "0.26 0.30 0.36 1",
                     friction="0.6 0.02 0.001")
    hx, hy = cfg.bin_center
    hix2, hiy2 = cfg.bin_hole_half
    x.leaf("geom", name="hull_bin_bottom", type="box",
           pos=[hx, hy, (-cfg.deck_height - cfg.bin_depth) / 2],
           size=[hix2, hiy2, (cfg.deck_height - cfg.bin_depth) / 2],
           rgba="0.26 0.30 0.36 1")
    # 舷墙（挡住垃圾不直接掉海里, 也给出"甲板边界"的视觉线索）
    for nm, px, py, sx, sy in (
            ("bulwark_aft", -cfg.deck_half[0] + 0.01, 0.0, 0.01, cfg.deck_half[1]),
            ("bulwark_fwd", cfg.deck_half[0] - 0.01, 0.0, 0.01, cfg.deck_half[1]),
            ("bulwark_port", 0.0, cfg.deck_half[1] - 0.01, cfg.deck_half[0], 0.01),
            ("bulwark_stbd", 0.0, -cfg.deck_half[1] + 0.01, cfg.deck_half[0], 0.01)):
        x.leaf("geom", name=nm, type="box", pos=[px, py, 0.025],
               size=[sx, sy, 0.025], rgba="0.55 0.57 0.60 1")
    # 甲板: 先按孔切开, 舱底与舱壁随后补齐 —— 这样"掉进去"是真的掉进一个凹坑
    _plate_with_hole(x, "deck", -cfg.deck_thick, 0.0, cfg.deck_half,
                     cfg.bin_hole_half, cfg.bin_center, "0.60 0.47 0.32 1")
    bx, by = cfg.bin_center
    hix, hiy = cfg.bin_hole_half
    x.leaf("geom", name="bin_floor", type="box",
           pos=[bx, by, -cfg.bin_depth + 0.005], size=[hix, hiy, 0.005],
           rgba="0.05 0.06 0.07 1", friction="1.0 0.02 0.001", condim=4)
    wall = 0.006
    for nm, px, py, sx, sy in (
            ("bin_wall_xp", bx + hix - wall / 2, by, wall / 2, hiy),
            ("bin_wall_xn", bx - hix + wall / 2, by, wall / 2, hiy),
            ("bin_wall_yp", bx, by + hiy - wall / 2, hix, wall / 2),
            ("bin_wall_yn", bx, by - hiy + wall / 2, hix, wall / 2)):
        x.leaf("geom", name=nm, type="box", pos=[px, py, -cfg.bin_depth / 2],
               size=[sx, sy, cfg.bin_depth / 2], rgba="0.14 0.15 0.17 1")
    x.leaf("camera", name=cfg.bow_cam, pos=list(cfg.bow_cam_pos), quat=list(q),
           fovy=float(cfg.bow_cam_fovy))
    x.raw(arm.xml)
    x.close("body")

    for it in items:
        _object_xml(cfg, it, x)
    x.close("worldbody")

    x.raw(actuator_block(arm, spec, gripper, joint_kp=joint_kp, joint_kv=joint_kv))

    x.close("mujoco")

    return SceneBuild(
        xml=x.text(),
        arm_joints=arm.joint_names,
        finger_joints=arm.finger_joints,
        object_names=[it.name for it in items],
        tcp_site=arm.tcp_site,
        wrist_cam=arm.wrist_camera,
        bow_cam=cfg.bow_cam,
        vessel_body=cfg.vessel_body,
        tool_body=arm.tool_body,
    )
