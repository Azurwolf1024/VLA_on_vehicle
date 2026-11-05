"""MuJoCo 场景装配: 低干舷作业艇 + 舷侧作业口 + 内置垃圾舱 + NERO + 双相机 + 漂在水面的垃圾。

坐标与坐标系
------------------------------------------------------------------
``vessel`` 体 (mocap, 运动学平台) 的原点定义在**甲板面、机械臂安装点正下方**,
+ x 指向船头, + z 向上。于是:

* 机械臂的 q 全部是"相对甲板"的关节角 —— 与真机一致;
* 垃圾舱开在甲板上 (0.32, 0.06), 深 0.10 —— 这就是"丢进载具**内置**垃圾桶";
* 载具 6-DOF 运动由上游 ``veh_arm.platform`` 的引擎逐帧写进 mocap_pos/mocap_quat,
  质量视为无限大, 机械臂的反作用推不动它。

**垃圾在水面上、船舷之外**, 是独立的 world 自由体:

* 位置写在**世界水平面**上, 所以船一横摇, 它们在甲板系里就会漂移 ——
  这正是本任务区别于"在甲板上捡"的地方;
* 水面本身是纯视觉平面 (contype=0), 另有一块 ``contype=16`` 的支撑面只与垃圾碰撞,
  让它们"浮"在水面而不穿下去, 同时不会给船体施加冲量。

干舷只有 0.12 m: 臂要伸到船舷外 0.45 m 处再向下 0.12 m 才够得着水面。
−y 舷墙在作业口处断开 0.44 m, 臂从这里伸出去。

双相机
------------------------------------------------------------------
``bow_cam``    船头桅杆位, 视野同时覆盖"舷外水面上的垃圾 + 甲板 + 垃圾舱" —— 全局语义。
``arm_wrist_cam`` 装在掌部**侧前方** (沿 x, 与指爪开合方向正交) 并自动瞄准 TCP ——
局部毫米级操作语义。两支都是刚体相机, 会随船摇动, 这正是我们要的数据分布。
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
    # 甲板面到水面的高度。**低干舷**是这条任务成立的前提: 垃圾漂在水面上,
    # 机械臂要伸到船舷外 0.45 m 处再向下 0.12 m 才够得着。
    deck_height: float = 0.12
    hull_half: tuple = (0.60, 0.38, 0.045)   # 前两维用于开孔板, z 维当前未用

    # 垃圾舱放在**作业侧同一侧**: 打捞口外就是水面, 捞起来转身就近投放。
    # 早期放在对角 (0.26,0.26), 机械臂要横跨 0.72 m 甲板 ->
    # 回合 97 步、船头相机必须同时包住对角两侧, 垃圾在 160² 画面里只有 6 px。
    # 移到 (0.32,0.06) 后 85 步、8 px, 通过率不变 (4/4)。
    bin_center: tuple = (0.32, 0.06)   # 甲板上的内置垃圾舱（甲板系 xy）
    bin_inner_half: tuple = (0.075, 0.065)
    bin_wall: float = 0.012
    bin_depth: float = 0.100           # 必须 < deck_height, 否则舱底退化

    # 侧舷**作业口**: 舷墙在这里断开, 机械臂才能伸出去夹水面上的垃圾。
    work_side: float = -1.0            # -1 = -y 侧舷
    work_opening_half: float = 0.22    # 作业口半宽（沿 x）
    bulwark_height: float = 0.050

    # 船头全局相机: 任务改成"舷外水面取物"后, 一帧里必须同时看到
    # 水面上的垃圾(在 -y 侧, 低于甲板 0.12 m)、甲板、以及甲板上的垃圾舱。
    # 视野因此要往前上方挪、并加宽。
    # 船头全局相机: 一帧里要同时看到"舷外水面上的垃圾"和"甲板上的垃圾舱"。
    # 两者都在 -y 侧之后, 视野可以从 56° 收到 55° 并压低到 0.62 m,
    # 垃圾在画面里的直径因此从 6 px 涨到 8 px (160² 输入)。
    bow_cam_pos: tuple = (0.55, 0.02, 0.62)
    bow_cam_target: tuple = (0.10, -0.24, -0.08)
    bow_cam_fovy: float = 55.0
    wrist_cam_fovy: float = 80.0
    # 工具系: 掌侧外挂, 朝工具轴内倾。倾角留 None = 自动瞄准 TCP。
    # 横向要拉开到 7.5 cm 以上, 否则指垫会糊掉半个画面 (实测 4.2 cm 时指垫占 40%)。
    # 工具系: 掌部侧前方 8 cm 的支架上, 自动瞄准 TCP。
    # 侧移必须**沿 x**(与指爪开合方向正交): 沿 y 移会正对指垫背面,
    # 画面中间是一根 73 mm 长的黄色柱子。
    wrist_cam_pos: tuple = (0.080, 0.0, 0.040)
    wrist_cam_tilt: float | None = None
    wrist_cam_ahead: float = 0.0

    actuator_mode: str = "position"    # "position"=关节位置伺服 | "torque"=TSID 力矩环
    # 机械臂视觉: "mesh"=上游真实 STL | "capsule"=骨架胶囊(快) | "auto"=有网格就用网格。
    # 只影响渲染, 不影响物理。
    visual: str = "auto"
    img_width: int = 160
    img_height: int = 160

    dt_physics: float = 0.002
    gravity: float = 9.81

    # 垃圾撒布区（甲板系极坐标, 在**船舷外**的水面/地面上）。
    # 角度范围以 work_side 那侧的法向为中心; 半径上限由可达性决定
    # (r≤0.55 且 z=−0.12 时逆解误差 <2 mm, 见 docs/sim.md 的可达域表)。
    spawn_r: tuple = (0.435, 0.550)
    spawn_angle_deg: tuple = (-112.0, -68.0)
    spawn_hull_clearance: float = 0.035  # 离船体外缘的最小距离
    spawn_min_sep: float = 0.075
    spawn_settle: float = 0.6          # 回合开始前的静置时间 (s)

    # 水面/地面。水的可见平面不参与碰撞; 另有一块**只与垃圾碰撞**的支撑面
    # (contype=16), 让垃圾浮在水面上而不是穿下去。
    surface_kind: str = "water"        # "water" | "ground"（只影响外观）

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

    @property
    def surface_z(self) -> float:
        """水面/地面在**甲板系**下的高度。"""
        return -self.deck_height


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
    # conaffinity=17 = 1|16: 与甲板/舷墙/夹爪(位 1)碰撞, 也与水面支撑面(位 16)碰撞
    x.leaf("geom", name=f"{item.name}_g", type=s.geom, size=size, mass=float(s.mass),
           rgba=list(item.color.rgba), friction="1.4 0.03 0.001",
           condim=4, solref="0.004 1", solimp="0.9 0.95 0.001",
           conaffinity=17)
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
                    wrist_cam_pos=cfg.wrist_cam_pos, wrist_cam_tilt=cfg.wrist_cam_tilt,
                    wrist_cam_ahead=cfg.wrist_cam_ahead, visual=cfg.visual)
    q = look_at_quat(cfg.bow_cam_pos, cfg.bow_cam_target)

    x = _Xml()
    x.raw('<?xml version="1.0" encoding="utf-8"?>')
    x.open("mujoco", model="vla_vehicle_pickplace")
    # meshdir 必须是**绝对路径**: 场景是用 from_xml_string 加载的, 没有"XML 所在目录"
    # 可以当基准。上游 STL 不复制进本仓库, 所以路径直接指向上游资产目录。
    x.leaf("compiler", angle="radian", autolimits="true", inertiafromgeom="auto",
           balanceinertia="true", meshdir=(arm.mesh_dir or None))
    x.leaf("option", timestep=cfg.dt_physics, integrator="implicitfast", cone="elliptic",
           impratio=10.0, gravity=[0, 0, -cfg.gravity], iterations=60)
    x.open("visual")
    x.leaf("global", offwidth=cfg.img_width * 2, offheight=cfg.img_height * 2, fovy=45)
    x.leaf("quality", shadowsize=2048)
    x.close("visual")
    if arm.mesh_assets:
        x.open("asset")
        x.raw(arm.mesh_assets)
        x.close("asset")

    x.open("worldbody")
    x.leaf("light", name="sun", pos=[1.2, -1.0, 2.4], dir=[-0.35, 0.3, -1.0],
           directional="true", diffuse=[0.6, 0.6, 0.62], specular=[0.08, 0.08, 0.08])
    x.leaf("light", name="fill", pos=[-1.0, 1.2, 1.6], dir=[0.4, -0.3, -1.0],
           directional="true", diffuse=[0.35, 0.35, 0.38])
    # 水面/地面: **纯视觉**的可见平面 (contype=0)。
    water = cfg.surface_kind != "ground"
    x.leaf("geom", name="surface_vis", type="plane", pos=[0, 0, 0], size=[12, 12, 0.02],
           rgba=("0.13 0.28 0.38 1" if water else "0.30 0.30 0.29 1"),
           contype=0, conaffinity=0)
    # 支撑面: 垃圾要"浮"在水面上, 而不是穿下去。用一个**只有垃圾看得见**的平面:
    # 位掩码 contype=16, 垃圾的 conaffinity 含 16 —— 船体/机械臂/甲板都不与它碰撞,
    # 免得又重演"平面把船顶得乱抖"那条老路 (见 docs/sim.md §5.1)。
    x.leaf("geom", name="surface_support", type="plane", pos=[0, 0, 0.001],
           size=[12, 12, 0.02], rgba="0 0 0 0",
           contype=16, conaffinity=0, friction="0.9 0.02 0.001", condim=4)

    # 船体是 **mocap (运动学) 体**: 位姿由外部给定、质量视为无限大。
    # 用 freejoint 的话, 船每一步都真的在重力下自由下落 (实测 0.1 s 掉 4.9 cm),
    # 而我们只是每步把 qpos 覆写回去 —— 于是机械臂的基座一直在"下坠",
    # 任何以"基座运动被给定"为前提的控制器 (上游 TSID) 都会对不上。
    # 顺带: 机械臂 4 kg 的反作用力矩也不会再把 120 kg 的船推歪。
    # 代价: MuJoCo 里 mocap 体的速度恒为零, 臂的动力学感受不到基座速度耦合项
    # (上游模型里有 J_B·V_B), 这一项在 0.1 m/s 量级的海况下对应毫米级误差。
    x.open("body", name=cfg.vessel_body, pos=[0, 0, cfg.deck_height], mocap="true")
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
    # 舷墙（挡住垃圾不直接掉出去, 也是"甲板边界"的视觉线索）
    bh = cfg.bulwark_height / 2
    for nm, px, py, sx, sy in (
            ("bulwark_aft", -cfg.deck_half[0] + 0.01, 0.0, 0.01, cfg.deck_half[1]),
            ("bulwark_fwd", cfg.deck_half[0] - 0.01, 0.0, 0.01, cfg.deck_half[1]),
            ("bulwark_port", 0.0, cfg.deck_half[1] - 0.01, cfg.deck_half[0], 0.01)):
        x.leaf("geom", name=nm, type="box", pos=[px, py, bh],
               size=[sx, sy, bh], rgba="0.55 0.57 0.60 1")
    # 作业侧舷墙**断开**一段: 机械臂要从这个口子伸出去夹水面上的垃圾。
    wy = cfg.work_side * (cfg.deck_half[1] - 0.01)
    seg = (cfg.deck_half[0] - cfg.work_opening_half) / 2
    for nm, px in (("bulwark_work_a", -(cfg.work_opening_half + seg)),
                   ("bulwark_work_b", (cfg.work_opening_half + seg))):
        x.leaf("geom", name=nm, type="box", pos=[px, wy, bh],
               size=[seg, 0.01, bh], rgba="0.55 0.57 0.60 1")
    # 作业口两侧的护舷立柱, 给"窗口"一个视觉边界
    for nm, px in (("work_post_a", -(cfg.work_opening_half + 0.012)),
                   ("work_post_b", (cfg.work_opening_half + 0.012))):
        x.leaf("geom", name=nm, type="box", pos=[px, wy, bh + 0.015],
               size=[0.012, 0.014, bh + 0.015], rgba="0.85 0.62 0.20 1")
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

    x.raw(actuator_block(arm, spec, gripper, joint_kp=joint_kp, joint_kv=joint_kv,
                         mode=cfg.actuator_mode))

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
