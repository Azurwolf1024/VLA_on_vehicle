"""抓放环境: MuJoCo 物理 + 载具 6-DOF 运动驱动 + 双相机 + 决策/内环两级时序。

时序
------------------------------------------------------------------
::

    决策层 (VLA)      10 Hz   dt_decision = 0.1 s   输出 TCP 增量 + 夹爪指令
      └ 逆解           10 Hz   每决策一次 DLS, 得到关节目标 q_goal
          └ 内环     100 Hz   dt_ctrl = 0.01 s      关节目标线性插值 + 位置伺服
              └ 物理 500 Hz   dt_physics = 0.002 s   MuJoCo mj_step

决策层只看到 10 Hz 的图像与本体感觉, 内环负责平滑与跟踪 —— 与真机部署一致
(VLA 不会、也不该跑在 500 Hz 上)。

载具运动
------------------------------------------------------------------
上游 ``veh_arm.platform`` 给出解析的 6-DOF 位姿/速度, 每帧写入 vessel 的 qpos/qvel,
把 vessel 当**运动学平台**用: 接触力不会反过来推动船。上岸/上船都成立,
也是"把扰动当输入"而不是"把扰动当结果"的正确做法。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Tuple

import numpy as np

from ..deps import ensure_upstream

ensure_upstream()

import mujoco  # noqa: E402

from veh_arm.platform.vehicle import build_vehicle  # noqa: E402

from ..control.ik import DLSIK  # noqa: E402
from ..control.tsid import TSIDAdapter, TSIDConfig  # noqa: E402
from ..robot.arm import (READY_Q, GripperSpec, load_arm,  # noqa: E402
                         topdown_rotation)
from ..robot.mjcf import quat_from_rot  # noqa: E402
from .objects import (COLORS, SHAPES, Tokenizer, TrashItem,  # noqa: E402
                      make_instruction)
from .scene import SceneConfig, build_scene  # noqa: E402


# --------------------------------------------------------------------------- #
@dataclass
class EnvConfig:
    scene: SceneConfig = field(default_factory=SceneConfig)
    n_objects: int = 3
    dt_ctrl: float = 0.01
    dt_decision: float = 0.1
    action_scale: float = 0.02          # 每决策步的 TCP 位移上限 (m)
    resync_err: float = 0.10            # 指令与实测偏差超过此值时重新对齐
    # 舷外取物单程就要 0.72 m（2 cm/步 = 36 步），加上下探/抬升/投放，
    # 140 步偏紧：实测世界系有一条回合在第 139 步才刚开口，物体还没落进桶就超时。
    max_decisions: int = 170
    workspace_lo: tuple = (-0.30, -0.60, -0.16)
    workspace_hi: tuple = (0.45, 0.34, 0.42)
    vehicle_kind: str = "sea"
    vehicle_scale: float = 0.35
    vehicle_time_scale: float = 1.0
    vehicle_enabled: bool = True
    #: 透传给上游 ``build_vehicle`` 的额外参数（如 road 的 ``speed``/``cls``）。
    #: 不给就用上游 profile 的默认值 —— 这些是**上游**的物理档位, 本仓库不复制。
    vehicle_params: dict = field(default_factory=dict)
    lang: str = "zh"
    render: bool = True
    # 内环: "pos"=关节位置伺服 + DLS 逆解(默认) | "tsid"=上游解析模型 TSID 力矩环
    control_mode: str = "pos"
    # 任务参考系: "deck"=目标随甲板动(甲板上的物体就该这样) | "world"=目标钉在世界系
    frame: str = "deck"
    tsid: TSIDConfig = field(default_factory=TSIDConfig)
    grasp_mode: str = "attach"   # "attach"=运动学吸附(默认) | "contact"=纯接触摩擦
    # 目标（要抓的垃圾）自己怎么动 —— 与 frame 正交:
    #   frame 管"指令活在哪个系里", target_motion 管"物体在世界里怎么动"。
    #
    #   "pinned"  世界系静止。车载/岸上: 车停稳、地面不动, 静对静。
    #   "surface" 随水面升沉。**船载打捞的真实工况**: 垃圾和船漂在同一片波上,
    #             升沉几乎抵消（共模）, 真正难的是船的姿态摆动。
    #   "deck"    固连甲板。甲板上的货物/随船载物, 最严苛的一档。
    #
    # 默认保持 "pinned": 之前所有基准（30/30、抓取偏差、船摇抑制）都是在这个
    # 口径下测的, 悄悄改默认值等于让那些数字失去参照。
    target_motion: str = "pinned"
    #: 目标升沉 / 船升沉 的幅值比。1.0 = 目标完全跟着船升沉（同相位共模）。
    surface_follow: float = 1.0
    #: 目标升沉相对船升沉的相位滞后 (s)。RAO 的相位差 + 波在船长尺度上的传播时差,
    #: 真实值在 0.1~0.3 s 量级; 给 0 是最"宽容"的共模假设。
    surface_delay: float = 0.0
    grasp_close_below: float = 0.30   # **指令**小于此比例即视为"夹住"
    # 松开看的是**实测**开口, 不是指令: 刚下发张开时手指还没动, 此时就恢复碰撞
    # 会让物体和指垫重叠, 接触求解把它一步弹出 10 cm (实测 0.262→0.364)。
    # 0.80 对应开口 ≈78 mm, 比最宽的垃圾 (60 mm) 宽出 9 mm/侧。
    grasp_open_above: float = 0.80
    level_start: bool = True     # 回合从船体接近水平的时刻开始
    level_horizon: float = 40.0  # 在这个时间窗内挑起始相位
    rebuild_each_episode: bool = True

    @property
    def n_ctrl(self) -> int:
        return max(1, int(round(self.dt_decision / self.dt_ctrl)))

    @property
    def n_phys(self) -> int:
        return max(1, int(round(self.dt_ctrl / self.scene.dt_physics)))


@dataclass
class Obs:
    """策略可见的观测（不含任何特权量）。"""

    global_rgb: np.ndarray       # (H,W,3) uint8
    wrist_rgb: np.ndarray        # (H,W,3) uint8
    q: np.ndarray                # (7,) 关节角
    dq: np.ndarray               # (7,)
    tcp: np.ndarray              # (3,) 基座系 TCP
    grip: float                  # 0=闭合 1=全开（归一化开口）
    tokens: np.ndarray           # (L,) 指令 token id
    instruction: str
    t: float
    step: int

    def as_dict(self) -> dict:
        return dict(global_rgb=self.global_rgb, wrist_rgb=self.wrist_rgb, q=self.q,
                    dq=self.dq, tcp=self.tcp, grip=np.float32(self.grip),
                    tokens=self.tokens, instruction=self.instruction,
                    t=np.float32(self.t), step=np.int32(self.step))


# --------------------------------------------------------------------------- #
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_z(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


class PickPlaceEnv:
    """载具上的"捡垃圾 -> 丢进内置垃圾舱"环境。"""

    def __init__(self, cfg: EnvConfig | None = None, seed: int = 0):
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.robot, self.spec, self.meta = load_arm()
        self.gripper = GripperSpec()
        self.tok = Tokenizer()
        self.ik = DLSIK(self.robot, q_pref=READY_Q)
        self.vehicle = (build_vehicle(self.cfg.vehicle_kind,
                                      scale=float(self.cfg.vehicle_scale),
                                      time_scale=float(self.cfg.vehicle_time_scale),
                                      **dict(self.cfg.vehicle_params))
                        if self.cfg.vehicle_enabled else None)
        self.items: List[TrashItem] = []
        self.grasp_idx = -1
        self._attach = None
        self.grasp_err: float | None = None
        self.t0 = 0.0
        self.target_index = 0
        self.instruction = ""
        self.t = 0.0
        self.step_count = 0
        self.model = None
        self.data = None
        self.renderer = None
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # 构建 / 重置
    # ------------------------------------------------------------------ #
    def _sample_items(self, rng) -> Tuple[List[TrashItem], int]:
        # 形状、颜色都取互不相同的组合: 3 个物体在语言上永远可区分
        shapes = list(SHAPES.keys())
        colors = list(COLORS.keys())
        n = min(self.cfg.n_objects, len(shapes), len(colors))
        shape_keys = [shapes[i] for i in rng.permutation(len(shapes))[:n]]
        color_keys = [colors[i] for i in rng.permutation(len(colors))[:n]]
        sc = self.cfg.scene
        hull_y = sc.hull_half[1]
        picked, tries = [], 0
        while len(picked) < self.cfg.n_objects:
            tries += 1
            if tries > 8000:
                raise RuntimeError("垃圾撒布采样失败: 请放宽 spawn 区域或减小最小间距")
            r = rng.uniform(*sc.spawn_r)
            th = np.deg2rad(rng.uniform(*sc.spawn_angle_deg))
            x, y = r * np.cos(th), r * np.sin(th)
            # 必须在**船体之外**: 否则垃圾会叠在甲板/船体上, 任务就退回甲板系了。
            # work_side=-1 时要求 y ≤ -(hull_y + clearance), 即 work_side*y ≥ hull_y+clearance
            if sc.work_side * y < hull_y + sc.spawn_hull_clearance:
                continue
            if any(np.hypot(x - p[0], y - p[1]) < sc.spawn_min_sep for p in picked):
                continue
            picked.append((x, y))
        items = []
        for i, (shape_key, color_key, (x, y)) in enumerate(zip(shape_keys, color_keys, picked)):
            shape = SHAPES[shape_key]
            # item.pos 是**甲板系**; 垃圾躺在世界水平的水面/地面上, 对应甲板系 z = surface_z
            z = sc.surface_z + shape.half_height + 0.002
            # 长方体要让**窄边**正对指爪。指爪沿工具 y 轴开合, 而
            # y_t = (sin yaw, -cos yaw) (见 topdown_rotation); 物体局部
            # y = (-sinψ, cosψ)。令两者平行解得 ψ = yaw, 也就是物体偏航直接取
            # 目标的方位角 φ (工具偏航也是 φ, 见 _tcp_R_target)。
            # 曾经按 y_t=(cos,sin) 推成 90°-φ 与 φ-90°: 实测夹持瞬间
            # 物体 y 与工具 y 夹角 29°/86°, 100 mm 宽的纸箱被 84 mm 的爪子
            # 压进两侧各 8 mm, 下探卡死、抬起时被刮着走。
            phi = float(np.arctan2(y, x))
            yaw = float(phi % np.pi)
            if shape.geom == "box":
                yaw += float(rng.uniform(-0.17, 0.17))
            else:
                yaw = 0.0
            items.append(TrashItem(index=i, shape=shape, color=COLORS[color_key],
                                   pos=(x, y, z), yaw=yaw))
        return items, int(rng.integers(self.cfg.n_objects))

    def _build(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.t0 = self._pick_phase(rng)
        self.items, self.target_index = self._sample_items(rng)
        variant = int(rng.integers(0, 4))
        self.instruction = make_instruction(self.items[self.target_index],
                                            lang=self.cfg.lang, variant=variant)
        scene_cfg = self.cfg.scene
        if self.cfg.control_mode == "tsid":
            scene_cfg = replace(scene_cfg, actuator_mode="torque")
        self.scene = build_scene(scene_cfg, self.items, self.spec, self.gripper)
        self.model = mujoco.MjModel.from_xml_string(self.scene.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model

        def jid(name):
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if i < 0:
                raise KeyError(f"关节不存在: {name}")
            return i

        self.arm_qadr = np.array([m.jnt_qposadr[jid(j)] for j in self.scene.arm_joints])
        self.arm_vadr = np.array([m.jnt_dofadr[jid(j)] for j in self.scene.arm_joints])
        self.arm_act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                                   f"act_{j}") for j in self.scene.arm_joints])
        self.finger_qadr = np.array([m.jnt_qposadr[jid(j)] for j in self.scene.finger_joints])
        self.finger_act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                                      f"act_{j}") for j in self.scene.finger_joints])
        self.vessel_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                            self.scene.vessel_body)
        self.vessel_mocap = int(m.body_mocapid[self.vessel_bid])
        if self.vessel_mocap < 0:
            raise RuntimeError("vessel 不是 mocap 体: 见 scene.py 里关于"
                               "'船不能是自由体' 的说明")
        self.obj_qadr = np.array([m.jnt_qposadr[jid(f"{n}_free")]
                                  for n in self.scene.object_names])
        self.obj_vadr = np.array([m.jnt_dofadr[jid(f"{n}_free")]
                                  for n in self.scene.object_names])
        self.obj_bid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                                 for n in self.scene.object_names])
        self.obj_geoms = [np.where(m.geom_bodyid == b)[0] for b in self.obj_bid]
        self.tool_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.scene.tool_body)
        self.tcp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, self.scene.tcp_site)
        self.cam_bow = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.scene.bow_cam)
        self.cam_wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.scene.wrist_cam)

        # 力矩环: 上游解析模型 (M/H/J 与 MuJoCo 逐项吻合, 见 tests/test_dynamics.py)
        self.tsid = None
        if self.cfg.control_mode == "tsid":
            self.tsid = TSIDAdapter(self.robot, self.cfg.scene.dt_physics,
                                    self.cfg.tsid, q_ns=READY_Q)

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.cfg.render:
            self.renderer = mujoco.Renderer(self.model, self.cfg.scene.img_height,
                                            self.cfg.scene.img_width)
        self._vopt = None

    # ------------------------------------------------------------------ #
    def _pick_phase(self, rng) -> float:
        """在时间窗里挑一个"船体接近水平"的相位作为回合起点。

        只影响**初始条件**(垃圾摆在水平甲板上, 不会一开局就滑走), 不影响回合中的运动 ——
        难度仍然全部来自随后的摇晃。用 ``rpy`` 直接扫, 比 world_state 便宜。
        """
        if self.vehicle is None or not self.cfg.level_start:
            return 0.0
        ts = np.arange(0.0, self.cfg.level_horizon, 0.2)
        tilt = np.array([np.abs(self.vehicle.rpy(t)[:2]).sum() for t in ts])
        t_best = float(ts[int(np.argmin(tilt))])
        for step in (0.05, 0.01, 0.002):      # 局部细化
            cand = t_best + np.linspace(-10 * step, 10 * step, 21)
            cand = cand[cand >= 0.0]
            vals = [np.abs(self.vehicle.rpy(t)[:2]).sum() for t in cand]
            t_best = float(cand[int(np.argmin(vals))])
        return t_best

    def _vessel_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回甲板面参考点 (R, p, ω_world, v_world)。``t`` 为回合内时间。"""
        h = self.cfg.scene.deck_height
        if self.vehicle is None:
            return np.eye(3), np.array([0.0, 0.0, h]), np.zeros(3), np.zeros(3)
        st = self.vehicle.world_state(float(t) + self.t0)
        R, p = st["R_WB"], st["p_B"]
        r = R @ np.array([0.0, 0.0, h])
        omega = st["omega_w"]
        v = st["v_w"] + np.cross(omega, r)
        return R, p + r, omega, v

    def _deck_state(self, t: float) -> dict:
        """甲板安装点(即机械臂基座)的世界系位姿与一二阶运动, 供 TSID 前馈。"""
        R, p, omega, v = self._vessel_state(t)
        r = R @ np.array([0.0, 0.0, self.cfg.scene.deck_height])
        if self.vehicle is None:
            alpha = np.zeros(3)
            a_b = np.zeros(3)
        else:
            st = self.vehicle.world_state(float(t) + self.t0)
            alpha = np.asarray(st["alpha_w"], float)
            a_b = np.asarray(st["a_w"], float)
        a = a_b + np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))
        return dict(R=R, p=p, omega=omega, v=v, alpha=alpha, a=a)

    def _write_vessel(self, t: float, with_velocity: bool = True) -> None:
        """把甲板位姿写进 mocap。

        ``with_velocity`` 为兼容旧调用保留 —— mocap 体没有速度自由度。
        """
        R, p, omega, v = self._vessel_state(t)
        self.data.mocap_pos[self.vessel_mocap] = p
        self.data.mocap_quat[self.vessel_mocap] = quat_from_rot(R)

    # ------------------------------------------------------------------ #
    # 目标随载具运动（船载 vs 车载）
    # ------------------------------------------------------------------ #
    def _heave(self, t: float) -> Tuple[float, float]:
        """载具在时刻 ``t`` 的升沉 (m) 与升沉速度 (m/s)。

        波面高度用**船体升沉**当代理, 而不是另造一条正弦: 船和漂浮垃圾漂在同一片
        波上, 都由同一个水面驱动, 差别只在 RAO 的幅值与相位上。用同一个信号源,
        "目标怎么动"与"船怎么动"从一开始就同源 —— 否则两者的相位关系是编出来的,
        任何关于"升沉是共模的、能相互抵消"的结论都不可信。
        """
        if self.vehicle is None:
            return 0.0, 0.0
        st = self.vehicle.world_state(float(t) + self.t0)
        return float(st["p_B"][2]), float(st["v_w"][2])

    def _surface_motion(self) -> Tuple[float, float]:
        """目标相对锚点时刻的升沉量与升沉速度。"""
        delay = float(self.cfg.surface_delay)
        z1, v1 = self._heave(self.t - delay)
        z0, _ = self._heave(-delay)
        k = float(self.cfg.surface_follow)
        return k * (z1 - z0), k * v1

    def _capture_anchors(self) -> None:
        """静置结束后, 记下每个垃圾的实测位姿作为随动锚点。

        用实测而不是采样值: 静置 0.6 s 里物体在重力下落了几毫米、可能还蹭了一下。
        锚点偏一点, 后面整条轨迹就整体偏一点。
        """
        d = self.data
        R0, p0, _, _ = self._vessel_state(0.0)
        self._obj_anchor, self._obj_anchor_q, self._obj_deck = [], [], []
        for adr in self.obj_qadr:
            p_w = d.qpos[adr:adr + 3].copy()
            q_w = d.qpos[adr + 3:adr + 7].copy()
            R_w = _quat_to_rot(q_w)
            self._obj_anchor.append(p_w)
            self._obj_anchor_q.append(q_w)
            self._obj_deck.append((R0.T @ (p_w - p0), R0.T @ R_w))

    def _apply_target_motion(self) -> None:
        """按 ``target_motion`` 把未被夹持的物体运动学驱动一遍。

        为什么是**运动学驱动**而不是给它们加力: 漂浮物由水动力支承、地面上的轻物由
        支撑面支承, 它们的位置由水面/地面决定, 不是自由刚体积分出来的。每步覆写
        qpos 并给上解析速度, 等价于"无穷大的支承刚度" —— 与 vessel 用 mocap 同思路。

        速度必须一起给对: 只写 qpos 而把 qvel 留成 0, 接触求解器会认为物体与其他
        表面在相互滑动, 于是在"跟甲板动"和"躺在支撑面上"之间凭空造出摩擦冲量。
        """
        mode = self.cfg.target_motion
        if mode == "pinned" or not getattr(self, "_obj_anchor", None):
            return
        if mode not in ("surface", "deck"):
            raise ValueError(f"未知 target_motion={mode!r}; 可选 pinned/surface/deck")
        d = self.data
        if mode == "surface":
            dz, vz = self._surface_motion()
        else:
            R, p, omega, v = self._vessel_state(self.t)

        for i, (qa, va) in enumerate(zip(self.obj_qadr, self.obj_vadr)):
            if i == self.grasp_idx or i in self._obj_free:
                continue                         # 夹在手上的交给 _apply_attach
            qa, va = int(qa), int(va)
            if mode == "surface":
                p_exp = self._obj_anchor[i] + np.array([0.0, 0.0, dz])
                q_exp, v_lin, v_ang = self._obj_anchor_q[i], (0.0, 0.0, vz), np.zeros(3)
            else:
                pos_d, R_d = self._obj_deck[i]
                r = R @ pos_d
                R_w = R @ R_d
                p_exp = p + r
                q_exp = quat_from_rot(R_w)
                v_lin = v + np.cross(omega, r)
                v_ang = R_w.T @ omega
            # 兜底: 物体已经被**别的东西**带离了它该在的位置（`contact` 抓取模式下
            # 没有吸附语义, 指垫把物体推着走, grasp_idx 永远是 -1）。再按锚点写回去,
            # 等于和接触求解器拔河。判据取 50 mm —— 正常随动一步的偏差在 0.2 mm 量级。
            if float(np.linalg.norm(d.qpos[qa:qa + 3] - p_exp)) > 0.05:
                self._obj_free.add(i)
                continue
            d.qpos[qa:qa + 3] = p_exp
            d.qpos[qa + 3:qa + 7] = q_exp
            d.qvel[va:va + 3] = v_lin
            d.qvel[va + 3:va + 6] = v_ang

    def reset(self, seed: int | None = None) -> Obs:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._seed = int(seed)
        if self.model is None or self.cfg.rebuild_each_episode:
            self._build(int(getattr(self, "_seed", 0)))
        d = self.data
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0

        self.t = 0.0
        self.step_count = 0
        self.grasp_idx = -1
        self.grasp_err = None
        self._tcp_cmd = np.zeros(3)
        self._deck_target = np.zeros(3)
        self._deck_R = np.eye(3)
        self._world_target = np.zeros(3)
        self._world_R = np.eye(3)
        self._obj_anchor, self._obj_anchor_q, self._obj_deck = [], [], []
        self._obj_free = set()

        R0, p0, w0, v0 = self._vessel_state(0.0)
        self._write_vessel(0.0)
        d.qpos[self.arm_qadr] = READY_Q
        d.ctrl[self.arm_act] = READY_Q
        d.ctrl[self.finger_act] = self.gripper.finger_open      # 张开
        d.qpos[self.finger_qadr] = self.gripper.finger_open
        # 垃圾贴着甲板摆放（姿态与甲板一致, 而不是与世界水平）
        qv = quat_from_rot(R0)
        for it, adr in zip(self.items, self.obj_qadr):
            d.qpos[adr:adr + 3] = p0 + R0 @ np.asarray(it.pos, float)
            d.qpos[adr + 3:adr + 7] = _quat_mul(qv, _quat_z(it.yaw))
        mujoco.mj_forward(self.model, d)

        # 复位时的目标就是 READY 位姿本身: 力矩环靠它做重力保持,
        # 位置伺服靠它当"别动"的 ctrl。两种内环都从同一个参考出发。
        R_ready, p_ready = self.ik.tcp_pose(READY_Q)
        self._tcp_cmd = p_ready.copy()
        self._deck_target = p_ready.copy()
        self._deck_R = R_ready.copy()
        R_v, p_v, _, _ = self._vessel_state(0.0)
        self._world_target = p_v + R_v @ p_ready
        self._world_R = topdown_rotation(0.0)
        if self.tsid is not None:
            self.tsid.reset()
            self._write_vessel(0.0, with_velocity=False)

        # 静置: 让垃圾落到甲板上, 期间甲板固定在 t=0 位姿
        n_settle = int(round(self.cfg.scene.spawn_settle / self.cfg.scene.dt_physics))
        for _ in range(max(1, n_settle)):
            # 静置期间把甲板**连速度一起**钉死: 否则每步都会把 t=0 的船速灌给接触求解器,
            # 垃圾会被"拖着走"几厘米, 相当于回合一开始就滑了一跤。
            self._write_vessel(0.0, with_velocity=False)
            # 力矩模式下 ctrl=0 是**零力矩**, 臂会在重力下直接掉下来 (实测 120°),
            # 所以静置期间必须让力矩环真的跑起来。
            # 参考必须**固定**在 READY: 若每步把参考刷成实测位姿, 就成了"追着自己下垂",
            # 误差恒为零, 重力补偿永远不会真正起作用。
            if self.tsid is not None:
                d.ctrl[self.arm_act] = self._tsid_tau()
            mujoco.mj_step(self.model, d)
        self.t = 0.0
        # 静置结束后才定随动锚点: 此刻物体的位姿才是回合真正的起点。
        self._capture_anchors()
        R_tcp, p_tcp = self.tcp_pose()
        self._tcp_cmd = p_tcp.copy()
        self._deck_target = p_tcp.copy()
        self._deck_R = R_tcp.copy()
        return self._obs()

    # ------------------------------------------------------------------ #
    # 状态读取
    # ------------------------------------------------------------------ #
    @property
    def q_arm(self) -> np.ndarray:
        return self.data.qpos[self.arm_qadr].copy()

    @property
    def dq_arm(self) -> np.ndarray:
        return self.data.qvel[self.arm_vadr].copy()

    def tcp_pose(self):
        return self.ik.tcp_pose(self.q_arm)

    def _grip_norm(self) -> float:
        q = float(np.mean(self.data.qpos[self.finger_qadr]))
        width = 2.0 * (self.gripper.finger_gap0 + q)
        return float(np.clip((width - self.gripper.min_width)
                             / (self.gripper.max_width - self.gripper.min_width), 0.0, 1.0))

    def _vessel_pose(self):
        d = self.data
        return d.xmat[self.vessel_bid].reshape(3, 3).copy(), d.xpos[self.vessel_bid].copy()

    def object_states(self) -> List[dict]:
        """垃圾在**甲板系**下的位姿（专家/评测用的特权信息）。"""
        Rv, pv = self._vessel_pose()
        out = []
        for it, bid in zip(self.items, self.obj_bid):
            p_w = self.data.xpos[bid].copy()
            R_w = self.data.xmat[bid].reshape(3, 3).copy()
            out.append(dict(item=it, p=Rv.T @ (p_w - pv), R=Rv.T @ R_w, p_world=p_w))
        return out

    def _to_deck(self, p_world, R_world):
        """世界系 (位置, 姿态) -> 当前甲板系。逆解只在甲板系里做。"""
        R, p, _, _ = self._vessel_state(self.t)
        return R.T @ (np.asarray(p_world, float) - p), R.T @ np.asarray(R_world, float)

    def _tcp_R_target(self, p_tcp: np.ndarray | None = None) -> np.ndarray:
        """工具目标姿态: 朝下, **偏航跟随目标方位角**（甲板系）。

        为什么不能像早期那样恒定 yaw=0: 目标一旦移到侧舷 (r≈0.45),
        腕部 J7 就必须转过去, 恒定零偏航会把它顶到限位 —— 实测 r=0.45 时
        位置误差 2.9 cm、旋转误差 0.29 rad, 也就是"够不着"。
        改成 ``yaw = atan2(y, x)`` 后同一目标达到 0.5 mm, J6/J7 余量 >0.6 rad。

        这不是把特权信息漏给策略: 姿态由**指令目标点本身**决定, 策略只给位移,
        姿态仍是内环的职责 (与"恒定朝下"属于同一类假设)。
        """
        if p_tcp is None:
            return topdown_rotation(0.0)
        p = np.asarray(p_tcp, float)
        if float(np.hypot(p[0], p[1])) < 0.05:      # 太靠近基座轴, 偏航无意义
            return topdown_rotation(0.0)
        return topdown_rotation(float(np.arctan2(p[1], p[0])))

    # ------------------------------------------------------------------ #
    # 抓取 (运动学吸附)
    # ------------------------------------------------------------------ #
    def _grasp_candidate(self) -> int:
        """找出此刻落在指垫之间的物体: 横向 < 45 mm 且高度落在指垫范围内。"""
        p = self.tcp_pose()[1]
        best, best_d = -1, 1e9
        for i, st in enumerate(self.object_states()):
            d = st["p"] - p
            lateral = float(np.hypot(d[0], d[1]))
            along = float(d[2])                     # 沿工具轴(向下为负)
            if lateral < 0.045 and -0.035 < along < 0.075 and lateral < best_d:
                best, best_d = i, lateral
        return best

    def _update_grasp(self, grip: float) -> None:
        """闭合到位即吸附, 张开即释放。

        ``attach`` 用**运动学吸附**: 记录抓住瞬间"物体相对工具"的位姿, 之后每个物理子步
        直接把物体 qpos/qvel 写成工具的刚体变换结果。它没有约束力, 因此不存在质量比失稳,
        代价是"被夹住时物体不会滑动/不会被挤出去" —— 对抓放任务恰好是想要的语义,
        也让示范数据里"抓没抓住"这件事只由**接近位姿**决定, 不被接触退化的随机性污染。
        """
        if self.cfg.grasp_mode != "attach":
            return
        if self.grasp_idx >= 0 and self._grip_norm() > self.cfg.grasp_open_above:
            self._set_object_collision(self.grasp_idx, True)
            self.grasp_idx = -1
            self._attach = None
        elif self.grasp_idx < 0 and grip < self.cfg.grasp_close_below:
            cand = self._grasp_candidate()
            if cand >= 0:
                R1, p1 = self._body_pose(self.tool_bid)
                R2, p2 = self._body_pose(int(self.obj_bid[cand]))
                self.grasp_idx = cand
                self._attach = (R1.T @ (p2 - p1), R1.T @ R2)
                # 抓取瞬间 TCP 相对物体中心的**水平**偏差（世界系）。
                # 取水平是因为竖直方向本来就有几何量: 指尖落在"顶面以下 45 mm",
                # 对高罐子来说指尖比中心还高一点, 那不是误差。
                # 水平偏差才是"有没有对准", 也才是船摇会污染的量。
                R_v, p_v = self._vessel_pose()
                p_tcp_w = p_v + R_v @ self.tcp_pose()[1]
                d = np.asarray(p2) - p_tcp_w
                self.grasp_err = float(np.hypot(d[0], d[1]))
                # 吸附是把物体**刚性地钉在工具上**, 此时它和指垫的接触没有物理意义;
                # 不关掉的话, 60 mm 宽的物体夹在 24 mm 的闭合行程里会把手指顶开
                # (实测夹持中 grip 从 0.06 跳到 1.00, 画面上等于"张着手搬运")。
                self._set_object_collision(cand, False)
                # 被搬动过的物体**永久**退出"随水面/甲板运动"的集合。
                # 它已经不在原来的位置上了: 再按锚点写回去, 等于把刚丢进桶的垃圾
                # 瞬移回舷外水面 (实测 surface 档第一版就是这样, 回合跑满 170 步超时,
                # 而物体又端端正正躺在原来的水面上)。掉出来的就让它掉出来的样子。
                self._obj_free.add(int(cand))

    def _set_object_collision(self, idx: int, on: bool) -> None:
        """开关某个垃圾的碰撞位 (吸附期间关掉, 释放后恢复)。"""
        if idx < 0:
            return
        for g in self.obj_geoms[idx]:
            self.model.geom_contype[g] = 1 if on else 0
            self.model.geom_conaffinity[g] = 17 if on else 0

    def _body_pose(self, bid: int):
        d = self.data
        return d.xmat[bid].reshape(3, 3).copy(), d.xpos[bid].copy()

    def _apply_attach(self) -> None:
        """把被吸附的物体钉在工具上（每个物理子步调用）。"""
        if self.grasp_idx < 0 or self._attach is None:
            return
        R1, p1 = self._body_pose(self.tool_bid)
        p_rel, R_rel = self._attach
        d = self.data
        adr = int(self.obj_qadr[self.grasp_idx])
        v_adr = int(self.obj_vadr[self.grasp_idx])
        d.qpos[adr:adr + 3] = p1 + R1 @ p_rel
        d.qpos[adr + 3:adr + 7] = quat_from_rot(R1 @ R_rel)
        d.qvel[v_adr:v_adr + 6] = 0.0

    # ------------------------------------------------------------------ #
    # 观测
    # ------------------------------------------------------------------ #
    def _render(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.renderer is None:
            h, w = self.cfg.scene.img_height, self.cfg.scene.img_width
            return np.zeros((h, w, 3), np.uint8), np.zeros((h, w, 3), np.uint8)
        self.renderer.update_scene(self.data, camera=self.cam_bow,
                                   scene_option=self._vopt)
        bow = self.renderer.render().copy()
        self.renderer.update_scene(self.data, camera=self.cam_wrist,
                                   scene_option=self._vopt)
        wrist = self.renderer.render().copy()
        return bow, wrist

    def _obs(self) -> Obs:
        bow, wrist = self._render()
        p = self.tcp_pose()[1]
        return Obs(global_rgb=bow, wrist_rgb=wrist, q=self.q_arm, dq=self.dq_arm,
                   tcp=p, grip=self._grip_norm(),
                   tokens=np.asarray(self.tok.encode(self.instruction), np.int64),
                   instruction=self.instruction, t=self.t, step=self.step_count)

    # ------------------------------------------------------------------ #
    # 单步
    # ------------------------------------------------------------------ #
    def _tsid_tau(self) -> np.ndarray:
        """当前时刻的关节力矩: 上游 TSID 力矩环 + 甲板运动前馈。"""
        st = self._deck_state(self.t)
        if self.cfg.frame == "world":
            p_d, R_d = self._world_target, self._world_R
            xd_dot = np.zeros(6)
        else:
            p_d = st["p"] + st["R"] @ self._deck_target
            R_d = st["R"] @ self._deck_R
            r = p_d - st["p"]
            # 目标随甲板动 -> 目标本身有速度; 不给前馈会留下稳态滞后
            xd_dot = np.concatenate([st["omega"],
                                     st["v"] + np.cross(st["omega"], r)])
        return self.tsid.step(self.q_arm, self.dq_arm, st, R_d, p_d, xd_dot=xd_dot)

    def _advance(self, q_cmd: np.ndarray, grip: float) -> None:
        d = self.data
        finger = float(self.gripper.ctrl_from_norm(grip))
        torque = self.cfg.control_mode == "tsid"
        for _ in range(self.cfg.n_phys):
            self.t += self.cfg.scene.dt_physics
            self._write_vessel(self.t)
            # 物体随即被"水面/甲板"带动 —— 必须在 vessel 之后, deck 档要用同一时刻的船位姿
            self._apply_target_motion()
            # 力矩环跑在物理步上(500 Hz), 与上游论文的 1 kHz 同量级;
            # 位置伺服则跑在 100 Hz 内环上, 中间线性插值。
            d.ctrl[self.arm_act] = self._tsid_tau() if torque else q_cmd
            d.ctrl[self.finger_act] = finger
            self._apply_attach()
            mujoco.mj_step(self.model, d)

    def step(self, action) -> Tuple[Obs, float, bool, dict]:
        action = np.asarray(action, float).reshape(-1)
        dxyz = np.clip(action[:3], -1.0, 1.0) * self.cfg.action_scale
        grip = float(np.clip(action[3], 0.0, 1.0)) if action.size > 3 else 1.0
        self._update_grasp(grip)

        # 目标点是**指令**的累加, 不是测量值的累加。
        # 若用 p_now + dxyz, 内环的任何漂移都会被下一步继承, 形成"棘轮":
        # 实测出现过动作全零、机械臂却自己往上爬 2 cm/2 s 的现象。
        if float(np.linalg.norm(self._tcp_cmd - self.tcp_pose()[1])) > self.cfg.resync_err:
            self._tcp_cmd = self.tcp_pose()[1].copy()      # 跟踪彻底掉队时重新对齐
        lo, hi = np.asarray(self.cfg.workspace_lo), np.asarray(self.cfg.workspace_hi)
        world = self.cfg.frame == "world"
        if world:
            # 世界系: **指令活在世界上**。增量按"当前甲板朝向"施加 (策略说的
            # "往前 2 cm"仍然是相对甲板), 但零动作时目标不随船漂 —— 这样机械臂
            # 才会真的去抵消船摇。边界仍按甲板系裁剪, 免得跑出可达域。
            R_now, p_now, _, _ = self._vessel_state(self.t)
            self._world_target = self._world_target + R_now @ dxyz
            deck_eq = R_now.T @ (self._world_target - p_now)
            deck_eq = np.clip(deck_eq, lo, hi)
            self._world_target = p_now + R_now @ deck_eq
            p_target = deck_eq
            # 姿态也要在世界系给定: 垃圾是**世界水平**的, 工具必须垂直于水面,
            # 而不是垂直于正在横摇的甲板。位置在甲板系里裁剪, 姿态在世界系里算。
            d_w = self._world_target - p_now
            yaw_w = 0.0 if float(np.hypot(d_w[0], d_w[1])) < 0.05 else float(
                np.arctan2(d_w[1], d_w[0]))
            self._world_R = topdown_rotation(yaw_w)
            self._deck_R = R_now.T @ self._world_R
        else:
            p_target = np.clip(self._tcp_cmd + dxyz, lo, hi)
            self._deck_R = self._tcp_R_target(p_target)
        self._tcp_cmd = p_target.copy()
        self._deck_target = p_target.copy()

        if self.cfg.control_mode == "tsid":
            for _ in range(self.cfg.n_ctrl):
                self._advance(p_target, grip)
        else:
            # 位置伺服: **每个内环步都重解一次**（热启动）。
            # 早期是"每决策解一次 + 关节线性插值", 但插值会引入半步滞后 ——
            # 在世界系下落后的就是几十毫米。更重要的是: 若只在某个参考系下重解,
            # "甲板系 vs 世界系"的对照就会把"重解频率"混进去, 比出来的东西不干净。
            for _ in range(self.cfg.n_ctrl):
                if world:
                    p_d, R_d = self._to_deck(self._world_target, self._world_R)
                else:
                    p_d, R_d = p_target, self._deck_R
                sol = self.ik.solve(p_d, R_d, self.q_arm, max_iter=3)
                self._advance(sol.q, grip)
        self.step_count += 1

        info = self.evaluate()
        done = self.step_count >= self.cfg.max_decisions or info["success"]
        return self._obs(), 0.0, bool(done), info

    # ------------------------------------------------------------------ #
    # 评测
    # ------------------------------------------------------------------ #
    def evaluate(self) -> dict:
        sc = self.cfg.scene
        states = self.object_states()
        bx, by = sc.bin_center
        hix, hiy = sc.bin_inner_half
        inside = []
        for s in states:
            p = s["p"]
            inside.append(bool(abs(p[0] - bx) < hix - 0.005 and abs(p[1] - by) < hiy - 0.005
                               and p[2] < -0.02))
        tgt = states[self.target_index]
        in_gripper = self._holding(tgt)
        success = bool(inside[self.target_index] and not in_gripper)
        return dict(success=success, target_in_bin=bool(inside[self.target_index]),
                    grasp_err=(None if self.grasp_err is None else float(self.grasp_err)),
                    inside=inside, target_index=self.target_index,
                    target_p=tgt["p"].copy(), holding=in_gripper,
                    step=self.step_count, t=self.t)

    def _holding(self, state: dict) -> bool:
        """目标是否还在夹爪里。"""
        if self.cfg.grasp_mode == "attach":
            return bool(self.grasp_idx >= 0
                        and self.items[self.grasp_idx].name == state["item"].name)
        R, p = self.tcp_pose()
        dp = state["p"] - p
        horiz = float(np.hypot(dp[0], dp[1]))
        return bool(horiz < 0.045 and -0.06 < dp[2] < 0.10 and self._grip_norm() < 0.45)

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
