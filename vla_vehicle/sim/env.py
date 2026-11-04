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
    max_decisions: int = 90
    workspace_lo: tuple = (-0.10, -0.34, 0.004)
    workspace_hi: tuple = (0.44, 0.30, 0.42)
    vehicle_kind: str = "sea"
    vehicle_scale: float = 0.35
    vehicle_time_scale: float = 1.0
    vehicle_enabled: bool = True
    lang: str = "zh"
    render: bool = True
    # 内环: "pos"=关节位置伺服 + DLS 逆解(默认) | "tsid"=上游解析模型 TSID 力矩环
    control_mode: str = "pos"
    # 任务参考系: "deck"=目标随甲板动(甲板上的物体就该这样) | "world"=目标钉在世界系
    frame: str = "deck"
    tsid: TSIDConfig = field(default_factory=TSIDConfig)
    grasp_mode: str = "attach"   # "attach"=运动学吸附(默认) | "contact"=纯接触摩擦
    grasp_close_below: float = 0.30   # 开口小于此比例即视为"夹住"
    grasp_open_above: float = 0.60    # 开口大于此比例即视为"松开"
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
                                      time_scale=float(self.cfg.vehicle_time_scale))
                        if self.cfg.vehicle_enabled else None)
        self.items: List[TrashItem] = []
        self.grasp_idx = -1
        self._attach = None
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
        bx, by = sc.bin_center
        lim_x, lim_y = sc.deck_half[0] - sc.spawn_margin, sc.deck_half[1] - sc.spawn_margin
        picked, tries = [], 0
        while len(picked) < self.cfg.n_objects:
            tries += 1
            if tries > 8000:
                raise RuntimeError("垃圾撒布采样失败: 请放宽 spawn 区域或减小最小间距")
            r = rng.uniform(*sc.spawn_r)
            th = np.deg2rad(rng.uniform(*sc.spawn_angle_deg))
            x, y = r * np.cos(th), r * np.sin(th)
            if abs(x) > lim_x or abs(y) > lim_y:
                continue
            if np.hypot(x - bx, y - by) < sc.spawn_bin_clearance:
                continue
            if any(np.hypot(x - p[0], y - p[1]) < sc.spawn_min_sep for p in picked):
                continue
            picked.append((x, y))
        items = []
        for i, (shape_key, color_key, (x, y)) in enumerate(zip(shape_keys, color_keys, picked)):
            shape = SHAPES[shape_key]
            # item.pos 是**甲板系**: 甲板面即 z=0, 世界高度由 vessel 位姿补上
            z = shape.half_height + 0.002
            # 初始朝向与夹爪开合方向(±y)对齐, 只留 ±10° 抖动。
            # 夹爪最大开口 84 mm, 若长方体随机偏航 45°, 投影宽度可达 (a+b)/√2 > 84 mm,
            # 爪子会压在物体顶上"抓不下去" —— 这属于能力边界, 不该由随机初始条件触发。
            yaw = float(rng.uniform(-0.17, 0.17)) if shape.geom == "box" else 0.0
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
        self._tcp_cmd = np.zeros(3)
        self._deck_target = np.zeros(3)
        self._deck_R = np.eye(3)
        self._world_target = np.zeros(3)
        self._world_R = np.eye(3)

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

    def _tcp_R_target(self, p_tcp: np.ndarray | None = None) -> np.ndarray:
        """工具目标姿态: 恒定"朝下 + 指爪沿 ±y 开合"。

        刻意不随目标位置改变偏航 —— 1) 决策层只输出位移, 姿态是内环职责;
        2) 偏航一旦随目标位置变, 腕部 J7 会在大范围运输时撞限位;
        3) 恒定姿态避免把"目标在哪"这类特权信息从控制层漏给策略。
        """
        return topdown_rotation(0.0)

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
        if self.grasp_idx >= 0 and grip > self.cfg.grasp_open_above:
            self.grasp_idx = -1
            self._attach = None
        elif self.grasp_idx < 0 and grip < self.cfg.grasp_close_below:
            cand = self._grasp_candidate()
            if cand >= 0:
                R1, p1 = self._body_pose(self.tool_bid)
                R2, p2 = self._body_pose(int(self.obj_bid[cand]))
                self.grasp_idx = cand
                self._attach = (R1.T @ (p2 - p1), R1.T @ R2)

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
        if self.cfg.frame == "world":
            # 世界系: **指令活在世界上**。增量按"当前甲板朝向"施加 (策略说的
            # "往前 2 cm"仍然是相对甲板), 但零动作时目标不随船漂 —— 这样机械臂
            # 才会真的去抵消船摇。边界仍按甲板系裁剪, 免得跑出可达域。
            R_now, p_now, _, _ = self._vessel_state(self.t)
            self._world_target = self._world_target + R_now @ dxyz
            deck_eq = R_now.T @ (self._world_target - p_now)
            deck_eq = np.clip(deck_eq, lo, hi)
            self._world_target = p_now + R_now @ deck_eq
            p_target = deck_eq
            self._deck_R = self._tcp_R_target(p_target)
        else:
            p_target = np.clip(self._tcp_cmd + dxyz, lo, hi)
        self._tcp_cmd = p_target.copy()
        self._deck_target = p_target.copy()
        if self.cfg.frame != "world":
            self._deck_R = self._tcp_R_target(p_target)

        if self.cfg.control_mode == "tsid":
            for _ in range(self.cfg.n_ctrl):
                self._advance(p_target, grip)
        else:
            sol = self.ik.solve(p_target, self._deck_R, self.q_arm, max_iter=8)
            q_start = self.q_arm
            for k in range(self.cfg.n_ctrl):
                s = (k + 1) / self.cfg.n_ctrl
                self._advance(q_start + s * (sol.q - q_start), grip)
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
