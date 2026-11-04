"""把上游 ``veh_arm`` 的解析模型与 TSID 力矩环接进 MuJoCo。

为什么这一步值得做
------------------------------------------------------------------
上游控制器不"导入"任何 MuJoCo 的东西, 它自己用 ``Robot`` 算刚体动力学::

    M(q) q̈ + H(q, q̇, R_WB) = τ,     H = C q̇ + g(R_WB) + τ_base

``τ_base`` 是基座运动耦合项 —— 论文的核心贡献。既然 MuJoCo 里那台臂的
质量矩阵/偏差力/雅可比与这份解析模型逐项吻合 (``tests/test_dynamics.py``:
M 相对误差 4.5e-6、H 1.8e-6、J 1.4e-7), 就说明 **同一份 NERO 模型可以直接当
MuJoCo 的控制器用**: MuJoCo 负责接触与摩擦, 上游负责算力矩。

唯一必须补的差异: 转子惯量
------------------------------------------------------------------
MuJoCo 的关节带 ``armature``(减速器折算到关节的转子惯量), 解析模型的 ``Link``
里没有这个字段, 于是 ``M_plant = M_up + A``。这不是小量: NERO 腕部三关节的连杆
惯量只有 1e-3 量级, 而 A = 2e-2 —— 差了 20 倍。

**必须在 QP 内部补, 不能在求完之后补。** 先算 τ 再补 ``A·q̈`` 的话, QP 里的力矩
约束用的还是不含 A 的 M, 会批准一个实际超出 ``tau_max`` 的解 (实测: 关节 4 要求
47 N·m, 上限 40), 电机一饱和, 环路就进入极限环。这里用 :class:`_ArmatureRobot`
把 A 加到 ``state_terms`` 返回的 M 上, 让 QP / 加速度边界 / 力矩约束看到同一个
被控对象。

第二处差异: TCP 与法兰
------------------------------------------------------------------
上游 ``Robot.ee_pose`` 给的是**法兰**, 而本项目的任务目标一直是 **TCP**
(法兰沿工具轴 +105 mm)。不换基等于让控制器去追一个偏出 10 cm 的虚点 ——
现象是"臂一直朝一个错误的方向爬, 而且误差永远不为零"。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..deps import ensure_upstream

ensure_upstream()

from veh_arm.control.tsid import TSIDController, TSIDGains, TSIDOptions  # noqa: E402
from veh_arm.robot.model import Robot  # noqa: E402

from ..robot.arm import TCP_OFFSET  # noqa: E402


class _ArmatureRobot:
    """给上游 ``Robot`` 套一层"转子惯量", 其余接口原样透传。

    TSID 只用到 ``n`` / ``tau_max`` / ``state_terms`` / ``accel_bounds``; 其余属性
    一并暴露, 免得将来上游换实现时莫名 AttributeError。
    """

    def __init__(self, robot: Robot, armature: float):
        self._robot = robot
        self._A = float(armature) * np.eye(robot.n)
        self.n = robot.n
        self.tau_max = robot.tau_max
        self.q_min = robot.q_min
        self.q_max = robot.q_max
        self.dq_max = robot.dq_max

    def state_terms(self, base_pose, q, dq, rows, fext_rows=None,
                    want_M: bool = True, want_J: bool = True) -> dict:
        t = self._robot.state_terms(base_pose, q, dq, rows, fext_rows=fext_rows,
                                    want_M=want_M, want_J=want_J)
        if want_M:
            # 只改质量矩阵; H(= C q̇ + g + τ_base) 与 q̈ 无关, 不能动
            t["M"] = np.asarray(t["M"], float) + self._A
        return t

    def accel_bounds(self, q, dq, dt):
        return self._robot.accel_bounds(q, dq, dt)


@dataclass
class TSIDConfig:
    """力矩环参数。

    ``Kp/Kd`` 默认值**不采用**上游论文 Table I 的 500/40: 那组增益是按论文那台臂
    与它的任务调的。在本环境里实测 500/40 会让 TCP 稳态落后目标 3 cm 以上 ——
    抓取需要 1 cm 级对准, 于是在下探阶段就压在物体顶上过不去 (seed 0 失败)。
    2000/80 下同一回合 88 步完成 (位置伺服 87 步), 且 QP 可行性 4694/4700。
    """

    Kp: float = 2000.0           # 任务空间位置刚度
    Kd: float = 80.0             # 任务空间阻尼
    Kp_ns: float = 50.0          # 零空间姿态回中刚度
    Kd_ns: float = 20.0
    lam: float = 1e-3
    pinv_eps: float = 1e-6
    compensate_base: bool = True  # 关掉 = 上游的 "ignore coupling" 消融基线
    use_nullspace: bool = True
    armature: float = 0.02        # 与 mjcf.build_arm 的 joint armature 保持一致
    tau_scale: float = 1.0
    torque_limits: bool = True
    accel_limits: bool = True
    tcp_offset: float = TCP_OFFSET


class TSIDAdapter:
    """``Robot`` + ``TSIDController`` 的薄封装: MuJoCo 状态进, 关节力矩出。"""

    def __init__(self, robot: Robot, dt: float, cfg: TSIDConfig | None = None,
                 q_ns: np.ndarray | None = None):
        self.robot = robot
        self.cfg = cfg or TSIDConfig()
        self.dt = float(dt)
        n = robot.n
        q_ns = np.asarray(q_ns if q_ns is not None else np.zeros(n), float)
        eye6, eyen = np.eye(6), np.eye(n)
        gains = TSIDGains(
            Kp=self.cfg.Kp * eye6, Kd=self.cfg.Kd * eye6,
            Kp_ns=self.cfg.Kp_ns * eyen, Kd_ns=self.cfg.Kd_ns * eyen,
            lam=self.cfg.lam, q_ns=q_ns,
        )
        opts = TSIDOptions(
            compensate_base=self.cfg.compensate_base,
            use_nullspace=self.cfg.use_nullspace,
            torque_limits=self.cfg.torque_limits,
            accel_limits=self.cfg.accel_limits,
            pinv_eps=self.cfg.pinv_eps,
        )
        self.model = _ArmatureRobot(robot, self.cfg.armature)
        self.ctrl = TSIDController(self.model, self.dt, gains, opts)
        self.tau_max = np.asarray(robot.tau_max, float) * self.cfg.tau_scale
        self.last: dict = {}

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """清掉上一回合的加速度记忆与统计。"""
        self.ctrl._prev_qdd[:] = 0.0
        for k in self.ctrl.stats:
            self.ctrl.stats[k] = 0
        self.last = {}

    def step(self, q: np.ndarray, dq: np.ndarray, base: dict,
             R_d: np.ndarray, p_d: np.ndarray,
             xd_dot: np.ndarray | None = None,
             xd_ddot: np.ndarray | None = None) -> np.ndarray:
        """一步力矩。

        Parameters
        ----------
        q, dq : 关节角/角速度（编码器反馈, 这里取自 MuJoCo）
        base  : 基座位姿与运动 ``{R, p, omega, v, alpha, a}``, 世界系
        R_d, p_d : **TCP** 的世界系目标（内部换算成法兰目标）
        xd_dot : 目标 twist ``[ω_d; v_d]``, 世界系; 目标随甲板动时必须给, 否则稳态滞后
        """
        est = {
            "R_WB": np.asarray(base["R"], float),
            "p_B": np.asarray(base["p"], float),
            "omega_w": np.asarray(base.get("omega", np.zeros(3)), float),
            "v_w": np.asarray(base.get("v", np.zeros(3)), float),
            "alpha_w": np.asarray(base.get("alpha", np.zeros(3)), float),
            "a_w": np.asarray(base.get("a", np.zeros(3)), float),
        }
        R_d = np.asarray(R_d, float)
        p_flange = np.asarray(p_d, float) - R_d @ np.array(
            [0.0, 0.0, self.cfg.tcp_offset])
        ref = {
            "R_d": R_d,
            "p_d": p_flange,
            "xd_dot": (np.zeros(6) if xd_dot is None
                       else np.asarray(xd_dot, float)),
            "xd_ddot": (np.zeros(6) if xd_ddot is None
                        else np.asarray(xd_ddot, float)),
        }
        out = self.ctrl.compute(np.asarray(q, float), np.asarray(dq, float), est, ref)
        tau = np.clip(out["tau"], -self.tau_max, self.tau_max)
        out["tau"] = tau
        self.last = out
        return tau

    # ------------------------------------------------------------------ #
    @property
    def stats(self) -> dict:
        return dict(self.ctrl.stats)
