"""阻尼最小二乘 (DLS) 逆解 —— 载具基座系内的笛卡尔目标 -> 关节角。

为什么用位置内环而不是 TSID 力矩环
------------------------------------------------------------------
VLA 输出的是**低频(5–10 Hz)笛卡尔增量**，本来就assume下面有一层"伺服 + 逆解"。
这里把这层做成明确的内环: 决策层给 TCP 目标, 内环解成关节目标, MuJoCo 的位置执行器
负责跟踪。等需要世界系定点(抵抗船摇)时, 再换成上游的 TSID 力矩环 ——
替换点就是 ``PickPlaceEnv._advance()`` 里的关节目标, 见 ``docs/roadmap.md`` v0.3。

DLS 细节
------------------------------------------------------------------
* 误差取 ``[log(R_d R^T); p_d - p]``（与上游 ``core.lie`` 同口径）;
* ``dq = J^T (J J^T + λ² I)^{-1} e``, 奇异处自动软化;
* 单步位移/关节增量限幅 + 关节限位内缩, 避免指令跳变;
* 零空间投影把姿态拉向 ``q_pref``（保持肘部构型, 避免解在等价支路上来回跳）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..deps import ensure_upstream

ensure_upstream()

from veh_arm.core.lie import log_so3  # noqa: E402
from veh_arm.robot.model import Robot  # noqa: E402

from ..robot.arm import TCP_OFFSET  # noqa: E402


@dataclass
class IKSolution:
    q: np.ndarray
    pos_err: float
    rot_err: float
    converged: bool
    iters: int


class DLSIK:
    def __init__(self, robot: Robot, lam: float = 0.05, alpha: float = 0.6,
                 max_joint_step: float = 0.10, max_iter: int = 12,
                 tol_pos: float = 2e-4, tol_rot: float = 5e-3,
                 q_pref: np.ndarray | None = None, null_gain: float = 0.02,
                 margin: float = 0.02, tcp_offset: float = TCP_OFFSET):
        self.robot = robot
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.max_joint_step = float(max_joint_step)
        self.max_iter = int(max_iter)
        self.tol_pos = float(tol_pos)
        self.tol_rot = float(tol_rot)
        self.q_pref = None if q_pref is None else np.asarray(q_pref, float).copy()
        self.null_gain = float(null_gain)
        self.margin = float(margin)
        self.tcp_offset = float(tcp_offset)
        self._I3 = np.eye(3)

    # ------------------------------------------------------------------ #
    def tcp_pose(self, q: np.ndarray):
        """基座系 TCP 位姿。TCP = 法兰沿工具 +z 伸出 ``tcp_offset``。"""
        R, p = self.robot.ee_pose(self._I3, np.zeros(3), q)
        return R, p + R @ np.array([0.0, 0.0, self.tcp_offset])

    # ------------------------------------------------------------------ #
    def solve(self, p_target: np.ndarray, R_target: np.ndarray,
              q0: np.ndarray, max_iter: int | None = None) -> IKSolution:
        robot = self.robot
        q = np.asarray(q0, float).copy()
        q_lo = robot.q_min + self.margin
        q_hi = robot.q_max - self.margin
        p_target = np.asarray(p_target, float)
        R_target = np.asarray(R_target, float)
        it = 0
        e_p = e_R = np.zeros(3)
        n_iter = self.max_iter if max_iter is None else int(max_iter)
        for it in range(1, n_iter + 1):
            R, p = self.tcp_pose(q)
            e_p = p_target - p
            e_R = log_so3(R_target @ R.T)
            if np.linalg.norm(e_p) < self.tol_pos and np.linalg.norm(e_R) < self.tol_rot:
                break
            J = robot.jacobian((self._I3, np.zeros(3)), q)      # (6, n) 基座系
            e = np.concatenate([e_R, e_p])
            JJt = J @ J.T + (self.lam ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            # 零空间: 拉回偏好位形（不改变末端任务）
            if self.q_pref is not None:
                Jpinv = J.T @ np.linalg.inv(JJt)
                N = np.eye(robot.n) - Jpinv @ J
                dq = dq + self.null_gain * (N @ (self.q_pref - q))
            m = float(np.max(np.abs(dq)))
            if m > self.max_joint_step:
                dq *= self.max_joint_step / m
            q = np.clip(q + self.alpha * dq, q_lo, q_hi)
        ok = (np.linalg.norm(e_p) < 5e-3) and (np.linalg.norm(e_R) < 5e-2)
        return IKSolution(q=q, pos_err=float(np.linalg.norm(e_p)),
                          rot_err=float(np.linalg.norm(e_R)), converged=ok, iters=it)

    # ------------------------------------------------------------------ #
    def reachable(self, p_target, R_target, q0, trials: int = 3) -> bool:
        """从若干个不同初值试解, 判断目标是否在工作空间内。"""
        cands = [np.asarray(q0, float)]
        if self.q_pref is not None:
            cands.append(self.q_pref)
        cands.append(0.5 * (self.robot.q_min + self.robot.q_max))
        best = None
        for c in cands[:trials]:
            sol = self.solve(p_target, R_target, c, max_iter=self.max_iter * 2)
            if best is None or sol.pos_err < best.pos_err:
                best = sol
            if sol.converged:
                return True
        return bool(best is not None and best.converged)
