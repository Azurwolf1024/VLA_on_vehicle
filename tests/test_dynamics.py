"""上游解析动力学 vs MuJoCo 刚体模型: 它们是不是同一个被控对象。

这一条是"能不能把上游 NERO 模型直接当 MuJoCo 控制器"的全部依据。若 M/H/J 都对不上,
那么 TSID 算出来的力矩与 MuJoCo 里的实际对象无关, 力矩环只能靠反馈硬撑,
论文那套"基座运动前馈"的结论在仿真里也就无从检验。

做法是**纯臂、无夹爪、基座固定**的最小对照: 把两边的差异压到只剩
"上游没有 armature 字段"这一项, 而这一项是可以用 M_mj - A 精确扣掉的。
"""

from __future__ import annotations

import unittest

import numpy as np

from vla_vehicle.robot.arm import load_arm   # 同时负责 ensure_upstream()
from vla_vehicle.robot.mjcf import build_arm, solo_xml

import mujoco


class TestAnalyticModelMatchesMuJoCo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robot, cls.spec, _ = load_arm()
        cls.n = cls.spec.n
        cls.arm = build_arm(cls.spec, q_ref=np.zeros(cls.n), name="nero",
                            with_tool=False)
        cls.model = mujoco.MjModel.from_xml_string(
            solo_xml(cls.arm, cls.spec, with_actuators=False))
        cls.data = mujoco.MjData(cls.model)
        cls.dof = np.array([
            cls.model.jnt_dofadr[mujoco.mj_name2id(
                cls.model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
            for jn in cls.arm.joint_names])

    # ------------------------------------------------------------------ #
    def _mj(self, q, dq=None):
        d, m = self.data, self.model
        d.qpos[:] = q
        d.qvel[:] = 0.0 if dq is None else dq
        mujoco.mj_forward(m, d)
        return d

    def test_mass_matrix(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(5):
            q = rng.uniform(np.maximum(self.spec.q_min, -1.2),
                            np.minimum(self.spec.q_max, 1.2))
            d = self._mj(q)
            M = np.zeros((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, d, M)
            M_mj = M[np.ix_(self.dof, self.dof)]
            M_up = self.robot.mass_matrix(q)
            # 上游没有转子惯量; MuJoCo 的 armature 是 0.02, 扣掉后应逐项相等
            err = np.abs(M_up - (M_mj - 0.02 * np.eye(self.n))).max()
            self.assertLess(err / np.abs(M_up).max(), 1e-4,
                            f"质量矩阵不匹配: rel {err / np.abs(M_up).max():.2e}")

    def test_bias_force(self) -> None:
        rng = np.random.default_rng(1)
        base = (np.eye(3), np.zeros(3))
        for _ in range(5):
            q = rng.uniform(np.maximum(self.spec.q_min, -1.2),
                            np.minimum(self.spec.q_max, 1.2))
            dq = rng.uniform(-1.0, 1.0, self.n)
            d = self._mj(q, dq)
            H_mj = d.qfrc_bias[self.dof]
            H_up = self.robot.bias(base, q, dq, np.zeros(12))
            err = np.abs(H_up - H_mj).max() / max(1.0, np.abs(H_up).max())
            self.assertLess(err, 1e-4, f"偏差力不匹配: rel {err:.2e}")

    def test_jacobian(self) -> None:
        rng = np.random.default_rng(2)
        base = (np.eye(3), np.zeros(3))
        for _ in range(5):
            q = rng.uniform(np.maximum(self.spec.q_min, -1.2),
                            np.minimum(self.spec.q_max, 1.2))
            d = self._mj(q)
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "nero_link7")
            self.assertGreater(bid, 0)
            jp = np.zeros((3, self.model.nv))
            jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, d, jp, jr, bid)
            J_mj = np.vstack([jr, jp])[:, self.dof]
            J_up = self.robot.jacobian(base, q)
            self.assertLess(np.abs(J_up - J_mj).max(), 1e-5)

    def test_gravity_compensation_holds_the_arm(self) -> None:
        """torque = H 时臂应当站住 —— 这条同时守住"船不是自由体"。"""
        from vla_vehicle.sim.env import EnvConfig, PickPlaceEnv
        from vla_vehicle.robot.arm import READY_Q

        env = PickPlaceEnv(EnvConfig(control_mode="tsid", vehicle_enabled=False,
                                     render=False), seed=0)
        d, m = env.data, env.model
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0
        env._write_vessel(0.0)
        d.qpos[env.arm_qadr] = READY_Q
        d.qpos[env.finger_qadr] = env.gripper.finger_open
        d.ctrl[env.finger_act] = env.gripper.finger_open
        mujoco.mj_forward(m, d)

        H = env.robot.bias((np.eye(3), np.zeros(3)), READY_Q, np.zeros(env.robot.n),
                           np.zeros(12))
        for _ in range(250):        # 0.5 s
            env._write_vessel(0.0)
            d.ctrl[env.arm_act] = H
            mujoco.mj_step(m, d)
        drift = float(np.abs(d.qpos[env.arm_qadr] - READY_Q).max())
        env.close()
        # 0.05 rad 是关节阻尼/干摩擦吃掉的那一点点, 不是"基座在自由下落"
        self.assertLess(drift, 0.08,
                        f"重力前馈没站住 (漂了 {drift:.3f} rad): 基座可能不是运动学平台")
