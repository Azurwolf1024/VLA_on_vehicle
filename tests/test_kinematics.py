"""校验 MuJoCo 里的机械臂与上游 ``veh_arm`` 模型**逐点一致**。

这是整个项目最重要的一个测试: 一旦仿真与控制器用的不是同一条运动链,
"控制没问题、仿真不听话"这类现象会花掉几天时间去查, 而且不会报错。
"""

from __future__ import annotations

import unittest

import numpy as np

from tests._util import load_rig


class TestKinematicsMatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        cls.robot, cls.spec, cls.arm, cls.model, cls.data = load_rig()
        cls.mj = mujoco

    def _fk_mj(self, q):
        m, d = self.model, self.data
        for i, jn in enumerate(self.arm.joint_names):
            d.qpos[m.jnt_qposadr[self.mj.mj_name2id(m, self.mj.mjtObj.mjOBJ_JOINT, jn)]] = q[i]
        self.mj.mj_forward(m, d)
        b7 = self.mj.mj_name2id(m, self.mj.mjtObj.mjOBJ_BODY, "nero_link7")
        assert b7 >= 0, "找不到 nero_link7: 名字对不上时 mj_name2id 会返回 -1, 而 -1 索引仍能取到某个 body, 极易误判"
        return d.xmat[b7].reshape(3, 3), d.xpos[b7]

    def test_fk_matches_over_random_configs(self):
        rng = np.random.default_rng(0)
        worst_p, worst_r = 0.0, 0.0
        for _ in range(20):
            q = rng.uniform(self.spec.q_min * 0.8, self.spec.q_max * 0.8)
            q = np.clip(q, self.spec.q_min + 1e-3, self.spec.q_max - 1e-3)
            R_mj, p_mj = self._fk_mj(q)
            R_up, p_up = self.robot.ee_pose(np.eye(3), np.zeros(3), q)
            worst_p = max(worst_p, float(np.linalg.norm(p_mj - p_up)))
            worst_r = max(worst_r, float(np.abs(R_mj - R_up).max()))
        self.assertLess(worst_p, 1e-5, f"末端位置不一致: {worst_p:.2e} m")
        self.assertLess(worst_r, 1e-5, f"末端姿态不一致: {worst_r:.2e}")

    def test_tcp_site_matches_model(self):
        from vla_vehicle.robot.arm import TCP_OFFSET
        q = np.array([1.1484, -0.5525, 1.5619, 1.9012, -0.9940, 0.2752, -0.6078])
        self._fk_mj(q)
        sid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_SITE, "nero_tcp")
        self.assertGreaterEqual(sid, 0, "找不到 nero_tcp site")
        R, p = self.robot.ee_pose(np.eye(3), np.zeros(3), q)
        expect = p + R @ np.array([0.0, 0.0, TCP_OFFSET])
        self.assertLess(float(np.linalg.norm(self.data.site_xpos[sid] - expect)), 1e-6)

    def test_urdf_inertia_patch(self):
        """上游 spec_from_urdf 会把 NERO 的惯量读成 0, 本仓库必须补回来。"""
        for i, L in enumerate(self.spec.links):
            w = np.linalg.eigvalsh(np.asarray(L.inertia, float))
            self.assertTrue(np.all(w > 0), f"连杆 {i + 1} 惯量非正定: {w}")
            self.assertGreater(w.max(), 1e-7)


if __name__ == "__main__":
    unittest.main()
