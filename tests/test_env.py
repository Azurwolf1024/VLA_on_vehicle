"""环境与专家的端到端检查。

包含一个**慢测试**（跑完整条示范）, 它守的是"整条抓放链路是否还能走通";
把这条测试去掉的代价是: 环境里任何一处几何/时序改动都可能在几十回合之后才被发现。
"""

from __future__ import annotations

import unittest

import numpy as np

from tests._util import build_min_scene


class TestEnvSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = build_min_scene(vehicle_scale=0.2)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_reset_shapes(self):
        obs = self.env.reset(seed=1)
        self.assertEqual(obs.global_rgb.dtype, np.uint8)
        h, w = self.env.cfg.scene.img_height, self.env.cfg.scene.img_width
        self.assertEqual(obs.global_rgb.shape, (h, w, 3))
        self.assertEqual(obs.wrist_rgb.shape, (h, w, 3))
        self.assertEqual(obs.q.shape, (self.env.robot.n,))
        self.assertEqual(obs.tcp.shape, (3,))
        self.assertGreater(len(obs.tokens), 0)

    def test_objects_rest_on_surface(self):
        """垃圾躺在**水面/地面**上, 而不是甲板上。"""
        self.env.reset(seed=2)
        sc = self.env.cfg.scene
        for s in self.env.object_states():
            z = float(s["p"][2])
            half = s["item"].shape.half_height
            self.assertAlmostEqual(z, sc.surface_z + half, delta=0.012,
                                   msg=f"{s['item'].name} 没落在水面/地面上 (z={z:.3f})")

    def test_objects_are_outboard(self):
        """垃圾必须在**船体之外** —— 否则任务又退回"在甲板上捡"了。"""
        self.env.reset(seed=2)
        sc = self.env.cfg.scene
        for seed in range(4):
            self.env.reset(seed=seed)
            for s in self.env.object_states():
                y = sc.work_side * float(s["p"][1])
                self.assertGreater(y, sc.hull_half[1],
                                   f"seed={seed} {s['item'].name} 在船体范围内 (y={s['p'][1]:.3f})")

    def test_tool_yaw_follows_target(self):
        """工具偏航要跟着目标方位角走。

        恒定 yaw=0 时, 目标一移到侧舷就会把腕部 J7 顶到限位 (实测 r=0.45 处
        位置误差 2.9 cm) —— 也就是"够不着"。这条测试守的是那个回归。
        """
        from vla_vehicle.robot.arm import topdown_rotation

        self.env.reset(seed=0)
        for p in ((0.0, -0.45, -0.10), (0.30, 0.26, 0.16), (-0.20, -0.50, -0.08)):
            R = self.env._tcp_R_target(np.array(p, float))
            yaw = float(np.arctan2(p[1], p[0]))
            self.assertLess(float(np.abs(R - topdown_rotation(yaw)).max()), 1e-9)

    def test_step_contract(self):
        obs = self.env.reset(seed=3)
        a = np.array([0.2, -0.1, 0.0, 1.0], np.float32)
        obs2, rew, done, info = self.env.step(a)
        self.assertIsInstance(done, bool)
        for key in ("success", "target_in_bin", "holding", "step"):
            self.assertIn(key, info)
        self.assertFalse(np.allclose(obs.tcp, obs2.tcp), "动作没有让 TCP 移动")

    def test_idle_hold_is_stable(self):
        """零动作必须真的"原地不动" —— 曾经因为目标点用测量值累加而出现过棘轮漂移。"""
        obs = self.env.reset(seed=4)
        p0 = np.asarray(obs.tcp, float)
        for _ in range(10):
            obs, _, done, info = self.env.step(np.array([0, 0, 0, 1.0], np.float32))
            if done:
                break
        self.assertLess(float(np.linalg.norm(np.asarray(obs.tcp) - p0)), 0.004)


class TestExpertEpisode(unittest.TestCase):
    def test_expert_succeeds(self):
        from vla_vehicle.control.expert import ScriptedExpert

        env = build_min_scene(vehicle_scale=0.2)
        ok = False
        try:
            for seed in (0, 1, 2):
                obs = env.reset(seed=seed)
                exp = ScriptedExpert(env)
                info = env.evaluate()
                for _ in range(env.cfg.max_decisions):
                    obs, _, done, info = env.step(exp.act(obs))
                    if info["success"]:
                        break
                if info["success"]:
                    ok = True
                    break
        finally:
            env.close()
        self.assertTrue(ok, "脚本专家在 3 个种子上都没能完成抓放 —— 环境几何或时序被改坏了")


class TestWorkspace(unittest.TestCase):
    """所有要抓/要放的点都必须在工具朝下时可逆解到达。"""

    def test_targets_reachable(self):
        """所有要抓/要放的点都必须在工具朝下 + 偏航跟随方位角时逆解得到。"""
        env = build_min_scene(vehicle_scale=0.0)
        try:
            sc = env.cfg.scene
            for seed in range(4):
                env.reset(seed=seed)
                for s in env.object_states():
                    half = s["item"].shape.half_height
                    # 抓取点: 顶面以下 45 mm, 但离支撑面至少 4 mm
                    top = float(s["p"][2]) + half
                    gz = max(sc.surface_z + 0.004, top - 0.045)
                    p = np.array([s["p"][0], s["p"][1], gz])
                    # 120 次: 从 READY 到最远的舷外目标要绕一大圈, 60 次在边缘
                    # 位形上只收敛到 13 mm —— 那是迭代次数不够, 不是不可达
                    # (实跑时是热启动逐帧逼近的)。
                    sol = env.ik.solve(p, env._tcp_R_target(p), env.q_arm, max_iter=120)
                    self.assertLess(sol.pos_err, 0.005,
                                    f"seed={seed} {s['item'].name} 不可达 (err={sol.pos_err:.3f})")
                bx, by = sc.bin_center
                q = np.array([bx, by, 0.03])
                sol = env.ik.solve(q, env._tcp_R_target(q), env.q_arm, max_iter=120)
                self.assertLess(sol.pos_err, 0.005, "垃圾舱中心不可达")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
