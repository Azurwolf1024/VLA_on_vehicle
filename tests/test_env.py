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

    def test_objects_rest_on_deck(self):
        self.env.reset(seed=2)
        sc = self.env.cfg.scene
        for s in self.env.object_states():
            z = float(s["p"][2])
            half = s["item"].shape.half_height
            self.assertAlmostEqual(z, half, delta=0.01,
                                   msg=f"{s['item'].name} 没落在甲板上 (z={z:.3f})")

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
        from vla_vehicle.robot.arm import topdown_rotation

        env = build_min_scene(vehicle_scale=0.0)
        try:
            for seed in range(4):
                env.reset(seed=seed)
                for s in env.object_states():
                    p = np.array([s["p"][0], s["p"][1], 0.02])
                    sol = env.ik.solve(p, topdown_rotation(0), env.q_arm, max_iter=40)
                    self.assertLess(sol.pos_err, 0.01,
                                    f"seed={seed} {s['item'].name} 不可达 (err={sol.pos_err:.3f})")
                bx, by = env.cfg.scene.bin_center
                sol = env.ik.solve(np.array([bx, by, 0.05]), topdown_rotation(0),
                                   env.q_arm, max_iter=40)
                self.assertLess(sol.pos_err, 0.01, "垃圾舱中心不可达")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
