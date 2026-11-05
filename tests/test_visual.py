"""视觉档（真实网格 / 骨架胶囊）不许改变被控对象。

换视觉的**唯一**正当理由是"看得像", 一旦把质量、惯量、碰撞也换了, 那所有
在胶囊档跑出来的跟踪/抓取结论就与网格档无关 —— 这正是最容易悄悄发生的事故:
``<geom type="mesh">`` 没有 ``<inertial>`` 时 MuJoCo 按网格体积当实心体估质量,
基座凭空多 0.73 kg (实测)。

因此这里守三条不变量:
1. 两档的 M(q) 与 bias 力在同一 q/dq 下逐位相同;
2. 两档的整臂质量相同（基座质量来自 URDF, 不来自网格体积）;
3. 视觉 geom 一律 ``contype=0`` —— 网格不能突然参与碰撞。
"""

from __future__ import annotations

import unittest

import numpy as np

from tests._util import load_rig


class TestVisualTiers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import mujoco

        from vla_vehicle.robot.arm import READY_Q, load_arm
        from vla_vehicle.robot.mjcf import build_arm, mesh_dir, solo_xml

        cls.has_mesh = mesh_dir() is not None
        if not cls.has_mesh:
            raise unittest.SkipTest("上游没有 models/nero_stl, 网格档无从比较")

        _, cls.spec, _ = load_arm()
        cls.rig = {}
        for tier in ("capsule", "mesh"):
            arm = build_arm(cls.spec, q_ref=READY_Q, name="nero", visual=tier)
            model = mujoco.MjModel.from_xml_string(solo_xml(arm, cls.spec))
            data = mujoco.MjData(model)
            data.qpos[: cls.spec.n] = READY_Q
            data.qvel[:] = 0.37
            mujoco.mj_forward(model, data)
            M = np.zeros((model.nv, model.nv), order="C")
            mujoco.mj_fullM(model, data, M)
            cls.rig[tier] = dict(model=model, M=M.copy(),
                                 bias=data.qfrc_bias.copy(), arm=arm)

    def test_dynamics_identical(self):
        a, b = self.rig["capsule"], self.rig["mesh"]
        np.testing.assert_allclose(a["M"], b["M"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(a["bias"], b["bias"], atol=1e-10, rtol=0)

    def test_mass_identical(self):
        a, b = self.rig["capsule"]["model"], self.rig["mesh"]["model"]
        self.assertAlmostEqual(float(a.body_mass.sum()), float(b.body_mass.sum()), places=9)

    def test_mesh_is_visual_only(self):
        model = self.rig["mesh"]["model"]
        n = self.spec.n
        # 机械臂连杆 + 基座: 8 个网格 geom, 全部不参与碰撞
        mesh_geoms = [g for g in range(model.ngeom) if model.geom_type[g] == 7]
        self.assertEqual(len(mesh_geoms), n + 1)
        for g in mesh_geoms:
            self.assertEqual(int(model.geom_contype[g]), 0)
            self.assertEqual(int(model.geom_conaffinity[g]), 0)

    def test_mesh_assets_wired(self):
        arm = self.rig["mesh"]["arm"]
        self.assertTrue(arm.mesh_assets.strip())
        self.assertTrue(arm.mesh_dir)
        self.assertIn("nero_link1", arm.mesh_assets)


class TestArmStillLoads(unittest.TestCase):
    """默认档（auto）在任何机器上都应能构建出可加载的模型。"""

    def test_default_builds(self):
        robot, spec, arm, model, data = load_rig()
        self.assertEqual(len(arm.joint_names), spec.n)
        self.assertIn(arm.visual, ("mesh", "capsule"))
        self.assertGreater(model.ngeom, 0)


if __name__ == "__main__":
    unittest.main()
