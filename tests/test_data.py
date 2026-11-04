"""数据层与语言分词的检查。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests._util import build_min_scene


class TestEpisodeIO(unittest.TestCase):
    def test_roundtrip(self):
        from vla_vehicle.data.schema import Episode

        T, H, W = 3, 8, 8
        ep = Episode(
            global_rgb=np.random.randint(0, 255, (T, H, W, 3), np.uint8),
            wrist_rgb=np.random.randint(0, 255, (T, H, W, 3), np.uint8),
            q=np.zeros((T, 7), np.float32), dq=np.zeros((T, 7), np.float32),
            tcp=np.zeros((T, 3), np.float32), grip=np.zeros(T, np.float32),
            action=np.zeros((T, 4), np.float32),
            obj_track=np.zeros((T, 3, 3), np.float32),
            tokens=np.array([1, 2, 3], np.int16),
            instruction="把红色易拉罐丢进垃圾舱", success=True, target_index=1,
            seed=7, phase=["hover"] * T, meta={"k": "v"},
        )
        with tempfile.TemporaryDirectory() as td:
            p = ep.to_npz(Path(td) / "a.npz")
            ep2 = Episode.load(p)
        self.assertEqual(ep2.instruction, ep.instruction)
        self.assertTrue(ep2.success)
        self.assertEqual(ep2.length, T)
        np.testing.assert_array_equal(ep2.global_rgb, ep.global_rgb)
        np.testing.assert_array_equal(ep2.tokens, ep.tokens)


class TestRecorder(unittest.TestCase):
    def test_record_one_episode(self):
        from vla_vehicle.control.expert import ScriptedExpert
        from vla_vehicle.data.recorder import Recorder
        from vla_vehicle.data.schema import Episode

        env = build_min_scene(vehicle_scale=0.0)
        try:
            obs = env.reset(seed=0)
            rec = Recorder(env, env.instruction, obs.tokens, seed=0)
            exp = ScriptedExpert(env)
            info = env.evaluate()
            for _ in range(12):
                a = exp.act(obs)
                rec.add(obs, a, phase=exp.phase)
                obs, _, done, info = env.step(a)
                if done:
                    break
            ep = rec.finish(info["success"], env.target_index)
        finally:
            env.close()
        self.assertIsInstance(ep, Episode)
        self.assertEqual(ep.length, len(rec))
        self.assertEqual(ep.action.shape[1], 4)
        self.assertEqual(ep.obj_track.shape[1], env.cfg.n_objects)
        self.assertTrue(np.all(np.abs(ep.action[:, :3]) <= 1.0 + 1e-6))


class TestTokenizer(unittest.TestCase):
    def test_roundtrip_zh(self):
        from vla_vehicle.sim.objects import Tokenizer

        tok = Tokenizer()
        text = "把红色的易拉罐放进垃圾舱"
        ids = tok.encode(text)
        self.assertEqual(len(ids), tok.max_len)
        self.assertEqual(tok.decode(ids), text)

    def test_roundtrip_en(self):
        from vla_vehicle.sim.objects import Tokenizer, make_instruction, SHAPES, COLORS, TrashItem

        tok = Tokenizer()
        item = TrashItem(index=0, shape=SHAPES["can"], color=COLORS["red"])
        text = make_instruction(item, lang="en")
        self.assertEqual(tok.decode(tok.encode(text)), text)


if __name__ == "__main__":
    unittest.main()
