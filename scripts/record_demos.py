#!/usr/bin/env python
"""采集示范数据: 脚本专家 -> ``.npz`` 回合。

    python scripts/record_demos.py --episodes 20 --out data/demos
    python scripts/record_demos.py --episodes 20 --vehicle-scale 0.0 --out data/calm
    python scripts/record_demos.py --episodes 5 --lang en --out data/en

产出
------------------------------------------------------------------
``<out>/ep0000.npz`` ... 每条回合一份; ``<out>/index.jsonl`` 一行一条摘要。
失败回合默认也保存（``--save-failed``）, 方便之后分析失败模式; 训练时用
``--only-success``（默认）过滤。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 先做依赖检查再导入重模块: 缺上游时给一句能照做的提示, 而不是一屏 traceback
from vla_vehicle import deps as _deps                     # noqa: E402
try:
    _deps.ensure_upstream()
except _deps.MissingUpstreamError as _e:
    sys.exit(f"\n[依赖缺失] {_e}\n")

from vla_vehicle.config import load_env_config           # noqa: E402
from vla_vehicle.control.expert import ExpertConfig, ScriptedExpert  # noqa: E402
from vla_vehicle.data.recorder import Recorder          # noqa: E402
from vla_vehicle.sim.env import PickPlaceEnv            # noqa: E402


def run_episode(env: PickPlaceEnv, seed: int, expert_cfg: ExpertConfig):
    obs = env.reset(seed=seed)
    expert = ScriptedExpert(env, expert_cfg)
    rec = Recorder(env, env.instruction, obs.tokens, seed=seed)
    info = env.evaluate()
    for _ in range(env.cfg.max_decisions):
        a = expert.act(obs)
        rec.add(obs, a, phase=expert.phase)
        obs, _, done, info = env.step(a)
        if info["success"]:
            rec.add(obs, np.array([0, 0, 0, 1.0], np.float32), phase="done")
            break
        if expert.done:
            break
    ep = rec.finish(success=info["success"], target_index=env.target_index,
                    extra=dict(t0=env.t0, n_steps=env.step_count,
                               stalled=expert.stalled))
    return ep, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", type=str, default="data/demos")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--config", type=str, default=None, help="YAML 配置")
    ap.add_argument("--vehicle-scale", type=float, default=None)
    ap.add_argument("--vehicle-kind", type=str, default=None)
    ap.add_argument("--vehicle", dest="vehicle_enabled", action="store_true", default=None)
    ap.add_argument("--no-vehicle", dest="vehicle_enabled", action="store_false")
    ap.add_argument("--lang", type=str, default=None, choices=["zh", "en"])
    ap.add_argument("--max-decisions", type=int, default=None)
    ap.add_argument("--save-failed", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_env_config(args.config,
                          **_clean(dict(vehicle_scale=args.vehicle_scale,
                                        vehicle_kind=args.vehicle_kind,
                                        vehicle_enabled=args.vehicle_enabled,
                                        lang=args.lang,
                                        max_decisions=args.max_decisions)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.jsonl"
    index_path.write_text("", encoding="utf-8")

    expert_cfg = ExpertConfig()
    t_start = time.time()
    n_ok = 0
    for k in range(args.episodes):
        seed = args.seed0 + k
        env = PickPlaceEnv(cfg, seed=seed)
        ep, info = run_episode(env, seed, expert_cfg)
        env.close()
        n_ok += int(ep.success)
        if ep.success or args.save_failed:
            path = out / f"ep{seed:04d}.npz"
            ep.to_npz(path)
            with index_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(file=path.name, seed=seed, length=ep.length,
                                        success=ep.success, instruction=ep.instruction,
                                        target=ep.target_index),
                                   ensure_ascii=False) + "\n")
        if not args.quiet:
            print(f"[{k + 1}/{args.episodes}] seed={seed} len={ep.length:3d} "
                  f"success={ep.success} \"{ep.instruction}\"", flush=True)

    dt = time.time() - t_start
    print(f"\n完成 {args.episodes} 条, 成功 {n_ok} ({n_ok / max(1, args.episodes):.0%}), "
          f"用时 {dt:.1f}s ({dt / max(1, args.episodes):.2f}s/条)")
    print(f"数据目录: {out.resolve()}")
    return 0 if n_ok else 1


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


if __name__ == "__main__":
    raise SystemExit(main())
