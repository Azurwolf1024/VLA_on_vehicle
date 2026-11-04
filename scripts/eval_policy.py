#!/usr/bin/env python
"""闭环评测: 把策略放进环境里跑 N 条回合, 统计成功率。

    python scripts/eval_policy.py --ckpt runs/bc/policy.pt --episodes 20
    python scripts/eval_policy.py --policy expert --episodes 20 --vehicle-scale 0.0
    python scripts/eval_policy.py --policy random --episodes 5      # 反例基线

注意 ``--seed0`` 只改回合起点, 与训练集是否重叠由你控制: 训练/测试要用不同 seed 段,
否则报出来的成功率会偏高（同一物体摆放被反复考）。
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

from vla_vehicle.config import load_env_config          # noqa: E402
from vla_vehicle.control.expert import ScriptedExpert   # noqa: E402
from vla_vehicle.sim.env import PickPlaceEnv            # noqa: E402


def make_policy(name: str, args, env):
    if name == "expert":
        return ScriptedExpert(env)
    if name == "random":
        from vla_vehicle.policies.base import RandomPolicy
        return RandomPolicy(seed=args.seed0)
    if name == "zero":
        from vla_vehicle.policies.base import ZeroPolicy
        return ZeroPolicy()
    if name == "bc":
        from vla_vehicle.policies.bc import BCPolicy
        return BCPolicy.load(args.ckpt, device=args.device)
    raise SystemExit(f"未知策略 {name!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=str, default=None,
                    choices=["expert", "bc", "random", "zero"])
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--vehicle-scale", type=float, default=None)
    ap.add_argument("--vehicle-kind", type=str, default=None)
    ap.add_argument("--no-vehicle", dest="vehicle_enabled", action="store_false", default=None)
    ap.add_argument("--max-decisions", type=int, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--json", type=str, default=None, help="把结果写到 json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.policy is None:
        args.policy = "bc" if args.ckpt else "expert"
    if args.policy == "bc" and not args.ckpt:
        raise SystemExit("--policy bc 需要 --ckpt")

    cfg = load_env_config(args.config,
                          **{k: v for k, v in dict(vehicle_scale=args.vehicle_scale,
                                                   vehicle_kind=args.vehicle_kind,
                                                   vehicle_enabled=args.vehicle_enabled,
                                                   max_decisions=args.max_decisions).items()
                             if v is not None})
    results = []
    t0 = time.time()
    for k in range(args.episodes):
        seed = args.seed0 + k
        env = PickPlaceEnv(cfg, seed=seed)
        pol = make_policy(args.policy, args, env)
        obs = env.reset(seed=seed)
        pol.reset(env.instruction)
        info = env.evaluate()
        for _ in range(env.cfg.max_decisions):
            obs, _, done, info = env.step(pol.act(obs))
            if done:
                break
        results.append(dict(seed=seed, success=info["success"],
                            steps=env.step_count, instruction=env.instruction,
                            target_in_bin=info["target_in_bin"]))
        if not args.quiet:
            print(f"[{k + 1}/{args.episodes}] seed={seed} success={info['success']} "
                  f"steps={env.step_count:3d} \"{env.instruction}\"", flush=True)
        env.close()

    n = len(results)
    sr = sum(r["success"] for r in results) / max(1, n)
    summary = dict(policy=args.policy, episodes=n, success_rate=sr,
                   mean_steps=float(np.mean([r["steps"] for r in results])),
                   vehicle_scale=cfg.vehicle_scale, vehicle_kind=cfg.vehicle_kind,
                   vehicle_enabled=cfg.vehicle_enabled,
                   elapsed_s=time.time() - t0, results=results)
    print(f"\n{args.policy}: 成功率 {sr:.0%} ({sum(r['success'] for r in results)}/{n}), "
          f"平均 {summary['mean_steps']:.0f} 步, "
          f"载具={cfg.vehicle_kind if cfg.vehicle_enabled else 'static'} "
          f"scale={cfg.vehicle_scale:.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"结果: {args.json}")
    return 0 if sr > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
