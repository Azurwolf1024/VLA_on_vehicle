"""对照两种内环: 关节位置伺服 vs 上游解析模型 + TSID 力矩环。

两个问题分开测, 不要混在一起
------------------------------------------------------------------
1. **任务能不能做完**（甲板系目标）。垃圾就摆在甲板上, 所以任务本来就该在甲板系
   定义 —— 这一列比的是"两种内环谁把 TCP 送到位"。
2. **船摇能不能被压住**（世界系目标）。这才是上游那套"基座运动前馈"要解决的问题。
   做法是零动作保持 3 s, 量 TCP 在**世界系**里游走多少: 甲板系目标会跟着船走,
   世界系目标不会。

用法::

    python scripts/bench_control.py --episodes 4 --out results/bench/control.json
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

from vla_vehicle import deps as _deps                     # noqa: E402
try:
    _deps.ensure_upstream()
except _deps.MissingUpstreamError as _e:
    sys.exit(f"\n[依赖缺失] {_e}\n")

from vla_vehicle.control.expert import ScriptedExpert      # noqa: E402
from vla_vehicle.sim.env import EnvConfig, PickPlaceEnv     # noqa: E402


def _cfg(mode: str, frame: str, sea: float) -> EnvConfig:
    return EnvConfig(control_mode=mode, frame=frame, render=False,
                     vehicle_enabled=sea > 0, vehicle_scale=sea)


# --------------------------------------------------------------------------- #
def bench_task(mode: str, frame: str, sea: float, episodes: int, seed0: int) -> dict:
    ok, steps, times = 0, [], []
    for i in range(episodes):
        seed = seed0 + i
        env = PickPlaceEnv(_cfg(mode, frame, sea), seed=seed)
        obs = env.reset(seed=seed)
        exp = ScriptedExpert(env)
        t0 = time.time()
        for k in range(env.cfg.max_decisions):
            obs, _, done, info = env.step(exp.act(obs))
            if info["success"] or exp.done:
                break
        times.append(time.time() - t0)
        steps.append(k + 1)
        ok += int(info["success"])
        env.close()
    return dict(mode=mode, frame=frame, sea=sea, episodes=episodes,
                success=ok, rate=ok / episodes,
                mean_steps=float(np.mean(steps)),
                sec_per_episode=float(np.mean(times)))


def bench_hold(mode: str, frame: str, sea: float, seconds: float, seed: int) -> dict:
    """零动作保持, 量 TCP 在**世界系**的游走（船摇抑制能力）。"""
    env = PickPlaceEnv(_cfg(mode, frame, sea), seed=seed)
    obs = env.reset(seed=seed)

    def world_tcp() -> np.ndarray:
        R, p, _, _ = env._vessel_state(env.t)
        return p + R @ np.asarray(obs.tcp, float)

    p0 = world_tcp()
    n = int(round(seconds / env.cfg.dt_decision))
    dev = []
    for _ in range(n):
        obs, _, done, _ = env.step(np.zeros(4))
        dev.append(float(np.linalg.norm(world_tcp() - p0)))
        if done:
            break
    stats = env.tsid.stats if env.tsid is not None else {}
    env.close()
    return dict(mode=mode, frame=frame, sea=sea, seconds=seconds,
                mean_dev=float(np.mean(dev)), max_dev=float(np.max(dev)),
                tsid=stats)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--out", type=str, default="results/bench/control.json")
    ap.add_argument("--skip-task", action="store_true")
    ap.add_argument("--skip-hold", action="store_true")
    args = ap.parse_args()

    report = {"task": [], "hold": []}

    if not args.skip_task:
        print("== 任务成功率（甲板系目标, 专家策略）==")
        print(f"{'内环':<6}{'海况':<8}{'成功':<10}{'平均步数':<10}{'秒/回合'}")
        for mode in ("pos", "tsid"):
            for sea in (0.0, 0.35, 1.0):
                r = bench_task(mode, "deck", sea, args.episodes, args.seed0)
                report["task"].append(r)
                print(f"{mode:<8}{sea:<9.2f}{r['success']}/{r['episodes']:<8}"
                      f"{r['mean_steps']:<11.1f}{r['sec_per_episode']:.1f}")

    if not args.skip_hold:
        print("\n== 世界系定点保持（零动作 3 s, 有义波高 sea=1.0）==")
        print(f"{'内环/参考系':<18}{'平均游走(mm)':<16}{'最大游走(mm)'}")
        for mode, frame in (("pos", "deck"), ("tsid", "deck"), ("tsid", "world")):
            r = bench_hold(mode, frame, 1.0, 3.0, args.seed0)
            report["hold"].append(r)
            print(f"{mode + '/' + frame:<20}{r['mean_dev'] * 1000:<17.1f}"
                  f"{r['max_dev'] * 1000:.1f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
