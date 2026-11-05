"""对照: 内环 (pos / tsid) × 参考系 (deck / world) × 海况。

任务本身是**世界系**的: 垃圾漂在水面上, 不跟船走; 只有甲板上的垃圾舱是甲板系的。
所以这里比两件事:

* ``frame="deck"`` —— 指令点跟着甲板走。船一摇, 末端就跟着摇; 而水面上的垃圾不动,
  抓的时候就对不上。
* ``frame="world"`` —— 指令点钉在世界上。抓水面垃圾对了, 但往甲板上的舱里丢的时候,
  舱会从指令点下面"漂走"。

两个都不是全对的 —— 这正是这条任务真正难的地方; 表里看的是**差距有多大**。

除了成功率, 还报一个更敏感的量: **抓取瞬间 TCP 与物体中心的世界系偏差**
(``info["grasp_err"]``)。成功率在 6 个种子上很容易打平, 这个量不会。

用法::

    python scripts/bench_control.py --episodes 5 --out results/bench/control.json
    python scripts/bench_control.py --skip-tsid --skip-hold      # 只跑位置伺服那张表
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


def bench_task(mode: str, frame: str, sea: float, episodes: int, seed0: int) -> dict:
    ok, steps, errs, times = 0, [], [], []
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
        if info.get("grasp_err") is not None:
            errs.append(float(info["grasp_err"]))
        env.close()
    return dict(mode=mode, frame=frame, sea=sea, episodes=episodes,
                success=ok, rate=ok / episodes,
                mean_steps=float(np.mean(steps)),
                # 只统计"确实发生了抓取"的回合: 没抓到的回合没有这个量
                mean_grasp_err=float(np.mean(errs)) if errs else None,
                sec_per_episode=float(np.mean(times)))


def bench_hold(mode: str, frame: str, sea: float, seconds: float = 3.0, seed0: int = 200) -> dict:
    """零动作保持 ``seconds`` 秒, 量 TCP 在**世界系**里的游走。

    这是"船摇抑制"的直接度量: 目标钉不钉在世界系, 决定了末端会不会跟着船跑。
    ``deck`` 系下目标随船动 -> 末端也跟着走, 这不是控制器的错;
    ``world`` 系下目标钉住 -> 剩下的就是控制器真正压住了多少。
    """
    env = PickPlaceEnv(_cfg(mode, frame, sea), seed=seed0)
    env.reset(seed=seed0)
    n = int(round(seconds / env.cfg.dt_decision))     # 决策步 = 0.1 s -> 3 s = 30 步
    wanders = []
    R0, p0, _, _ = env._vessel_state(env.t)
    ref = p0 + R0 @ env.tcp_pose()[1]
    for _ in range(n):
        env.step(np.zeros(4))
        R, p, _, _ = env._vessel_state(env.t)
        wanders.append(float(np.linalg.norm((p + R @ env.tcp_pose()[1]) - ref)))
    env.close()
    return dict(mode=mode, frame=frame, sea=sea, seconds=seconds,
                wander_mean=float(np.mean(wanders)), wander_max=float(np.max(wanders)))


def _row(r: dict) -> str:
    ge = "—" if r["mean_grasp_err"] is None else f"{r['mean_grasp_err'] * 1000:.1f}"
    return (f"{r['frame']:<9}{r['sea']:<8.2f}{r['success']}/{r['episodes']:<7}"
            f"{r['mean_steps']:<11.1f}{ge:<15}{r['sec_per_episode']:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--tsid-episodes", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=200)
    ap.add_argument("--out", type=str, default="results/bench/control.json")
    ap.add_argument("--skip-tsid", action="store_true")
    ap.add_argument("--skip-hold", action="store_true")
    args = ap.parse_args()

    report = {"task": [], "tsid": [], "hold": []}
    head = (f"{'参考系':<9}{'海况':<8}{'成功':<9}{'平均步数':<11}"
            f"{'抓取偏差(mm)':<15}{'秒/回合'}")

    print("== 位置伺服: 参考系 × 海况（舷外水面取物 -> 甲板垃圾舱）==")
    print(head)
    for frame in ("deck", "world"):
        for sea in (0.0, 0.35, 1.0):
            r = bench_task("pos", frame, sea, args.episodes, args.seed0)
            report["task"].append(r)
            print(_row(r), flush=True)

    if not args.skip_tsid:
        print("\n== 力矩环 (上游 TSID) 在同一条任务上（sea 0.35）==")
        print(head)
        for frame in ("deck", "world"):
            r = bench_task("tsid", frame, 0.35, args.tsid_episodes, args.seed0)
            report["tsid"].append(r)
            print(_row(r), flush=True)

    if not args.skip_hold:
        print("\n== 船摇抑制: 零动作保持 3 s, TCP 世界系游走 (sea 1.0) ==")
        print(f"{'内环':<7}{'参考系':<9}{'均值(mm)':<11}{'最大(mm)'}")
        for mode, frame in (("pos", "deck"), ("tsid", "deck"), ("tsid", "world")):
            r = bench_hold(mode, frame, 1.0)
            report["hold"].append(r)
            print(f"{mode:<7}{frame:<9}{r['wander_mean']*1e3:<11.1f}"
                  f"{r['wander_max']*1e3:.1f}", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
