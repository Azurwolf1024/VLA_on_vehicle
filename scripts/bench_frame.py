"""参考系对照：`env.frame = deck | world` 在同一任务上的精度与代价。

回答的问题很具体：垃圾漂在**世界系固定**的水面上，而机械臂基座在晃 ——
把指令累积在甲板系（目标随船动）和累积在世界系（目标钉住）到底差多少？

指标
----
* ``success``    回合成功率（物体最终是否落进垃圾舱）
* ``grasp_err``  抓取瞬间 TCP 相对物体中心的**水平**偏差（米）。取水平是因为竖直方向
                 本来就有几何量（指尖落在顶面以下 45 mm），那不是误差。
                 这个量直接反映内环有没有把船摇补掉，是成功率饱和之后唯一还有区分度的指标。
* ``steps``      回合决策步数。``world`` 系要主动抵消船摇，每一段位移都会被船吃掉一部分，
                 所以步数是**代价**。

用法
----
    python scripts/bench_frame.py --episodes 5 --out results/bench/frame_compare.json
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

from vla_vehicle.control.expert import ScriptedExpert     # noqa: E402
from vla_vehicle.sim.env import EnvConfig, PickPlaceEnv   # noqa: E402


def run_one(frame: str, scale: float, seed: int) -> dict:
    env = PickPlaceEnv(EnvConfig(frame=frame, vehicle_scale=scale, render=False), seed=seed)
    try:
        obs = env.reset(seed=seed)
        expert = ScriptedExpert(env)
        for k in range(env.cfg.max_decisions):
            obs, _, done, info = env.step(expert.act(obs))
            if done:
                break
        return dict(seed=seed, success=bool(info["success"]), steps=k + 1,
                    grasp_err=info["grasp_err"])
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seas", type=float, nargs="+", default=[0.0, 0.35, 1.0])
    ap.add_argument("--frames", nargs="+", default=["deck", "world"],
                    choices=["deck", "world"])
    ap.add_argument("--out", type=Path, default=ROOT / "results/bench/frame_compare.json")
    args = ap.parse_args()

    out = {"episodes": args.episodes, "seed0": args.seed0, "rows": []}
    print(f"{'frame':6} {'sea':>5} {'succ':>6} {'grasp_err 均值/最大':>20} {'步数 均值/最大':>14}")
    for sea in args.seas:
        for frame in args.frames:
            t0 = time.time()
            rows = [run_one(frame, sea, args.seed0 + i) for i in range(args.episodes)]
            errs = [r["grasp_err"] for r in rows if r["grasp_err"] is not None]
            steps = [r["steps"] for r in rows]
            ok = sum(r["success"] for r in rows)
            row = dict(frame=frame, sea=sea, n=len(rows), success=ok,
                       grasp_err_mean=(float(np.mean(errs)) if errs else None),
                       grasp_err_max=(float(np.max(errs)) if errs else None),
                       steps_mean=float(np.mean(steps)), steps_max=int(np.max(steps)),
                       wall_s=round(time.time() - t0, 1), episodes=rows)
            out["rows"].append(row)
            ge = f"{np.mean(errs)*1e3:5.1f} / {np.max(errs)*1e3:5.1f} mm" if errs else "—"
            print(f"{frame:6} {sea:5.2f} {ok:>3}/{len(rows):<2} {ge:>20} "
                  f"{np.mean(steps):7.0f} /{np.max(steps):4d}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {args.out}")

    # 提炼一句结论，避免读表时看错方向
    by = {(r["frame"], r["sea"]): r for r in out["rows"]}
    if ("deck", 1.0) in by and ("world", 1.0) in by:
        d, w = by[("deck", 1.0)], by[("world", 1.0)]
        if d["grasp_err_mean"] and w["grasp_err_mean"]:
            print(f"sea 1.0: 抓取偏差 deck {d['grasp_err_mean']*1e3:.1f} mm -> "
                  f"world {w['grasp_err_mean']*1e3:.1f} mm "
                  f"({d['grasp_err_mean']/w['grasp_err_mean']:.1f}×), "
                  f"步数 {d['steps_mean']:.0f} -> {w['steps_mean']:.0f}")


if __name__ == "__main__":
    main()
