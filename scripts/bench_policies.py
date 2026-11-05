#!/usr/bin/env python
"""三种抓取方法的同条件对比：传统 CV / 视觉伺服 / 学习式策略。

对比纪律
------------------------------------------------------------------
三种方法必须看到**同样的观测**、用**同样的动作接口**、跑**同样的回合**：

* 同一个 `Obs`（两张 160×160 图 + 关节 + TCP + 夹爪 + 指令）；
* 同一个 4 维动作（甲板系 TCP 增量 ±2 cm + 夹爪 0/1）；
* 同一批 seed、同一档海况、同一个成功判据。

**唯一允许的差别是"怎么把眼睛变成动作"**。所以：

======================  ==========================================
方法                     拿到什么，不拿什么
======================  ==========================================
`expert`（上界）          甲板系真值物体位姿 —— 仿真特权，只作上界
`cv`（检测式）            颜色分割 + 相机标定 + 物体类别半高先验；每阶段看一次
`pbvs`（视觉伺服）        同上，但**每一步**都重新检测并闭环
`bc`（学习式）            只有像素和本体感觉，不碰任何标定
======================  ==========================================

三者都不读 `env.object_states()`（`expert` 除外，它就是要当上界）。

"视觉伺服"这一档为什么是 PBVS 而不是 IBVS
------------------------------------------------------------------
试过、失败过，原因是几何而非调参，细节写在 `policies/ibvs.py` 的 docstring 里：
腕部相机瞄准 TCP 而物体在 TCP 下方 25 mm（期望像素落在画面边缘）；
改用全局 eye-to-hand 后像素能收敛到 1 px，抓取误差仍有 25 mm ——
斜视单目下"同一视线"不等于"同一 xy"。所以用基于位置的伺服。

用法
----
    python scripts/bench_policies.py --episodes 5
    python scripts/bench_policies.py --policies cv pbvs --seas 0.35 1.0
    python scripts/bench_policies.py --policies bc --ckpt runs/bc/policy.pt
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

from vla_vehicle.config import load_env_config            # noqa: E402
from vla_vehicle.control.expert import ScriptedExpert     # noqa: E402
from vla_vehicle.sim.env import PickPlaceEnv              # noqa: E402


def build(name: str, env, args):
    if name == "expert":
        return ScriptedExpert(env)
    if name == "cv":
        from vla_vehicle.policies.cv_grasp import CVGraspPolicy
        return CVGraspPolicy(env)
    if name == "pbvs":
        from vla_vehicle.policies.ibvs import PBVSPolicy
        return PBVSPolicy(env)
    if name == "bc":
        from vla_vehicle.policies.bc import BCPolicy
        return BCPolicy.load(args.ckpt)
    if name == "random":
        from vla_vehicle.policies.base import RandomPolicy
        return RandomPolicy(seed=args.seed0)
    raise ValueError(name)


def run_one(name: str, sea: float, seed: int, args) -> dict:
    cfg = load_env_config(args.config, vehicle_scale=sea)
    cfg.render = True                     # 视觉方法必须真的看到图
    env = PickPlaceEnv(cfg, seed=seed)
    try:
        obs = env.reset(seed=seed)
        pol = build(name, env, args)
        pol.reset(obs.instruction)
        k = 0
        for k in range(cfg.max_decisions):
            obs, _, done, info = env.step(pol.act(obs))
            if done:
                break
        extra = pol.info() if hasattr(pol, "info") else {}
        return dict(seed=seed, success=bool(info["success"]), steps=k + 1,
                    grasp_err=info["grasp_err"],
                    phase=extra.get("phase"), hits=extra.get("hits"),
                    misses=extra.get("misses"))
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seas", type=float, nargs="+", default=[0.0, 0.35, 1.0])
    ap.add_argument("--policies", nargs="+",
                    default=["expert", "cv", "pbvs", "bc"],
                    choices=["expert", "cv", "pbvs", "bc", "random"])
    ap.add_argument("--ckpt", default=str(ROOT / "runs/bc/policy.pt"))
    ap.add_argument("--out", type=Path, default=ROOT / "results/bench/policies.json")
    args = ap.parse_args()

    out = {"episodes": args.episodes, "seed0": args.seed0, "rows": []}
    print(f"{'policy':7} {'sea':>5} {'succ':>7} {'grasp_err 均值/最大':>20} "
          f"{'步数 均值/最大':>14}")
    for sea in args.seas:
        for name in args.policies:
            t0 = time.time()
            rows = [run_one(name, sea, args.seed0 + i, args) for i in range(args.episodes)]
            errs = [r["grasp_err"] for r in rows if r["grasp_err"] is not None]
            steps = [r["steps"] for r in rows]
            ok = sum(r["success"] for r in rows)
            row = dict(policy=name, sea=sea, n=len(rows), success=ok,
                       success_rate=ok / len(rows),
                       grasp_err_mean=(float(np.mean(errs)) if errs else None),
                       grasp_err_max=(float(np.max(errs)) if errs else None),
                       steps_mean=float(np.mean(steps)), steps_max=int(np.max(steps)),
                       wall_s=round(time.time() - t0, 1), episodes=rows)
            out["rows"].append(row)
            ge = f"{np.mean(errs)*1e3:5.1f} / {np.max(errs)*1e3:5.1f} mm" if errs else "—"
            print(f"{name:7} {sea:5.2f} {ok:>4}/{len(rows):<2} {ge:>20} "
                  f"{np.mean(steps):7.0f} /{np.max(steps):4d}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {args.out}")

    # 一句话把"谁比谁好在哪"点出来，避免读表时把精度和成功率混着看
    print()
    for sea in args.seas:
        r = {row["policy"]: row for row in out["rows"] if row["sea"] == sea}
        parts = []
        for name in args.policies:
            if name in r and r[name]["grasp_err_mean"] is not None:
                parts.append(f"{name} {r[name]['success']}/{r[name]['n']} 条、"
                             f"{r[name]['grasp_err_mean']*1e3:.1f} mm")
        if parts:
            print(f"sea {sea:.2f}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
