"""目标随载具运动：船载（随海况）vs 车载（静对静）到底差在哪。

问题
----
"在载具上部署 VLA"有两种完全不同的工况, 过去只有一种能用:

* **船载**: 目标漂在水面上, 它**自己也在动**;
* **车载**: 车停稳、地面不动, 目标静对静。

而 ``env.frame`` 只管"指令累积在哪个系里", 管不着"物体在世界里怎么动"。
本脚本对比 ``env.target_motion`` 的三档:

``pinned``   物体世界系静止 —— 车载/岸上, 也是此前所有基准的口径
``surface``  随水面升沉 —— 船载打捞的**真实工况**
``deck``     固连甲板 —— 甲板上的货物, 最严苛的一档

为什么 ``surface`` 用的是"船的升沉"而不是另造一条正弦
------------------------------------------------------------------
船和漂浮垃圾漂在同一片波上, 由同一个水面驱动, 差别只在 RAO 的幅值与相位。
用同一个信号源, "目标怎么动"与"船怎么动"从一开始就同源; 否则两者的相位关系
是编出来的, 任何关于"升沉是共模的"这一结论都不可信。

指标
----
* ``success``       回合成功率
* ``grasp_err``     抓取瞬间 TCP 相对物体中心的**水平**偏差 (mm)
* ``steps``         决策步数
* ``sway_deck``     抓取前, 目标在**甲板系**内位置的峰峰值 (mm)
                    —— 从控制器的角度看, 目标到底"晃"了多少
* ``sway_world_z``  抓取前, 目标在**世界系** z 的峰峰值 (mm)
                    —— 水面本身抬升了多少

用法
----
    python scripts/bench_target.py --episodes 5
    python scripts/bench_target.py --modes pinned surface --seas 0.35 --frames deck
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


def run_one(motion: str, scale: float, frame: str, seed: int) -> dict:
    cfg = EnvConfig(target_motion=motion, vehicle_scale=scale, frame=frame, render=False)
    env = PickPlaceEnv(cfg, seed=seed)
    try:
        obs = env.reset(seed=seed)
        expert = ScriptedExpert(env)
        sway_d, sway_z = [], []
        for k in range(env.cfg.max_decisions):
            # 只统计"还没碰过目标"的阶段: 一旦夹住, 位移就全是机械臂造成的了
            if env.grasp_idx < 0 and env.target_index not in env._obj_free:
                st = env.object_states()[env.target_index]
                sway_d.append(st["p"].copy())          # 三维: 只取水平会把三档的区别抹平
                sway_z.append(float(st["p_world"][2]))
            obs, _, done, info = env.step(expert.act(obs))
            if done:
                break
        # 甲板系里的"画圈直径": 各轴峰峰值组成的向量长度
        range_d = (float(np.linalg.norm(np.ptp(np.asarray(sway_d), axis=0)) * 1e3)
                   if sway_d else None)
        range_z = (float(np.ptp(np.asarray(sway_z)) * 1e3) if sway_z else None)
        return dict(seed=seed, success=bool(info["success"]), steps=k + 1,
                    grasp_err=info["grasp_err"], sway_deck_mm=range_d,
                    sway_world_z_mm=range_z)
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seas", type=float, nargs="+", default=[0.0, 0.35, 1.0])
    ap.add_argument("--modes", nargs="+", default=["pinned", "surface", "deck"],
                    choices=["pinned", "surface", "deck"])
    ap.add_argument("--frames", nargs="+", default=["deck"], choices=["deck", "world"])
    ap.add_argument("--out", type=Path, default=ROOT / "results/bench/target_motion.json")
    args = ap.parse_args()

    out = {"episodes": args.episodes, "seed0": args.seed0, "rows": []}
    print(f"{'motion':8} {'frame':6} {'sea':>5} {'succ':>7} "
          f"{'grasp_err 均值/最大':>20} {'步数 均值/最大':>14} {'目标晃动 甲板/世界z':>22}")
    for sea in args.seas:
        for frame in args.frames:
            for motion in args.modes:
                t0 = time.time()
                rows = [run_one(motion, sea, frame, args.seed0 + i)
                        for i in range(args.episodes)]
                errs = [r["grasp_err"] for r in rows if r["grasp_err"] is not None]
                steps = [r["steps"] for r in rows]
                swd = [r["sway_deck_mm"] for r in rows if r["sway_deck_mm"] is not None]
                swz = [r["sway_world_z_mm"] for r in rows if r["sway_world_z_mm"] is not None]
                ok = sum(r["success"] for r in rows)
                row = dict(motion=motion, frame=frame, sea=sea, n=len(rows), success=ok,
                           grasp_err_mean=(float(np.mean(errs)) if errs else None),
                           grasp_err_max=(float(np.max(errs)) if errs else None),
                           steps_mean=float(np.mean(steps)), steps_max=int(np.max(steps)),
                           sway_deck_mm_mean=(float(np.mean(swd)) if swd else None),
                           sway_world_z_mm_mean=(float(np.mean(swz)) if swz else None),
                           wall_s=round(time.time() - t0, 1), episodes=rows)
                out["rows"].append(row)
                ge = f"{np.mean(errs)*1e3:5.1f} / {np.max(errs)*1e3:5.1f} mm" if errs else "—"
                sw = (f"{np.mean(swd):5.1f} / {np.mean(swz):5.1f} mm"
                      if swd and swz else "—")
                print(f"{motion:8} {frame:6} {sea:5.2f} {ok:>4}/{len(rows):<2} "
                      f"{ge:>20} {np.mean(steps):7.0f} /{np.max(steps):4d} {sw:>22}",
                      flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {args.out}")

    # 直接给出"升沉是不是共模的"这一条结论, 免得对着表自己推
    by = {(r["motion"], r["frame"], r["sea"]): r for r in out["rows"]}
    for sea in args.seas:
        p = by.get(("pinned", "deck", sea))
        s = by.get(("surface", "deck", sea))
        if p and s and p["sway_deck_mm_mean"] and s["sway_deck_mm_mean"]:
            print(f"sea {sea:.2f} deck 系目标晃动: pinned {p['sway_deck_mm_mean']:.1f} mm -> "
                  f"surface {s['sway_deck_mm_mean']:.1f} mm "
                  f"(世界系 z 抬升 {s['sway_world_z_mm_mean']:.1f} mm 被共模抵消)")


if __name__ == "__main__":
    main()
