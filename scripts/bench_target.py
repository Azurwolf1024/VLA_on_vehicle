"""载具工况矩阵：目标怎么动 × 基座怎么动，到底各自贡献多少误差。

背景
----
"在载具上部署 VLA"不是一种工况。真正正交的是两件事：

* **基座动没动** —— 车停稳 / 船在摇；
* **目标动没动** —— 地面不说谎 / 垃圾漂在水面上、随波起伏。

过去只有一种可用（`pinned` + 船摇）。本脚本把五种组合都跑一遍：

===================  ==========  ================  ============================
case                 基座        目标              对应现实
===================  ==========  ================  ============================
``road-static``      静止        静止              车载静对静（基线）
``shore-sea``        静止        随波              **岸边/岸基**: 车停在码头, 夹海里的
``sea-pinned``       船摇        钉在世界系        船载, 但目标被当成固定点（旧口径）
``sea-surface``      船摇        随波              **船载打捞的真实工况**
``sea-deck``         船摇        固连甲板          甲板上的货物（最严苛）
===================  ==========  ================  ============================

波面从哪来
----------
``surface`` 档的升沉用的是**真实波面**，不是船体升沉。两者在本环境默认海况下
近乎反相：船升沉相对波面滞后 **151°**、幅值只有 **0.34 倍**（波周期 2 s 高于
heave 固有频率 1.70 rad/s，船来不及响应短波）。拿船升沉当波面会把目标驱动到
相反方向。细节见 ``PickPlaceEnv._build_wave`` 的 docstring。

空间相位差按深水色散 ``k = ω²/g`` 逐物体计入：0.5 m 的间距在 Tp=2 s 的浪上
约等于 **0.20 s** 的相位滞后 —— 与你说的"离得近相位差小"一致，但要注意它和
船体 RAO 的 0.84 s 是**两个不同的东西**，不能混为一谈。

指标
----
* ``success``     回合成功率
* ``grasp_err``   抓取瞬间 TCP 相对物体中心的水平偏差 (mm)
* ``steps``       决策步数
* ``sway_target`` 抓取前目标在**甲板系**内的三维晃动峰峰值 (mm) —— 目标视角
* ``sway_base``   甲板安装点在**世界系**的晃动峰峰值 (mm) —— 基座视角

后两个指标是把"目标动"与"基座动"分开看的关键：``road-static`` / ``shore-sea``
的 ``sway_base`` 恒为 0，两者的差异就纯粹来自目标自己动。

用法
----
    python scripts/bench_target.py --episodes 5
    python scripts/bench_target.py --cases road-static shore-sea --seas 0.35
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


#: case 名 -> EnvConfig 覆盖项。``base`` 表示基座动不动, ``target`` 是目标怎么动。
CASES = {
    "road-static": dict(base="静止", target="静止",
                        cfg=dict(vehicle_enabled=False, target_motion="pinned")),
    "shore-sea": dict(base="静止", target="随波",
                      cfg=dict(vehicle_enabled=False, target_motion="surface",
                               surface_source="independent")),
    "sea-pinned": dict(base="船摇", target="钉世界",
                       cfg=dict(vehicle_enabled=True, target_motion="pinned")),
    "sea-surface": dict(base="船摇", target="随波",
                        cfg=dict(vehicle_enabled=True, target_motion="surface")),
    "sea-deck": dict(base="船摇", target="固连甲板",
                     cfg=dict(vehicle_enabled=True, target_motion="deck")),
}


def run_one(case: str, sea: float, frame: str, seed: int) -> dict:
    over = dict(CASES[case]["cfg"])
    cfg = EnvConfig(frame=frame, render=False, vehicle_scale=sea,
                    surface_scale=sea, **over)
    env = PickPlaceEnv(cfg, seed=seed)
    try:
        obs = env.reset(seed=seed)
        expert = ScriptedExpert(env)
        sway_t, sway_z, sway_b = [], [], []
        for k in range(env.cfg.max_decisions):
            # 目标视角: 只统计"还没碰过目标"的阶段, 否则位移全是机械臂造成的
            if env.grasp_idx < 0 and env.target_index not in env._obj_free:
                st = env.object_states()[env.target_index]
                sway_t.append(st["p"].copy())          # 三维: 只取水平会把三档抹平
                sway_z.append(float(st["p_world"][2]))
            sway_b.append(env._deck_state(env.t)["p"].copy())
            obs, _, done, info = env.step(expert.act(obs))
            if done:
                break

        def rng3(seq):
            return float(np.linalg.norm(np.ptp(np.asarray(seq), axis=0)) * 1e3) if seq else None

        return dict(seed=seed, success=bool(info["success"]), steps=k + 1,
                    grasp_err=info["grasp_err"],
                    sway_target_mm=rng3(sway_t), sway_base_mm=rng3(sway_b),
                    sway_world_z_mm=(float(np.ptp(np.asarray(sway_z)) * 1e3)
                                     if sway_z else None))
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seas", type=float, nargs="+", default=[0.0, 0.35, 1.0])
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--frames", nargs="+", default=["deck"], choices=["deck", "world"])
    ap.add_argument("--out", type=Path, default=ROOT / "results/bench/target_motion.json")
    args = ap.parse_args()

    out = {"episodes": args.episodes, "seed0": args.seed0, "rows": []}
    print(f"{'case':12} {'base':5} {'tgt':7} {'frame':6} {'sea':>5} {'succ':>7} "
          f"{'grasp_err 均/最大':>18} {'步数':>5} {'目标晃':>8} {'基座晃':>8}")
    for sea in args.seas:
        for frame in args.frames:
            for case in args.cases:
                t0 = time.time()
                rows = [run_one(case, sea, frame, args.seed0 + i)
                        for i in range(args.episodes)]
                errs = [r["grasp_err"] for r in rows if r["grasp_err"] is not None]
                steps = [r["steps"] for r in rows]
                swt = [r["sway_target_mm"] for r in rows if r["sway_target_mm"] is not None]
                swb = [r["sway_base_mm"] for r in rows if r["sway_base_mm"] is not None]
                swz = [r["sway_world_z_mm"] for r in rows if r["sway_world_z_mm"] is not None]
                ok = sum(r["success"] for r in rows)
                row = dict(case=case, base=CASES[case]["base"], target=CASES[case]["target"],
                           frame=frame, sea=sea, n=len(rows), success=ok,
                           grasp_err_mean=(float(np.mean(errs)) if errs else None),
                           grasp_err_max=(float(np.max(errs)) if errs else None),
                           steps_mean=float(np.mean(steps)), steps_max=int(np.max(steps)),
                           sway_target_mm_mean=(float(np.mean(swt)) if swt else None),
                           sway_base_mm_mean=(float(np.mean(swb)) if swb else None),
                           sway_world_z_mm_mean=(float(np.mean(swz)) if swz else None),
                           wall_s=round(time.time() - t0, 1), episodes=rows)
                out["rows"].append(row)
                ge = f"{np.mean(errs)*1e3:5.1f} / {np.max(errs)*1e3:5.1f} mm" if errs else "—"
                print(f"{case:12} {CASES[case]['base']:5} {CASES[case]['target']:7} "
                      f"{frame:6} {sea:5.2f} {ok:>4}/{len(rows):<2} {ge:>18} "
                      f"{np.mean(steps):5.0f} {np.mean(swt) if swt else 0:7.1f} "
                      f"{np.mean(swb) if swb else 0:7.1f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {args.out}")

    # 把"目标动"与"基座动"的贡献拆开算清楚, 免得对着表自己推
    by = {(r["case"], r["sea"]): r for r in out["rows"]}
    for sea in args.seas:
        base = by.get(("road-static", sea))
        shore = by.get(("shore-sea", sea))
        sp = by.get(("sea-pinned", sea))
        ss = by.get(("sea-surface", sea))
        if not (base and shore and sp and ss):
            continue
        if None in (base["grasp_err_mean"], shore["grasp_err_mean"],
                    sp["grasp_err_mean"], ss["grasp_err_mean"]):
            continue
        print(f"sea {sea:.2f} 抓取偏差: 静对静 {base['grasp_err_mean']*1e3:.2f} → "
              f"仅目标动(岸边) {shore['grasp_err_mean']*1e3:.2f} → "
              f"仅基座动(船/钉世界) {sp['grasp_err_mean']*1e3:.2f} → "
              f"两者都动(船/随波) {ss['grasp_err_mean']*1e3:.2f} mm")


if __name__ == "__main__":
    main()
