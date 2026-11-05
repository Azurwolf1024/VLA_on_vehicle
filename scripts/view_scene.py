#!/usr/bin/env python
"""可视化: 交互式看场景 / 导出回合图像。

    python scripts/view_scene.py                       # 打开 MuJoCo 窗口跑专家
    python scripts/view_scene.py --headless --frames 4 --out runs/scene.png

无显示环境（远程/CI）时用 ``--headless``: 只做离屏渲染并把若干帧拼成一张图。
"""

from __future__ import annotations

import argparse
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

from vla_vehicle.config import load_env_config            # noqa: E402
from vla_vehicle.control.expert import ScriptedExpert     # noqa: E402
from vla_vehicle.sim.env import PickPlaceEnv              # noqa: E402


def headless(env, out: Path, frames: int, steps_every: int = 12) -> None:
    """按**阶段**取帧，而不是按固定步数。

    固定步数取样在"舷外取物"任务里几乎全落在接近段（前 40 步都在伸臂），
    出来的图看不出抓、抬、投。按阶段各取一帧能直接讲清楚一条回合。
    """
    import cv2
    obs = env.reset(seed=0)
    expert = ScriptedExpert(env)
    wanted = ["hover", "descend", "lift", "descend_bin"][:frames]
    rows = []
    for k in range(env.cfg.max_decisions):
        if expert.phase in wanted and expert.phase not in [r[0] for r in rows]:
            rows.append((expert.phase, np.hstack([obs.global_rgb, obs.wrist_rgb])))
        obs, _, done, info = env.step(expert.act(obs))
        if done:
            break
    if not rows:
        rows = [("end", np.hstack([obs.global_rgb, obs.wrist_rgb]))]
    img = np.vstack([r[1] for r in rows])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img[..., ::-1])
    print(f"已保存 {out} ({img.shape[1]}x{img.shape[0]}), 阶段={[r[0] for r in rows]}, "
          f"success={info['success']}")


def interactive(env) -> None:
    import mujoco
    import mujoco.viewer
    print("窗口已打开: 关闭窗口结束。专家策略正在执行。")
    obs = env.reset(seed=0)
    expert = ScriptedExpert(env)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 1.6
        viewer.cam.lookat[:] = [0.15, 0.05, 0.35]
        k = 0
        while viewer.is_running():
            t0 = time.perf_counter()
            obs, _, done, info = env.step(expert.act(obs))
            if done:
                print(f"回合结束 success={info['success']}, 重开")
                obs = env.reset(seed=int(time.time()) % 100000)
                expert = ScriptedExpert(env)
            viewer.sync()
            dt = 1.0 / 10.0 - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)
            k += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vehicle-scale", type=float, default=None)
    ap.add_argument("--control-mode", choices=("pos", "tsid"), default=None,
                    help="内环: pos=关节位置伺服(默认) | tsid=上游 TSID 力矩环")
    ap.add_argument("--control-frame", choices=("deck", "world"), default=None,
                    help="任务参考系: deck=随甲板动 | world=钉在世界系")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--out", type=str, default="runs/scene.png")
    args = ap.parse_args()

    cfg = load_env_config(args.config, vehicle_scale=args.vehicle_scale,
                          control_mode=args.control_mode, frame=args.control_frame)
    env = PickPlaceEnv(cfg, seed=args.seed)
    try:
        if args.headless:
            headless(env, Path(args.out), args.frames)
        else:
            interactive(env)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
