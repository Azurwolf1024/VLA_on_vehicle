#!/usr/bin/env python
"""标定感知链路：检测器阈值 + 相机反投影精度。

为什么要有这个脚本
------------------------------------------------------------------
传统 CV 基线的精度完全由两件事决定：**检测到的像素质心**和**反投影用的平面高度**。
这两个都必须实测，不能靠"看起来差不多"：

* 检测器漏检/误检 → 直接抓空；
* 反投影偏 1 cm → 夹爪 84 mm 还能兜住，但偏差会稳定地吃进 ``grasp_err``。

脚本做的事：把物体在甲板系的**真值**投影到像素，与检测质心比较（这是检测误差）；
再把检测质心反投影回平面，与真值比较（这是定位误差）。两者分开报，才知道该修哪一段。

    python scripts/probe_perception.py --seeds 4
    python scripts/probe_perception.py --out runs/percept.png     # 顺便存一张标注图
"""

from __future__ import annotations

import argparse
import sys
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
from vla_vehicle.perception import ColorDetector, bow_camera, parse_target  # noqa: E402
from vla_vehicle.sim.env import PickPlaceEnv              # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--vehicle-scale", type=float, default=None)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    cfg = load_env_config(args.config, vehicle_scale=args.vehicle_scale)
    cfg.render = True
    cam_cfg = cfg.scene
    cam = bow_camera(cam_cfg)
    det = ColorDetector()
    plane_z = cam_cfg.surface_z          # 标定值: 水面/地面在甲板系的固定高度

    print(f"相机: f={cam.f:.1f}px  {cam.width}x{cam.height}  fovy={cam.fovy}°"
          f"  水面 z={plane_z:+.3f}")
    print(f"{'seed':>4} {'目标':10} {'真值(x,y)':>16} {'检测像素':>12} {'面积':>5} "
          f"{'重投影(px)':>10} {'定位误差':>9} {'只用水面':>9}")

    errs, errs_naive, miss = [], [], 0
    total = 0
    for s in range(args.seeds):
        env = PickPlaceEnv(cfg, seed=s)
        try:
            obs = env.reset(seed=s)
            rgb = obs.global_rgb
            states = env.object_states()
            tgt = states[env.target_index]
            key = tgt["item"].color.key
            d = det.find(rgb, key)
            total += 1
            truth = np.asarray(tgt["p"], float)
            proj = cam.project(truth)
            if d is None:
                miss += 1
                print(f"{s:>4} {key:10} {truth[0]:7.3f},{truth[1]:7.3f}  "
                      f"{'未检出':>12}")
                continue
            px_err = float(np.linalg.norm(d.centroid - np.array(proj[:2])))
            # 流水线能拿到的平面高度 = 水面 + **物体类别的半高**（从指令解析出的 shape 查表）。
            # 不补这一项就会偏 10–50 mm: 视线穿过物体中心后还要往下延伸到水面,
            # 延伸量 = half_height / tan(俯角)。
            z_obj = plane_z + tgt["item"].shape.half_height
            p_est = cam.backproject(d.centroid[0], d.centroid[1], z_obj)
            loc = float("nan") if p_est is None else float(np.linalg.norm(p_est[:2] - truth[:2]))
            # 对照: 直接把检测质心反投影到水面（几何模型漏掉物体高度）
            p0 = cam.backproject(d.centroid[0], d.centroid[1], plane_z)
            loc0 = float("nan") if p0 is None else float(np.linalg.norm(p0[:2] - truth[:2]))
            errs.append(loc)
            errs_naive.append(loc0)
            print(f"{s:>4} {key:10} {truth[0]:7.3f},{truth[1]:7.3f}  "
                  f"({d.centroid[0]:5.1f},{d.centroid[1]:5.1f}) {d.area:5d} "
                  f"{px_err:10.2f} {loc*1e3:8.1f} mm {loc0*1e3:7.1f} mm")
        finally:
            env.close()

    if errs:
        e, e0 = np.asarray(errs), np.asarray(errs_naive)
        print(f"\n定位误差: 中位 {np.median(e)*1e3:.1f} mm, 最大 {e.max()*1e3:.1f} mm, "
              f"漏检 {miss}/{total}")
        print(f"对照(只反投到水面): 中位 {np.median(e0)*1e3:.1f} mm, "
              f"最大 {e0.max()*1e3:.1f} mm")


if __name__ == "__main__":
    main()
