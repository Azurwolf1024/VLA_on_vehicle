#!/usr/bin/env python
"""把上游控制仓库 ``robotic_arm_on_vehicle`` 拉到 ``third_party/``。

本项目**不复制** ``veh_arm`` 的源码, 只做依赖解析（见 ``vla_vehicle/deps.py``）。
默认情况下把仓库放在兄弟目录即可; 本脚本给"只有一个仓库"的场景兜底。

    python scripts/fetch_upstream.py
    python scripts/fetch_upstream.py --url https://github.com/Azurwolf1024/robotic_arm_on_vehicle.git
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "third_party" / "robotic_arm_on_vehicle"
DEFAULT_URL = "https://github.com/Azurwolf1024/robotic_arm_on_vehicle.git"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--depth", type=int, default=1)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from vla_vehicle import deps
    try:
        root = deps.ensure_upstream(verbose=True)
        print(f"已可导入 veh_arm: {root}")
        return 0
    except deps.MissingUpstreamError as e:
        print(str(e))

    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"{DEST} 已存在, 更新…")
        subprocess.run(["git", "-C", str(DEST), "pull", "--ff-only"], check=False)
    else:
        cmd = ["git", "clone", f"--depth={args.depth}", args.url, str(DEST)]
        print("执行:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    from vla_vehicle import deps as d2
    try:
        print("已就绪:", d2.ensure_upstream(verbose=True))
        return 0
    except d2.MissingUpstreamError as e:
        print("仍然失败:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
