"""
上游依赖解析: 本仓库**不复制**机械臂模型与控制核心, 而是直接复用
``robotic_arm_on_vehicle`` 里的 ``veh_arm`` 包 (NERO 的厂商 URDF 解析、刚体动力学、
雅可比、载具 6-DOF 运动引擎都在那里)。

解析顺序
------------------------------------------------------------------
1. 已在 ``sys.path`` / 已安装的 ``veh_arm``
2. 环境变量 ``VLA_VEHICLE_UPSTREAM`` 指向的上游仓库根目录
3. 本仓库的兄弟目录 ``../robotic_arm_on_vehicle``
4. ``third_party/robotic_arm_on_vehicle`` (由 ``scripts/fetch_upstream.py`` 拉取)

任何一步成功即在 ``sys.path`` 前面插入该根目录, 之后 ``import veh_arm`` 直接可用。
全部失败时抛 ``MissingUpstreamError`` 并给出可执行的修复命令 —— 不做静默降级,
因为"用一个玩具臂模型顶替"会让所有量化结论失真。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REPO_NAME = "robotic_arm_on_vehicle"

class MissingUpstreamError(RuntimeError):
    """找不到上游控制库。"""


def _candidate_roots() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("VLA_VEHICLE_UPSTREAM")
    if env:
        out.append(Path(env).expanduser())
    out.append(REPO_ROOT.parent / UPSTREAM_REPO_NAME)
    out.append(REPO_ROOT / "third_party" / UPSTREAM_REPO_NAME)
    # 去重且保序
    seen, uniq = set(), []
    for p in out:
        rp = p.resolve() if p.exists() else p
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def _importable() -> bool:
    try:
        import veh_arm  # noqa: F401
        return True
    except Exception:
        return False


def ensure_upstream(verbose: bool = False) -> str:
    """确保 ``veh_arm`` 可导入, 返回其所在根目录。"""
    if _importable():
        return str(Path(sys.modules["veh_arm"].__file__).resolve().parents[1])
    for root in _candidate_roots():
        if (root / "veh_arm" / "__init__.py").is_file():
            sys.path.insert(0, str(root))
            if _importable():
                if verbose:
                    print(f"[deps] 复用上游控制核心: {root}")
                return str(root)
            sys.path.remove(str(root))
    tried = "\n".join(f"  - {p}" for p in _candidate_roots())
    raise MissingUpstreamError(
        "未找到上游控制库 veh_arm (robotic_arm_on_vehicle)。\n"
        "本项目复用它的机械臂模型与载具运动引擎, 不做本地副本。修复方式任选其一:\n"
        f"  1) export VLA_VEHICLE_UPSTREAM=/path/to/robotic_arm_on_vehicle\n"
        f"  2) 把仓库克隆到兄弟目录: git clone <url> {REPO_ROOT.parent / UPSTREAM_REPO_NAME}\n"
        f"  3) 运行 python scripts/fetch_upstream.py (拉到 third_party/)\n"
        f"已尝试:\n{tried}"
    )


def veh_arm_root() -> str:
    return ensure_upstream()
