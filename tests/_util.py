"""测试共用的小工具。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_rig():
    """返回 (Robot, ArmSpec, ArmMJCF, MjModel, MjData) —— 只含机械臂的最小场景。"""
    import mujoco

    from vla_vehicle.robot.arm import READY_Q, GripperSpec, load_arm
    from vla_vehicle.robot.mjcf import build_arm, solo_xml

    robot, spec, _ = load_arm()
    arm = build_arm(spec, gripper=GripperSpec(), q_ref=READY_Q, name="nero")
    model = mujoco.MjModel.from_xml_string(solo_xml(arm, spec))
    data = mujoco.MjData(model)
    return robot, spec, arm, model, data


def build_min_scene(**env_kw):
    """构造一个关掉渲染的环境（测试里不需要 GL）。"""
    from vla_vehicle.sim.env import EnvConfig, PickPlaceEnv

    cfg = EnvConfig(render=False)
    for k, v in env_kw.items():
        setattr(cfg, k, v)
    return PickPlaceEnv(cfg)
