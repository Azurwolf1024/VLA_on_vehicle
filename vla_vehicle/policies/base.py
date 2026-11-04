"""策略接口。

所有策略（脚本专家、行为克隆、未来的预训练 VLA）对**环境**只有这一个契约:

    action = policy.act(obs)      # obs: vla_vehicle.sim.env.Obs, action: (4,) float

动作语义固定为:

    action[0:3] ∈ [-1, 1]   甲板系 TCP 位移增量, 乘 env.cfg.action_scale 得米
    action[3]   ∈ [0, 1]    夹爪 0=闭合 1=张开

把这四维写死是有意的: 换模型(BC/VLA/扩散策略)不会改变环境与数据格式,
换机器人只需要改内环, 而不是重录数据。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

ACTION_DIM = 4


class Policy(ABC):
    """策略基类。``act`` 必须是无状态的（状态放 ``reset``）。"""

    name: str = "policy"

    def reset(self, instruction: str = "") -> None:  # noqa: D401 - 可选钩子
        """回合开始时调用（清缓存 / 重设内部指令）。"""

    @abstractmethod
    def act(self, obs) -> np.ndarray:
        """返回 4 维动作。"""

    def close(self) -> None:
        """释放资源（可选）。"""


class ZeroPolicy(Policy):
    """什么都不做的策略 —— 用于冒烟测试与"无操作"基线。"""

    name = "zero"

    def act(self, obs) -> np.ndarray:
        return np.zeros(ACTION_DIM, dtype=np.float32)


class RandomPolicy(Policy):
    """随机策略 —— 只用于验证环境/评测脚本本身没有把成功判据写反。"""

    name = "random"

    def __init__(self, seed: int = 0, p_grasp: float = 0.35):
        self.rng = np.random.default_rng(seed)
        self.p_grasp = float(p_grasp)

    def act(self, obs) -> np.ndarray:
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        a[:3] = self.rng.uniform(-1, 1, 3)
        a[3] = 1.0 if self.rng.random() < self.p_grasp else 0.0
        return a
