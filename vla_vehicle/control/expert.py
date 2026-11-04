"""脚本专家 (privileged): 用真值物体位姿生成"抓 -> 抬 -> 送到垃圾舱 -> 松手"的示范。

定位
------------------------------------------------------------------
它**不是**要交付的策略, 而是数据引擎: 行为克隆需要"能干活的老师"。
它读的是甲板系下的真值物体位姿 (仿真才有的特权信息), 输出与策略**完全相同的动作格式**
(4 维: TCP 位移 3 + 夹爪 1), 因此录下来的 (观测, 动作) 可以直接喂给任何策略。

为什么用"分阶段 + 位移限幅"而不是轨迹插值
------------------------------------------------------------------
决策层是 10 Hz, 每步最多走 ``action_scale``(2 cm)。专家每步只做一件事:
朝当前阶段的目标点走一个限幅步长, 到位就切下一阶段。这样
* 标注天然是"闭环"的 (目标点每步按实测物体位置刷新, 船在晃也跟得上);
* 每条示范的动作幅值分布与策略推理时一致, 不会出现训练/推理的尺度错配。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ExpertConfig:
    hover: float = 0.09          # 抓取点上方的悬停高度（甲板系, m）
    grasp_below_top: float = 0.045   # 指尖落在物体顶面以下多少（决定夹在哪一段）
    grasp_min_z: float = 0.004       # 指尖最低高度（别磕甲板）
    phase_timeout: int = 26      # 单个阶段最长决策步数（超时即放弃该条示范）
    lift: float = 0.13           # 抬起高度
    above_bin: float = 0.16      # 垃圾舱上方的运输高度
    release_z: float = 0.05      # 松手时指尖高度
    close_steps: int = 4         # 闭合夹爪后保持的决策步数
    open_steps: int = 4          # 张开夹爪后保持的决策步数
    reach_tol: float = 0.006     # 判定"到位"的容差


PHASES = ["hover", "descend", "close", "lift", "to_bin", "descend_bin",
          "open", "retreat", "done"]


class ScriptedExpert:
    """脚本专家。对外同时满足两种用法:

    * 作为 :class:`~vla_vehicle.policies.base.Policy`: ``reset(instruction)`` 只清内部状态;
    * 作为数据引擎: :meth:`bind` 换一个环境继续用同一个对象。
    """

    def __init__(self, env, cfg: Optional[ExpertConfig] = None):
        self.cfg = cfg or ExpertConfig()
        self.env = env
        self._reset_state()

    def bind(self, env) -> "ScriptedExpert":
        self.env = env
        self._reset_state()
        return self

    def reset(self, instruction: str = "") -> None:
        """Policy 接口: 回合开始时清状态（专家不需要指令, 只用真值位姿）。"""
        self._reset_state()

    # ------------------------------------------------------------------ #
    def _reset_state(self) -> None:
        self.i = 0
        self.phase = PHASES[0]
        self.timer = 0
        self.hold: Optional[np.ndarray] = None
        self.done = False
        self.stalled = False

    @property
    def action_scale(self) -> float:
        return float(self.env.cfg.action_scale)

    # ------------------------------------------------------------------ #
    def grasp_z(self) -> float:
        """抓取时指尖的高度。

        由**物体顶面**反推, 不要写死: 指尖到掌部下沿只有 61 mm, 若从甲板面起抓,
        掌部会压在罐子顶上, 表现为"下不去、也夹不住"。落在顶面以下 45 mm 则
        指垫完整包住物体上半段, 掌部始终在顶面之上。
        """
        st = self.env.object_states()[self.env.target_index]
        top = float(st["p"][2]) + st["item"].shape.half_height
        return max(self.cfg.grasp_min_z, top - self.cfg.grasp_below_top)

    def _targets(self):
        """当前阶段的 (目标点, 夹爪指令)。"""
        env, cfg = self.env, self.cfg
        sc = env.cfg.scene
        obj = env.object_states()[env.target_index]["p"]
        gz = self.grasp_z()
        bx, by = sc.bin_center
        p_tcp = env.tcp_pose()[1]
        ph = self.phase
        if ph == "hover":
            return np.array([obj[0], obj[1], gz + cfg.hover]), 1.0
        if ph == "descend":
            return np.array([obj[0], obj[1], gz]), 1.0
        if ph == "close":
            return (self.hold if self.hold is not None else p_tcp), 0.0
        if ph == "lift":
            return np.array([obj[0], obj[1], gz + cfg.lift]), 0.0
        if ph == "to_bin":
            return np.array([bx, by, cfg.above_bin]), 0.0
        if ph == "descend_bin":
            return np.array([bx, by, cfg.release_z]), 0.0
        if ph == "open":
            return (self.hold if self.hold is not None else np.array([bx, by, cfg.release_z])), 1.0
        if ph == "retreat":
            return np.array([bx, by, cfg.above_bin]), 1.0
        return p_tcp, 1.0

    # ------------------------------------------------------------------ #
    def _advance_phase(self) -> None:
        self.i = min(self.i + 1, len(PHASES) - 1)
        self.phase = PHASES[self.i]
        self.timer = 0
        self.hold = None
        if self.phase == "done":
            self.done = True

    def act(self, obs) -> np.ndarray:
        cfg = self.cfg
        p_tcp = np.asarray(obs.tcp, float)
        target, grip = self._targets()
        target = np.asarray(target, float)

        delta = np.clip(target - p_tcp, -self.action_scale, self.action_scale)
        err = float(np.linalg.norm(target - p_tcp))

        # 阶段推进
        self.timer += 1
        if self.phase in ("close", "open"):
            if self.phase == "close" and self.hold is None:
                self.hold = p_tcp.copy()
            need = cfg.close_steps if self.phase == "close" else cfg.open_steps
            if self.timer >= need:
                self._advance_phase()
        elif self.phase not in ("done",):
            if err < cfg.reach_tol or self.timer >= cfg.phase_timeout:
                # 超时也推进: 卡住的阶段通常意味着抓取失败, 让回合正常收尾,
                # 由 env.evaluate() 判失败, 而不是把这步耗在死等上。
                if self.timer >= cfg.phase_timeout and self.phase not in ("hover",):
                    self.stalled = True
                self._advance_phase()

        action = np.zeros(4)
        action[:3] = delta / self.action_scale
        action[3] = float(grip)
        # done 阶段: 停在原地, 保持张开
        if self.phase == "done":
            action[:3] = 0.0
            action[3] = 1.0
        return action

    # ------------------------------------------------------------------ #
    def info(self) -> dict:
        return dict(phase=self.phase, phase_index=self.i, done=self.done,
                    stalled=self.stalled)
