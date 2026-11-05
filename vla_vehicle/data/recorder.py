"""回合录制器: 把专家/策略跑出来的 (观测, 动作) 攒成 :class:`Episode`。

设计要点
------------------------------------------------------------------
* **只在决策步采样** (10 Hz), 不存内环 100 Hz 的数据 —— VLA 的时间尺度就是决策步;
* 每步同时记 ``obj_track`` (特权真值), 但明确标注"不给策略" —— 训练集里带着它,
  评测时才能算"到底有没有放进桶里", 而不必重放仿真;
* 记录 ``phase`` (专家阶段名), 便于按阶段切片做诊断 / 分层训练。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .schema import Episode


class Recorder:
    def __init__(self, env, instruction: str, tokens: np.ndarray, seed: int = 0):
        self.env = env
        self.instruction = instruction
        self.tokens = np.asarray(tokens, np.int16)
        self.seed = int(seed)
        self._img: List[np.ndarray] = []
        self._wimg: List[np.ndarray] = []
        self._q: List[np.ndarray] = []
        self._dq: List[np.ndarray] = []
        self._tcp: List[np.ndarray] = []
        self._grip: List[float] = []
        self._act: List[np.ndarray] = []
        self._obj: List[np.ndarray] = []
        self._phase: List[str] = []

    # ------------------------------------------------------------------ #
    def add(self, obs, action: np.ndarray, phase: str = "") -> None:
        self._img.append(obs.global_rgb)
        self._wimg.append(obs.wrist_rgb)
        self._q.append(np.asarray(obs.q, np.float32))
        self._dq.append(np.asarray(obs.dq, np.float32))
        self._tcp.append(np.asarray(obs.tcp, np.float32))
        self._grip.append(float(obs.grip))
        self._act.append(np.asarray(action, np.float32).reshape(4))
        self._obj.append(np.stack([s["p"] for s in self.env.object_states()]).astype(np.float32))
        self._phase.append(phase)

    # ------------------------------------------------------------------ #
    def finish(self, success: bool, target_index: int, extra: Optional[dict] = None) -> Episode:
        meta = dict(extra or {})
        meta.update(env=dict(
            action_scale=float(self.env.cfg.action_scale),
            dt_decision=float(self.env.cfg.dt_decision),
            vehicle_kind=self.env.cfg.vehicle_kind,
            vehicle_scale=float(self.env.cfg.vehicle_scale),
            lang=self.env.cfg.lang,
            img_hw=[int(self.env.cfg.scene.img_height), int(self.env.cfg.scene.img_width)],
            bin_center=[float(v) for v in self.env.cfg.scene.bin_center],
            phase_0=self._phase[0] if self._phase else "",
        ))
        return Episode(
            global_rgb=np.stack(self._img),
            wrist_rgb=np.stack(self._wimg),
            q=np.stack(self._q), dq=np.stack(self._dq),
            tcp=np.stack(self._tcp), grip=np.asarray(self._grip, np.float32),
            action=np.stack(self._act), obj_track=np.stack(self._obj),
            tokens=self.tokens,
            instruction=self.instruction, success=bool(success),
            target_index=int(target_index), seed=self.seed,
            phase=list(self._phase), meta=meta,
        )

    def __len__(self) -> int:
        return len(self._act)
