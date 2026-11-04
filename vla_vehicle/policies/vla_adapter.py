"""预训练 VLA 的接入位（当前**未实现**, 只有契约与接线说明）。

为什么单独一个文件
------------------------------------------------------------------
OpenVLA / π0 / SmolVLA 这类模型各自的输入格式、动作空间、归一化都不同, 但它们对
本项目的接口是同一件事: **看到 (双相机图像 + 指令 + 本体感觉), 输出同一套 4 维动作**。
把差异关在这一个类里, 数据和环境就不必为每个模型改一遍。

接入需要做的三件事
------------------------------------------------------------------
1. ``prepare(obs)``: 把 ``Obs`` 变成该模型的输入（通常是 PIL 图像 + 一句 prompt,
   少数模型还要 proprio 张量）。见 :meth:`PretrainedVLA.prompt_for`。
2. ``infer(batch)``: 调模型前向, 拿到**末端增量**（不是关节角、不是绝对位姿）。
   若模型原生输出 6-DoF 位姿或关节角, 在这里换算成甲板系 TCP 增量。
3. ``postprocess(raw)``: 归一化到 ``[-1,1]^3`` + 夹爪 0/1, 并对齐本项目的控制频率
   (预训练 VLA 常以 3–10 Hz 出 chunk, 需要 action chunking + 时间对齐)。

已知的坑（设计文档里也有）
------------------------------------------------------------------
* **动作空间不一致**: OpenVLA 输出 7 维 EE delta(含旋转)+gripper, 本项目 v1 只吃 3 维
  平移 + 1 维夹爪。直接对接会丢掉旋转自由度, 抓方块类物体够用, 抓细长物体不够。
* **相机口径不一致**: 预训练数据多为单臂单目/腕部近景。我们喂"全局 + 腕部"两张图时,
  要么做拼图, 要么改模型的视觉塔 —— 这是精度损失的主要来源。
* **语言**: 本项目用字符级 token（中文/英文混）, 与 VLA 自带的分词器不通用,
  必须在这里重新分词, 不能用 ``obs.tokens``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ACTION_DIM, Policy


@dataclass
class VLAConfig:
    backend: str = "openvla"          # openvla | smolvla | pi0 | custom
    checkpoint: str = ""
    device: str = "cuda"
    image_size: int = 224
    action_chunk: int = 1
    max_new_tokens: int = 16


class PretrainedVLA(Policy):
    """预训练 VLA 适配基类。子类只需实现 :meth:`infer`。"""

    name = "vla"

    def __init__(self, cfg: VLAConfig | None = None):
        self.cfg = cfg or VLAConfig()
        self.model = None
        self.processor = None
        self._queue: list = []

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        raise NotImplementedError(
            "预训练 VLA 尚未接入。请在子类里实现 load()/infer(), "
            "或先用行为克隆基线跑通链路: python scripts/train_bc.py")

    def prompt_for(self, instruction: str) -> str:
        """构造给 VLA 的指令文本（含动作空间约定）。"""
        return (f"{instruction}\n"
                "Action: end-effector delta position in the deck frame, "
                "3 values in [-1,1], plus a binary gripper command.")

    def prepare(self, obs) -> dict:
        return dict(instruction=self.prompt_for(obs.instruction),
                    global_image=obs.global_rgb, wrist_image=obs.wrist_rgb,
                    proprio=np.concatenate([obs.q, obs.dq, obs.tcp, [obs.grip]]))

    def infer(self, batch: dict) -> np.ndarray:
        raise NotImplementedError("子类必须实现 infer(batch) -> (H, 4) 动作块")

    def postprocess(self, raw: np.ndarray) -> np.ndarray:
        a = np.asarray(raw, dtype=np.float32).reshape(-1)[:ACTION_DIM]
        out = np.zeros(ACTION_DIM, np.float32)
        out[:3] = np.clip(a[:3], -1.0, 1.0)
        out[3] = 1.0 if (a.size < 4 or a[3] >= 0.5) else 0.0
        return out

    # ------------------------------------------------------------------ #
    def reset(self, instruction: str = "") -> None:
        self._queue.clear()

    def act(self, obs) -> np.ndarray:
        if not self._queue:
            self._queue = [self.postprocess(a)
                           for a in np.atleast_2d(self.infer(self.prepare(obs)))]
        return self._queue.pop(0)
