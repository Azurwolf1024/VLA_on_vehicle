"""回合数据结构与落盘格式。

文件格式: ``.npz`` + 一段 JSON meta（``np.savez_compressed``）。

为什么不用 HDF5
------------------------------------------------------------------
单条回合只有几十 MB, 压缩 npz 的读写速度、零依赖、可被 ``numpy`` 直接打开这三点
比 HDF5 更划算; 需要多文件索引时用目录 + ``index.jsonl``。

字段约定
------------------------------------------------------------------
* 图像 ``uint8``: 省 4 倍空间, 训练时 ``/255`` 即可;
* 动作 ``float32 (T,4)``: ``[dx,dy,dz,g]``, 前三维是**归一化位移**
  (乘 ``action_scale`` 得米), 第四维是夹爪 0=闭 1=开;
* ``tokens int16 (L,)``: 指令 token, 每条回合一份（同一条回合内不变）;
* ``obj_track float32 (T,N,3)``: 物体在**甲板系**的位置, 仅用于评测/可视化, **不给策略**。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

FORMAT_VERSION = 1


@dataclass
class Episode:
    # ---- 序列 ----
    global_rgb: np.ndarray          # (T,H,W,3) uint8
    wrist_rgb: np.ndarray           # (T,H,W,3) uint8
    q: np.ndarray                   # (T,n) float32
    dq: np.ndarray                  # (T,n) float32
    tcp: np.ndarray                 # (T,3) 甲板系
    grip: np.ndarray                # (T,) 归一化开口
    action: np.ndarray              # (T,4)
    obj_track: np.ndarray           # (T,N,3) 甲板系
    tokens: np.ndarray              # (L,) int16
    # ---- 单值 ----
    instruction: str = ""
    success: bool = False
    target_index: int = 0
    seed: int = 0
    phase: Optional[List[str]] = None
    meta: Dict = field(default_factory=dict)
    version: int = FORMAT_VERSION

    @property
    def length(self) -> int:
        return int(self.action.shape[0])

    def to_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta)
        meta.update(version=self.version, instruction=self.instruction,
                    success=bool(self.success), target_index=int(self.target_index),
                    seed=int(self.seed), phase=self.phase)
        np.savez_compressed(
            path,
            global_rgb=self.global_rgb.astype(np.uint8),
            wrist_rgb=self.wrist_rgb.astype(np.uint8),
            q=self.q.astype(np.float32), dq=self.dq.astype(np.float32),
            tcp=self.tcp.astype(np.float32), grip=self.grip.astype(np.float32),
            action=self.action.astype(np.float32),
            obj_track=self.obj_track.astype(np.float32),
            tokens=self.tokens.astype(np.int16),
            meta=np.array(json.dumps(meta, ensure_ascii=False)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Episode":
        with np.load(str(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            return cls(
                global_rgb=z["global_rgb"], wrist_rgb=z["wrist_rgb"],
                q=z["q"], dq=z["dq"], tcp=z["tcp"], grip=z["grip"],
                action=z["action"], obj_track=z["obj_track"], tokens=z["tokens"],
                instruction=meta.get("instruction", ""),
                success=bool(meta.get("success", False)),
                target_index=int(meta.get("target_index", 0)),
                seed=int(meta.get("seed", 0)),
                phase=meta.get("phase"),
                meta=meta, version=int(meta.get("version", 0)),
            )


def list_episodes(root: str | Path) -> List[Path]:
    return sorted(Path(root).glob("*.npz"))


def episode_summary(path: str | Path) -> dict:
    ep = Episode.load(path)
    return dict(path=str(path), length=ep.length, instruction=ep.instruction,
                success=ep.success, seed=ep.seed,
                shape=tuple(ep.global_rgb.shape[1:3]))
