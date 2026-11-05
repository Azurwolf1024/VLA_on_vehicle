"""训练用 Dataset: 把回合拆成 (单步观测 -> 单步动作) 的样本。

为什么是"单步"而不是"序列"
------------------------------------------------------------------
第一版目标是**把链路跑通并得到可复现的基线**, 单步行为克隆 (frame-level BC) 的训练/评估
最简单, 也最容易定位问题（动作误差可以直接对着阶段看）。
序列模型 (ACT / Diffusion Policy / VLA) 只需要换 ``collate`` 与模型, 数据格式不用动 ——
回合里已经存了完整时间序列。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schema import Episode, list_episodes
from .preproc import resize_frame


def _to_torch():
    import torch  # 延迟导入: 只采集数据时不装 torch 也能跑
    return torch


class EpisodeDataset:
    """所有回合一次性载入内存（第一版数据量在 GB 以下）。"""

    def __init__(self, root: str | Path, only_success: bool = True, stride: int = 1,
                 image_size: Optional[int] = None, episodes: Optional[Sequence[str]] = None,
                 chunk: int = 1):
        self.root = Path(root)
        paths = [Path(p) for p in episodes] if episodes else list_episodes(self.root)
        if not paths:
            raise FileNotFoundError(f"{self.root} 下没有 .npz 回合; 先跑 scripts/record_demos.py")
        self.episodes: List[Episode] = []
        for p in paths:
            ep = Episode.load(p)
            if only_success and not ep.success:
                continue
            self.episodes.append(ep)
        if not self.episodes:
            raise FileNotFoundError("过滤后没有可用回合（all failed, only_success=True）")
        self.image_size = image_size
        self.stride = max(1, int(stride))
        self.chunk = max(1, int(chunk))
        # 建立 (回合, 帧) 索引
        self.index: List[Tuple[int, int]] = []
        for i, ep in enumerate(self.episodes):
            self.index.extend((i, t) for t in range(0, ep.length, self.stride))

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.index)

    def split(self, val_ratio: float, seed: int = 0):
        """按**回合**划分, 避免同一回合的相邻帧同时进训练和验证造成泄漏。"""
        n = len(self.episodes)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
        val = set(perm[:n_val].tolist())
        tr_idx = [(i, t) for (i, t) in self.index if i not in val]
        va_idx = [(i, t) for (i, t) in self.index if i in val]
        return _Subset(self, tr_idx), _Subset(self, va_idx)

    # ------------------------------------------------------------------ #
    def _frames(self, ep: Episode, t: int) -> Dict[str, np.ndarray]:
        g = ep.global_rgb[t]
        w = ep.wrist_rgb[t]
        if self.image_size:
            g = resize_frame(g, self.image_size)
            w = resize_frame(w, self.image_size)
        # action 可以是**未来 K 步**的动作序列（action chunk）。
        # chunk=1 时与单步 BC 完全一致（保持向后兼容）；
        # 末尾不足 K 步用最后一步补齐 —— 回合结束时的"停在原地"动作本来就是这个语义。
        a = ep.action[t:t + self.chunk]
        if a.shape[0] < self.chunk:
            pad = np.repeat(ep.action[-1:], self.chunk - a.shape[0], axis=0)
            a = np.concatenate([a, pad], axis=0)
        return dict(
            global_rgb=g, wrist_rgb=w, q=ep.q[t], dq=ep.dq[t], tcp=ep.tcp[t],
            grip=np.float32(ep.grip[t]), tokens=ep.tokens,
            action=a.astype(np.float32),
        )

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        ei, t = self.index[i]
        return self._frames(self.episodes[ei], t)

    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, np.ndarray]:
        q = np.concatenate([ep.q for ep in self.episodes])
        dq = np.concatenate([ep.dq for ep in self.episodes])
        tcp = np.concatenate([ep.tcp for ep in self.episodes])
        act = np.concatenate([ep.action for ep in self.episodes])
        out = {}
        for name, arr in (("q", q), ("dq", dq), ("tcp", tcp), ("action", act)):
            out[f"{name}_mean"] = arr.mean(0).astype(np.float32)
            out[f"{name}_std"] = (arr.std(0) + 1e-6).astype(np.float32)
        return out

    def summary(self) -> dict:
        eps = self.episodes
        return dict(n_episodes=len(eps), n_frames=len(self.index),
                    n_success=sum(e.success for e in eps),
                    instr=len({e.instruction for e in eps}),
                    length_mean=float(np.mean([e.length for e in eps])),
                    img_hw=tuple(eps[0].global_rgb.shape[1:3]))


class _Subset:
    """轻量子集（避免依赖 torch.utils.data.Subset 的语义差异）。"""

    def __init__(self, ds: EpisodeDataset, idx: List[Tuple[int, int]]):
        self.ds = ds
        self.idx = idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        ei, t = self.idx[i]
        return self.ds._frames(self.ds.episodes[ei], t)


def make_loader(ds, batch_size: int, shuffle: bool, device: str = "cpu",
                num_workers: int = 0):
    """把 dict 样本拼成 batch 并转到 torch。"""
    torch = _to_torch()

    def collate(batch: List[Dict[str, np.ndarray]]):
        import torch as th
        out = {}
        # HWC(uint8) -> NCHW(float, 0..1): 卷积要 channel-first
        out["global_rgb"] = th.from_numpy(
            np.stack([b["global_rgb"] for b in batch]).astype(np.float32) / 255.0
        ).permute(0, 3, 1, 2).contiguous()
        out["wrist_rgb"] = th.from_numpy(
            np.stack([b["wrist_rgb"] for b in batch]).astype(np.float32) / 255.0
        ).permute(0, 3, 1, 2).contiguous()
        for k in ("q", "dq", "tcp"):
            out[k] = th.from_numpy(np.stack([b[k] for b in batch]).astype(np.float32))
        out["grip"] = th.from_numpy(np.stack([b["grip"] for b in batch]).astype(np.float32))
        out["tokens"] = th.from_numpy(np.stack([b["tokens"] for b in batch]).astype(np.int64))
        out["action"] = th.from_numpy(np.stack([b["action"] for b in batch]).astype(np.float32))
        return out

    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=num_workers, collate_fn=collate)
