"""行为克隆基线: 双相机 + 指令 token + 本体感觉 -> 4 维动作。

网络刻意做小（~1.5 M 参数, CPU 可训）—— 这一版的目的是**证明链路与数据是通的**,
不是刷分。真上 VLA 时替换 :class:`BCNet` 即可, 数据与环境不用动。

输入拼接策略
------------------------------------------------------------------
全局相机与腕部相机**同一分辨率、同一时刻**, 因此沿通道拼成 6 通道送进同一个卷积干,
比两支独立编码器参数少一半, 也避免了"两个编码器各自漂移"的调试成本。
需要模态特异性时再拆开（见 docs/vla_interface.md 的取舍表）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .base import ACTION_DIM, Policy


def _mask(tokens):
    """pad 位置掩码（0 号 token 约定为 ``<pad>``, 见 sim/objects.build_vocab）。"""
    return (tokens != 0).float()


class BCNet:
    """延迟构造的 torch 模块（导入 torch 很慢, 只在实际训练/推理时才用）。"""

    @staticmethod
    def build(vocab_size: int, act_dim: int = ACTION_DIM, width: int = 32,
              lang_dim: int = 48, prop_dim: int = 64, feat_dim: int = 256,
              image_size: int = 96, drop: float = 0.1):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                c = width
                self.vision = nn.Sequential(
                    nn.Conv2d(6, c, 5, stride=2, padding=2), nn.ReLU(inplace=True),
                    nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(c * 2, c * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(c * 2, c * 3, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((3, 3)),
                )
                self.vis_out = c * 3 * 9
                self.lang = nn.Embedding(vocab_size, lang_dim, padding_idx=0)
                self.prop = nn.Sequential(
                    nn.Linear(7 + 7 + 3 + 1, prop_dim), nn.ReLU(inplace=True),
                    nn.Linear(prop_dim, prop_dim), nn.ReLU(inplace=True))
                self.trunk = nn.Sequential(
                    nn.Linear(self.vis_out + lang_dim + prop_dim, feat_dim),
                    nn.ReLU(inplace=True), nn.Dropout(drop),
                    nn.Linear(feat_dim, feat_dim // 2), nn.ReLU(inplace=True))
                self.pos_head = nn.Linear(feat_dim // 2, 3)
                self.grip_head = nn.Linear(feat_dim // 2, 1)
                self.image_size = image_size

            def forward(self, batch: Dict):
                img = torch.cat([batch["global_rgb"], batch["wrist_rgb"]], dim=1)
                v = self.vision(img).flatten(1)
                tok = batch["tokens"].long()
                emb = self.lang(tok)                          # (B,L,D)
                m = _mask(tok).unsqueeze(-1)
                lang = (emb * m).sum(1) / m.sum(1).clamp_min(1.0)
                prop = torch.cat([batch["q"], batch["dq"], batch["tcp"],
                                  batch["grip"].unsqueeze(-1)], dim=1)
                h = self.trunk(torch.cat([v, lang, self.prop(prop)], dim=1))
                pos = torch.tanh(self.pos_head(h))
                grip_logit = self.grip_head(h).squeeze(-1)
                return pos, grip_logit

        return _Net()


class BCPolicy(Policy):
    """训练好之后用于闭环评测的策略。"""

    name = "bc"

    def __init__(self, net, stats: Dict[str, np.ndarray], tokenizer, device: str = "cpu",
                 image_size: int = 96):
        import torch
        self.torch = torch
        self.net = net.to(device).eval()
        self.device = device
        self.stats = {k: torch.as_tensor(v, device=device) for k, v in stats.items()}
        self.tokenizer = tokenizer
        self.image_size = int(image_size)
        self._cache_tokens: Optional[np.ndarray] = None
        self._instruction = ""

    # ------------------------------------------------------------------ #
    def reset(self, instruction: str = "") -> None:
        self._instruction = instruction
        self._cache_tokens = np.asarray(self.tokenizer.encode(instruction), np.int64)

    def _prep_img(self, img: np.ndarray):
        h, w = img.shape[:2]
        if h != self.image_size:
            step = max(1, h // self.image_size)
            img = img[::step, ::step][: self.image_size, : self.image_size]
        return img

    def act(self, obs) -> np.ndarray:
        t = self.torch
        torch = t
        if self._cache_tokens is None:
            self.reset(obs.instruction)
        q = (torch.as_tensor(obs.q, dtype=t.float32, device=self.device)
             - self.stats["q_mean"]) / self.stats["q_std"]
        dq = torch.as_tensor(obs.dq, dtype=t.float32, device=self.device) \
            / self.stats["dq_std"]
        tcp = (torch.as_tensor(obs.tcp, dtype=t.float32, device=self.device)
               - self.stats["tcp_mean"]) / self.stats["tcp_std"]
        batch = dict(
            global_rgb=t.as_tensor(
                self._prep_img(obs.global_rgb).astype(np.float32)[None] / 255.0,
                device=self.device).permute(0, 3, 1, 2),
            wrist_rgb=t.as_tensor(
                self._prep_img(obs.wrist_rgb).astype(np.float32)[None] / 255.0,
                device=self.device).permute(0, 3, 1, 2),
            q=q[None], dq=dq[None], tcp=tcp[None],
            grip=t.as_tensor([obs.grip], dtype=t.float32, device=self.device),
            tokens=t.as_tensor(self._cache_tokens[None], device=self.device),
        )
        with t.no_grad():
            pos, logit = self.net(batch)
            p = t.sigmoid(logit).item()
        out = np.zeros(ACTION_DIM, dtype=np.float32)
        out[:3] = pos[0].cpu().numpy()
        out[3] = 1.0 if p >= 0.5 else 0.0
        return out

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        payload = dict(state_dict=self.net.state_dict(),
                       stats=self.stats, tokenizer=self.tokenizer.state_dict(),
                       image_size=self.image_size, extra=extra or {})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(payload, str(path))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "BCPolicy":
        import torch
        from ..sim.objects import Tokenizer
        payload = torch.load(str(path), map_location=device, weights_only=False)
        tok = Tokenizer.from_state(payload["tokenizer"])
        net = BCNet.build(vocab_size=tok.size, image_size=payload.get("image_size", 96))
        net.load_state_dict(payload["state_dict"])
        return cls(net, payload["stats"], tok, device=device,
                   image_size=payload.get("image_size", 96))
