"""配置加载: YAML -> dataclass。

只做"显式字段覆盖", 不引入 hydra/omegaconf —— 配置项就是 dataclass 的字段,
写错键名会直接报错, 而不是静默忽略（后者在多组对照实验里是最贵的 bug）。
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from .sim.env import EnvConfig
from .sim.scene import SceneConfig


def _apply(obj, data: Dict[str, Any], path: str = "") -> None:
    known = {f.name: f for f in fields(obj)}
    for k, v in (data or {}).items():
        if k not in known:
            raise KeyError(f"配置项 {path}{k!r} 不存在; 可用: {sorted(known)}")
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _apply(cur, v, f"{path}{k}.")
        else:
            setattr(obj, k, v)


def load_env_config(path: str | Path | None = None, **overrides) -> EnvConfig:
    """从 YAML 读 ``env:`` / ``scene:`` 两段, 再叠命令行覆盖。"""
    cfg = EnvConfig(scene=SceneConfig())
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        env_part = raw.get("env", {})
        scene_part = raw.get("scene", {})
        if scene_part:
            _apply(cfg.scene, scene_part, "scene.")
        if env_part:
            _apply(cfg, env_part, "env.")
        unknown = set(raw) - {"env", "scene", "train", "eval"}
        if unknown:
            raise KeyError(f"{path}: 未知顶层小节 {sorted(unknown)}")
    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise KeyError(f"未知覆盖项 {k!r}")
        setattr(cfg, k, v)
    return cfg


@dataclass
class TrainConfig:
    data: str = "data/demos"
    out: str = "runs/bc"
    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    image_size: int = 96          # 训练时缩放到的边长(不裁剪, 见 data/preproc.py)
    chunk: int = 1                # action chunk: 一次预测未来 K 步。1 = 单步 BC
    only_success: bool = True
    stride: int = 1
    val_ratio: float = 0.15
    seed: int = 0
    num_workers: int = 0
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "TrainConfig":
        cfg = cls()
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            _apply(cfg, raw.get("train", {}), "train.")
        return cfg
