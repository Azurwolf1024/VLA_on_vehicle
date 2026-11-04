"""可抓取物体（"垃圾"）库与语言指令。

设计原则
------------------------------------------------------------------
1. **只用基本几何体**: 圆柱(易拉罐) / 长方体(纸盒) / 球(瓶盖、浮球)。没有网格依赖,
   新加一类垃圾只改字典; 也更贴近"先跑通再美化"的路线。
2. **颜色 + 形状 = 语言锚点**。指令里同时出现两者, 策略必须真的看图+读句子,
   否则无法在多物体场景里挑对目标 —— 这是本项目最小的"VLA 必要性"。
3. 质量/摩擦/尺寸都在"能被 NERO + 30 N 夹爪稳定抓起"的区间内。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class ShapeSpec:
    key: str                 # 内部名
    geom: str                # mjcf geom 类型
    size: Sequence[float]    # cylinder: (r, half_h) / box: half extents / sphere: (r,)
    mass: float
    zh: str                  # 中文名（用于指令）
    en: str                  # 英文名

    @property
    def half_height(self) -> float:
        """几何中心到最低点的高度（决定初始摆放的 z）。"""
        if self.geom == "cylinder":
            return float(self.size[1])
        if self.geom == "sphere":
            return float(self.size[0])
        return float(self.size[2])

    @property
    def width(self) -> float:
        """夹爪需要张开的宽度 (两指内侧间距)。"""
        if self.geom == "cylinder":
            return 2.0 * float(self.size[0])
        if self.geom == "sphere":
            return 2.0 * float(self.size[0])
        return 2.0 * float(self.size[1])          # box: 沿 y 的半宽

    def to_dict(self) -> dict:
        return dict(key=self.key, geom=self.geom, size=list(map(float, self.size)),
                    mass=self.mass, zh=self.zh, en=self.en)


# 为什么没有球体
# ------------------------------------------------------------------
# 球在**倾斜且晃动的甲板**上会一直滚 (滚动几乎没有静摩擦下限), 实测 1 s 就滚走 12 cm,
# 十几个回合下来必然滚到舷边。这不是策略能解决的问题, 而是任务定义问题。
# 所以第一版只用"有稳定底面"的垃圾: 立着的罐 / 方块 / 扁纸箱 / 细瓶 —— 它们在 1–2° 倾角
# 下静止, 但会被船的**加速度**推动或掀翻, 难度来自动力学而不是拓扑。
SHAPES: Dict[str, ShapeSpec] = {
    "can":    ShapeSpec("can", "cylinder", (0.028, 0.050), 0.08, "易拉罐", "can"),
    "cube":   ShapeSpec("cube", "box", (0.028, 0.028, 0.032), 0.09, "方块", "cube"),
    "carton": ShapeSpec("carton", "box", (0.050, 0.030, 0.020), 0.07, "纸箱", "carton"),
    "bottle": ShapeSpec("bottle", "cylinder", (0.022, 0.062), 0.06, "水瓶", "bottle"),
}


@dataclass(frozen=True)
class ColorSpec:
    key: str
    rgba: Sequence[float]
    zh: str
    en: str


COLORS: Dict[str, ColorSpec] = {
    "red":    ColorSpec("red", (0.82, 0.16, 0.14, 1.0), "红色", "red"),
    "blue":   ColorSpec("blue", (0.16, 0.36, 0.82, 1.0), "蓝色", "blue"),
    "green":  ColorSpec("green", (0.18, 0.68, 0.32, 1.0), "绿色", "green"),
    "yellow": ColorSpec("yellow", (0.92, 0.76, 0.12, 1.0), "黄色", "yellow"),
}


@dataclass
class TrashItem:
    """场景中的一个垃圾实例。"""

    index: int
    shape: ShapeSpec
    color: ColorSpec
    pos: Sequence[float] = (0.0, 0.0, 0.0)   # 世界系初始位置（body 原点 = 几何中心）
    yaw: float = 0.0                          # 绕 z 的初始偏航（仅长方体有意义）

    @property
    def name(self) -> str:
        return f"trash_{self.index}"

    def to_dict(self) -> dict:
        return dict(name=self.name, shape=self.shape.key, color=self.color.key,
                    zh=f"{self.color.zh}{self.shape.zh}",
                    en=f"{self.color.en} {self.shape.en}",
                    pos=[float(x) for x in self.pos], yaw=float(self.yaw))


# --------------------------------------------------------------------------- #
# 语言指令
# --------------------------------------------------------------------------- #
ZH_TEMPLATES: List[str] = [
    "把{color}{shape}捡起来放进垃圾舱",
    "把{color}{shape}丢进垃圾舱",
    "捡起{color}{shape}放进载具的垃圾舱里",
    "把{color}{shape}收拾到垃圾舱",
]
EN_TEMPLATES: List[str] = [
    "pick up the {color} {shape} and put it in the bin",
    "throw the {color} {shape} into the deck bin",
    "grab the {color} {shape} and drop it into the trash hatch",
]


def make_instruction(item: TrashItem, lang: str = "zh", variant: int = 0) -> str:
    """生成一条目标物体的指令（同一物体多种说法, 增加语言多样性）。"""
    if lang == "zh":
        tpl = ZH_TEMPLATES[variant % len(ZH_TEMPLATES)]
        return tpl.format(color=item.color.zh, shape=item.shape.zh)
    tpl = EN_TEMPLATES[variant % len(EN_TEMPLATES)]
    return tpl.format(color=item.color.en, shape=item.shape.en)


# --------------------------------------------------------------------------- #
# 极简字符级分词（不依赖任何分词器/预训练权重）
# --------------------------------------------------------------------------- #
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]


def build_vocab(extra_texts: Sequence[str] = ()) -> List[str]:
    """由模板 + 物体词表收集字符, 固定顺序保证可复现。"""
    chars = {" "}          # 空格也要成 token, 否则英文句子 decode 回来没有分词
    for tpl in ZH_TEMPLATES + EN_TEMPLATES:
        chars.update(tpl)
    for s in list(SHAPES.values()) + list(COLORS.values()):
        chars.update(s.zh)
        chars.update(s.en)
    chars.update("0123456789")
    for t in extra_texts:
        chars.update(str(t))
    return SPECIALS + sorted(chars)


class Tokenizer:
    """字符级分词器: 中文天然按字, 英文按字母, 无需 BPE/词表下载。"""

    def __init__(self, vocab: Sequence[str] | None = None, max_len: int = 48):
        self.vocab = list(vocab) if vocab is not None else build_vocab()
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.max_len = max_len

    @property
    def size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        ids = [self.stoi[BOS]]
        for ch in text:
            ids.append(self.stoi.get(" " if ch.isspace() else ch, self.stoi[UNK]))
        ids.append(self.stoi[EOS])
        ids = ids[: self.max_len]
        ids += [self.stoi[PAD]] * (self.max_len - len(ids))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        out = []
        for i in ids:
            tok = self.vocab[int(i)]
            if tok in (PAD, EOS):
                break
            if tok == BOS:
                continue
            out.append("?" if tok == UNK else tok)
        return "".join(out)

    def state_dict(self) -> dict:
        return dict(vocab=self.vocab, max_len=self.max_len)

    @classmethod
    def from_state(cls, d: dict) -> "Tokenizer":
        return cls(vocab=d["vocab"], max_len=int(d.get("max_len", 48)))


def vocab_fingerprint(vocab: Sequence[str]) -> str:
    """词表指纹 —— 写进回合文件, 训练时校验, 防止词表漂移导致 id 错位。"""
    import hashlib
    h = hashlib.sha1(json.dumps(list(vocab), ensure_ascii=False).encode()).hexdigest()
    return h[:12]
