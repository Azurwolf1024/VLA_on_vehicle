"""上游 ``veh_arm.robot.urdf_io`` 的惯量解析缺陷修补。

问题
------------------------------------------------------------------
URDF 规范里 ``<inertia>`` 的六个分量是 **XML 属性**::

    <inertia ixx="0.00066" ixy="2.7e-06" ixz="1.4e-05"
             iyy="0.00054" iyz="1.8e-06" izz="0.00051"/>

而 ``urdf_io._inertia_from_urdf`` 用 ``el.find("ixx")`` 按**子元素**读取
(``<ixx>0.00066</ixx>`` 这种非标准写法)。对 NERO 的厂商 URDF 而言, 结果不是报错,
而是**静默返回全零惯量** —— 质量仍在、转动惯量全丢。运动学 (FK/Jacobian) 完全不受影响,
所以上游所有跟踪精度的结论依然成立, 但任何依赖 ``M(q)`` 的动力学量都会失真。

对本项目的影响
------------------------------------------------------------------
MuJoCo 直接拒绝非正定惯量 (``inertia must have positive eigenvalues``),
而且抓取的接触/摩擦动力学本来就要求真实的转动惯量。因此这里重读一遍 URDF,
把惯量补回 ``ArmSpec``, 其余字段 (偏置/轴向/限位/质量/质心) 全部沿用上游解析结果。

上游若修好该函数, 本模块自动退化为空操作 (检测到惯量已非零即跳过)。
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET

import numpy as np

from ..deps import ensure_upstream

ensure_upstream()

from veh_arm.core.lie import rpy_to_rot  # noqa: E402
from veh_arm.robot.model import ArmSpec  # noqa: E402
from veh_arm.robot.panda import attach_payload  # noqa: E402
from veh_arm.robot.urdf_io import _parse_urdf, spec_from_urdf  # noqa: E402


def _f(el, key, default=0.0) -> float:
    v = el.get(key)
    return float(v) if v is not None else default


def read_link_inertials(urdf_path: str) -> dict:
    """``link 名 -> (质心, 关于质心的惯量矩阵, 质量)``, 惯量按标准属性读取。"""
    root = ET.parse(urdf_path).getroot()
    out = {}
    for le in root.findall("link"):
        name = le.get("name")
        iner = le.find("inertial")
        if iner is None:
            out[name] = (np.zeros(3), np.zeros((3, 3)), 0.0)
            continue
        om = iner.find("origin")
        xyz = np.zeros(3) if om is None else np.array(
            [float(x) for x in (om.get("xyz") or "0 0 0").split()])
        rpy = np.zeros(3) if om is None else np.array(
            [float(x) for x in (om.get("rpy") or "0 0 0").split()])
        ie = iner.find("inertia")
        if ie is None:
            I = np.zeros((3, 3))
        else:
            ixx, iyy, izz = _f(ie, "ixx"), _f(ie, "iyy"), _f(ie, "izz")
            ixy, ixz, iyz = _f(ie, "ixy"), _f(ie, "ixz"), _f(ie, "iyz")
            I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        R = rpy_to_rot(rpy)
        mass_el = iner.find("mass")
        mass = float(mass_el.get("value")) if mass_el is not None else 0.0
        out[name] = (xyz, R @ I @ R.T, mass)
    return out


def chain_link_names(urdf_path: str, ee_link: str | None = None) -> list:
    """按上游 ``spec_from_urdf`` 的同一遍历顺序列出连杆名（用于对齐下标）。"""
    links, joints = _parse_urdf(urdf_path)
    base = None
    for j in joints:
        if j["type"] == "fixed" and (j["parent"] in ("world", "", None)
                                     or j["parent"] not in links):
            base = j["child"]
            break
    if base is None:
        children = {j["child"] for j in joints}
        base = next(nm for nm in links if nm not in children)
    names, cur = [], base
    while True:
        cands = [j for j in joints if j["parent"] == cur
                 and j["type"] in ("revolute", "continuous")]
        if not cands:
            break
        j = cands[0]
        names.append(j["child"])
        cur = j["child"]
        if ee_link is not None and j["child"] == ee_link:
            break
    return names


def safe_inertia(I: np.ndarray, mass: float, name: str = "") -> np.ndarray:
    """把惯量投影到"正定 + 三角不等式"的合法域 (MuJoCo 要求)。

    真实 URDF 一般都合法; 这里只是兜底, 一旦触发会告警 —— 触发本身就说明模型有问题。
    """
    I = np.asarray(I, float)
    I = 0.5 * (I + I.T)
    w, V = np.linalg.eigh(I)
    eps = max(1e-9, 1e-8 * float(mass))
    if np.any(w <= 0.0):
        warnings.warn(f"连杆 {name!r} 惯量非正定 (min eig {w.min():.2e}), 已投影修正")
    w = np.maximum(w, eps)
    # 三角不等式: 任一主惯量 <= 另外两个之和
    for _ in range(4):
        s = w.sum()
        over = w > (0.5 * s) * 1.0001
        if not over.any():
            break
        w = np.where(over, 0.499 * s, w)
    return V @ np.diag(w) @ V.T


def load_nero_spec(urdf_path: str, ee_link: str = "link7", tool_mass: float = 0.5,
                   payload_mass: float = 0.0, payload_offset=None,
                   dq_max=None, tau_max=None) -> ArmSpec:
    """``make_nero`` 的等价实现, 但惯量取自 URDF 属性 (见模块文档)。"""
    spec = spec_from_urdf(urdf_path, ee_link=ee_link, tool_mass=0.0, payload_mass=0.0)
    names = chain_link_names(urdf_path, ee_link=ee_link)
    if len(names) != spec.n:
        warnings.warn(f"URDF 链长 ({len(names)}) 与 ArmSpec.n ({spec.n}) 不一致, 跳过惯量修补")
    else:
        inert = read_link_inertials(urdf_path)
        for i, nm in enumerate(names):
            com, I, mass = inert.get(nm, (np.zeros(3), np.zeros((3, 3)), 0.0))
            L = spec.links[i]
            if not np.allclose(I, 0.0):
                L.inertia = safe_inertia(I, float(L.mass), nm)
            if float(L.mass) <= 0.0 and mass > 0.0:
                L.mass = mass
                L.com = com
    if tool_mass > 0.0:
        spec.links[-1] = attach_payload(spec.links[-1], np.zeros(3), tool_mass,
                                        np.diag([1e-3] * 3))
    if payload_mass > 0.0:
        po = np.zeros(3) if payload_offset is None else np.asarray(payload_offset, float)
        spec.links[-1] = attach_payload(spec.links[-1], po, payload_mass,
                                        np.diag([2e-4] * 3))
    if dq_max is not None:
        spec.dq_max = np.asarray(dq_max, float).reshape(spec.n)
    if tau_max is not None:
        spec.tau_max = np.asarray(tau_max, float).reshape(spec.n)
    return spec
