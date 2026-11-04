"""控制层。

两种内环, 语义相同（TCP 目标进, 关节动作出）, 实现与代价不同:

* :class:`~vla_vehicle.control.ik.DLSIK` —— 关节位置伺服 + 逆解, 便宜（采集默认）;
* :class:`~vla_vehicle.control.tsid.TSIDAdapter` —— 上游解析动力学 + TSID 力矩环,
  能真正对基座运动做前馈, 但每步解一次 QP（约 4 倍慢）。
"""

from .ik import DLSIK, IKSolution
from .tsid import TSIDAdapter, TSIDConfig

__all__ = ["DLSIK", "IKSolution", "TSIDAdapter", "TSIDConfig"]
