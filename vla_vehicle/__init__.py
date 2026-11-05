"""VLA on vehicle —— 载具（船/车）上的视觉-语言-动作抓取框架。

分层
------------------------------------------------------------------
``robot``    机械臂模型（复用上游 veh_arm 的 NERO 模型）+ MJCF 生成
``sim``      MuJoCo 场景 / 双相机 / 载具运动驱动的抓放环境
``control``  DLS 逆解内环 + 脚本专家（采集演示用）
``data``     回合数据结构 / 录制器 / 训练用 Dataset
``policies`` 策略接口（BC 基线 + 预训练 VLA 适配位）
"""

from .deps import ensure_upstream

__version__ = "0.2.0"

__all__ = ["ensure_upstream", "__version__"]
