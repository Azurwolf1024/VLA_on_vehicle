"""感知: 相机几何 + 颜色检测（传统 CV 流水线的上游）。"""

from .camera import PinholeCamera, bow_camera
from .detector import ColorDetector, Detection, parse_target

__all__ = ["PinholeCamera", "bow_camera", "ColorDetector", "Detection", "parse_target"]
