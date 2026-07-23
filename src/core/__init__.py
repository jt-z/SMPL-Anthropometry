"""
Core measurement functionality for SMPL body models.
"""

# 不在 __init__.py 中自动导入，避免循环依赖
# 用户需要显式导入，例如：
# from src.core.measure import MeasureBody
# from src.core.measurement_definitions import STANDARD_LABELS

__all__ = [
    'MeasureBody',
    'STANDARD_LABELS',
    'SMPLMeasurementDefinitions',
    'SMPLXMeasurementDefinitions',
]
