"""
SMPL model fitting from various input sources.
"""

# 不在 __init__.py 中自动导入，避免循环依赖
# 用户需要显式导入，例如：
# from src.fitting.fit_smpl_from_data import SMPLFitterFromData

__all__ = [
    'SMPLFitterFromData',
    'fit_smpl_from_keypoints',
]
