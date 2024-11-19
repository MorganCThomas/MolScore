import warnings

try:
    from .posecheck import PoseCheck

except Exception as e:
    warnings.warn(f"PoseCheck: currently unavailable due to the following: {e}")
