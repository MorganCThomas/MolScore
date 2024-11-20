import logging

logger = logging.getLogger(__name__)

try:
    from .posecheck import PoseCheck

except Exception as e:
    logger.warning(f"PoseCheck metrics: currently unavailable due to the following: {e}")
