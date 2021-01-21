from loguru import logger
import sys

from microquake.core.settings import settings

logging_level = settings.DEBUG_LEVEL
logger.add(sys.stderr, level=logging_level)
logger.remove(0)
