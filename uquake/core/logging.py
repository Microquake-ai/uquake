from loguru import logger
import sys
import os

if 'DEBUG_LEVEL' in os.environ:
    DEBUG_LEVEL = os.environ['DEBUG_LEVEL']
else:
    DEBUG_LEVEL = 0

logging_level = DEBUG_LEVEL
logger.add(sys.stderr, level=logging_level)
logger.remove(0)
