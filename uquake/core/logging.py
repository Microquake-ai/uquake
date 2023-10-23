# Copyright (C) 2023, Jean-Philippe Mercier
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from loguru import logger
import sys
import os

if 'DEBUG_LEVEL' in os.environ:
    DEBUG_LEVEL = os.environ['DEBUG_LEVEL']
else:
    DEBUG_LEVEL = 0

logger.remove()
logging_level = DEBUG_LEVEL
logger.add(sys.stderr, level=logging_level)
