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

from uquake.core import coordinates
from uquake.core import event
from importlib import reload

reload(coordinates)
reload(event)

coord = coordinates.Coordinates(10, 10, 10)

json_string = coord.to_json()

coord1 = coordinates.Coordinates.from_json(json_string)

assert(coord == coord1)

origin = event.Origin(coordinates=coordinates)
# origin.coordinates = coord
