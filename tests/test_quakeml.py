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