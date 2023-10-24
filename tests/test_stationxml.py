from uquake.core import inventory
from uquake.core.coordinates import Coordinates
from importlib import reload
import random
import numpy as np
reload(inventory)

stations = []
for i in range(0, 10):
    channels = []
    for channel_code in ['x', 'y', 'z']:
        x, y, z = [random.randint(0, 1000) for i in range(0, 3)]
        location_code = '00'
        orientation_vector = [random.randint(0, 1000) for i in range(0, 3)]
        orientation_vector = orientation_vector / np.linalg.norm(orientation_vector)

        channels.append(inventory.Channel(channel_code, location_code,
                                          coordinates=Coordinates(x, y, z),
                                          orientation_vector=orientation_vector))

        response = inventory.SystemResponse()
    station_code = f'STA{i:02d}'
    x, y, z = [random.randint(0, 1000) for i in range(0, 3)]
    stations.append(inventory.Station(station_code, coordinates=Coordinates(x, y, z),
                                      channels=channels))

network = inventory.Network(code='XX', stations=stations)
inv = inventory.Inventory(networks=[network])

