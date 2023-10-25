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

        if random.choice([True, False]):
            system_response = inventory.SystemResponse('geophone',
                                                       resonance_frequency=15,
                                                       gain=47)
        else:
            system_response = inventory.SystemResponse('accelerometer',
                                                       resonance_frequency=2300,
                                                       gain=100,
                                                       sensitivity=0.2)

        system_response.add_cable(output_resistance=2500, cable_length=1000,
                                  cable_capacitance=1e-12)

        channels.append(inventory.Channel(channel_code, location_code,
                                          coordinates=Coordinates(x, y, z),
                                          orientation_vector=orientation_vector,
                                          response=system_response.response))
    station_code = f'STA{i:02d}'
    x, y, z = [random.randint(0, 1000) for i in range(0, 3)]
    stations.append(inventory.Station(station_code, coordinates=Coordinates(x, y, z),
                                      channels=channels))

network = inventory.Network(code='XX', stations=stations)
inv = inventory.Inventory(networks=[network])

# reading and writing and checking that the object is the same before and after

inv.write('test.xml')

inv2 = inventory.read_inventory('test.xml')

