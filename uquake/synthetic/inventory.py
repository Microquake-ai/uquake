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

from uquake.core import inventory
from uquake.core.coordinates import Coordinates
import random
import numpy as np
import string


def generate_unique_instrument_code(n_codes: int = 1):

    codes = set()

    while len(codes) < n_codes:
        # Ensure station code is unique
        part1 = ''.join(random.choices(string.ascii_uppercase, k=5))
        part2 = ''.join(random.choices('0123456789', k=2))
        new_code = f'{part1}_{part2}'
        if new_code not in codes:
            codes.add(new_code)

    return list(codes)


def generate_inventory(n_station=30):
    stations = []
    for i in range(n_station):
        channels = []
        for channel_code in ['x', 'y', 'z']:
            x, y, z = [random.randint(0, 1000) for _ in range(3)]
            location_code = '00'
            orientation_vector = np.array([random.randint(0, 1000) for _ in
                                           range(3)]).astype(float)
            orientation_vector /= np.linalg.norm(orientation_vector)

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
        x, y, z = [random.randint(0, 1000) for _ in range(3)]
        stations.append(inventory.Station(station_code,
                                          coordinates=Coordinates(x, y, z),
                                          channels=channels))

    network = inventory.Network(code='XX', stations=stations)
    inv = inventory.Inventory(networks=[network])

    return inv