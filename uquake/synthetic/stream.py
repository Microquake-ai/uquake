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

from .event import generate_catalog
from .inventory import generate_inventory
from uquake.core.stream import Stream, Trace
from uquake.core.trace import Stats
from uquake.core.inventory import Inventory
import numpy as np
from uquake.core import UTCDateTime

catalogue = generate_catalog()
inventory = generate_inventory()


# generate random waveforms based on the inventory
def generate_waveform(inv: Inventory, sampling_rate: float = 5000,
                      duration: float = 2, start_time: UTCDateTime = UTCDateTime.now()):
    traces = []
    for station in inv[0]:
        for channel in station:
            data = np.random.randn(sampling_rate * duration)
            stats = Stats()
            stats.sampling_rate = sampling_rate
            stats.starttime = start_time
            stats.npts = len(data)
            stats.network = inv[0].code
            stats.station = station.code
            stats.location = channel.location_code
            stats.channel = channel.code
            trace = Trace(data=data, header=stats)
            traces.append(trace)

    return Stream(traces=traces)