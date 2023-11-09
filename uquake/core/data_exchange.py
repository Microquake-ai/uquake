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

from uquake.core.stream import Stream, Trace
from uquake.core.trace import Stats
from uquake.core.event import Catalog, EventTypeLookup
from uquake.core.inventory import Inventory
from uquake.core import read, read_events, read_inventory
from obspy.core.trace import Trace as ObspyTrace
import random
import tarfile
from io import BytesIO
import string
import pyasdf
from typing import List, Union


class MicroseismicDataExchange(object):
    """
    A class to handle the exchange of microseismic data (stream, catalog, and inventory) using the ASDF format.

    :ivar stream: The ObsPy/uQuake Stream object representing the seismic waveform data.
    :type stream: Stream, optional
    :ivar catalog: The ObsPy/uQuake Catalog object representing the event data.
    :type catalog: Catalog, optional
    :ivar inventory: The ObsPy/uQuake Inventory object representing the station and channel metadata.
    :type inventory: Inventory, optional
    """

    def __init__(self, stream: Stream = None, catalog: Catalog = None,
                 inventory: Inventory = None):
        """
        Initializes the MicroseismicDataExchange object.

        :param stream: The ObsPy/uQuake Stream object representing the seismic waveform data.
        :type stream: Stream, optional
        :param catalog: The ObsPy/uQuake Catalog object representing the event data.
        :type catalog: Catalog, optional
        :param inventory: The ObsPy/uQuake Inventory object representing the station and channel metadata.
        :type inventory: Inventory, optional
        """
        self.stream = stream
        self.catalog = catalog
        self.inventory = inventory

    def __eq__(self, other):
        if isinstance(other, MicroseismicDataExchange):
            return self.stream == other.stream and \
                   self.catalog == other.catalog and \
                   self.inventory == other.inventory
        else:
            return False

    def write(self, file_path: str, waveform_tag: str = 'default',
              compression: str = 'gzip-3', shuffle: bool = True):
        """
        Writes the stream, catalog, and inventory data to a ASDF file.

        :param file_path: The path to the ASDF file where the data will be written.
        :type file_path: str
        :param waveform_tag: Tag describing the waveforms.
        :type waveform_tag: str
        :param compression: The compression type to use when writing the ASDF file
        (see pyasdf documentation) - default 'gzip-3'.
        :type compression: str
        :shuffle: Whether or not to shuffle the data when writing the ASDF file.
        Shuffling could increase the compression ratio.
        """

        event_type_lookup = EventTypeLookup()
        event_types = [event.event_type for event in self.catalog]
        for i, event in enumerate(self.catalog):
            if event_type_lookup.is_valid_uquakeml(event.event_type):
                self.catalog[i].event_type = \
                    event_type_lookup.convert_to_quakeml(self.catalog[i].event_type)

        asdf_handler = ASDFHandler(file_path, compression=compression, mode='a',
                                   shuffle=shuffle)

        for i in range(len(self.catalog[0].picks)):
            self.catalog.pick[i].waveform_id.station_code.replace('.', '_')

        for i in range(len(self.inventory[0])):
            self.inventory[0][i].code.replace('.', '_')

        for i in range(len(self.stream)):
            self.stream[i].stats.station_code.replace('.', '_')

        asdf_handler.add_catalog(self.catalog)
        asdf_handler.add_inventory(self.inventory)

        # ensuring the waveform are not represented using double but single precision
        for tr in self.stream:
            tr.data = tr.data.astype('float32')

        asdf_handler.add_waveforms(self.stream, waveform_tag=waveform_tag)

        for i, event in enumerate(self.catalog):
            self.catalog[i].event_type = event_types[i]

    @classmethod
    def read(cls, file_path: str, waveform_tag: str = 'default') \
            -> 'MicroseismicDataExchange':
        """
        Reads the stream, catalog, and inventory data from a ASDF file.

        :param file_path: The path to the ASDF file from which the data will be read.
        :type file_path: str
        :param waveform_tag: Tag describing the waveforms.
        :type waveform_tag: str
        :return: An instance of the MicroseismicDataExchange class with the read data.
        :rtype: MicroseismicDataExchange
        """
        asdf_handler = ASDFHandler(file_path)
        stream = \
            Stream(asdf_handler.get_all_waveforms(tags=[waveform_tag])[waveform_tag])
        catalog = Catalog(obspy_obj=asdf_handler.get_catalog())
        inventory = Inventory.from_obspy_inventory_object(asdf_handler.get_inventory())

        event_type_lookup = EventTypeLookup()
        for i, event in enumerate(catalog):
            if event_type_lookup.is_valid_quakeml(event.event_type):
                catalog[i].event_type = \
                    event_type_lookup.convert_from_quakeml(catalog[i].event_type)

        return cls(stream=stream, catalog=catalog, inventory=inventory)


class ASDFHandler:
    def __init__(self, asdf_file_path, compression='gzip-3', mode='a', **kwargs):
        """
        Initialize the ASDFHandler with a given ASDF file path.
        :param asdf_file_path: Path to the ASDF file.
        :param kwargs: Keyword arguments to be passed to pyasdf.ASDFDataSet.__init__().
        """ + pyasdf.ASDFDataSet.__init__.__doc__
        self.asdf_file_path = asdf_file_path
        self.ds = pyasdf.ASDFDataSet(self.asdf_file_path, mode=mode,
                                     compression=compression,
                                     **kwargs)

    def add_catalog(self, catalog):
        """
        Add a seismic catalog to the ASDF dataset.
        :param catalog: ObsPy Catalog object
        """

        self.ds.add_quakeml(catalog)

    def get_catalog(self):
        """
        Retrieve the seismic catalog from the ASDF dataset.
        :return: ObsPy Catalog object
        """

        return self.ds.events
        return Catalog(obspy_obj=self.ds.events)

    def add_inventory(self, inventory):
        """
        Add a seismic inventory to the ASDF dataset.
        :param inventory: ObsPy Inventory object
        """
        self.ds.add_stationxml(inventory)

    def get_inventory(self):
        """
        Retrieve the seismic inventory from the ASDF dataset.
        :return: uQuake Inventory object
        """

        network = {}
        for station_name in self.ds.waveforms.list():
            inv = self.ds.waveforms[station_name].StationXML
            if inv.networks[0].code in network.keys():
                network[inv.networks[0].code].stations += inv[0].stations
            else:
                network[inv.networks[0].code] = inv[0]

        networks = []
        for key in network.keys():
            networks.append(network[key])

        inv = Inventory.from_obspy_inventory_object(Inventory(networks=networks))

        return inv

    def add_waveforms(self, stream, waveform_tag: str = 'default'):
        """
        Add a seismic stream to the ASDF dataset.
        :param stream: ObsPy Stream object
        :param waveform_tag: tag describing the waveforms to retrieve
        (e.g. 'raw', 'processed')
        """

        for tr in stream:
            self.ds.add_waveforms(tr, waveform_tag)

    def get_waveforms(self, network=None, station=None, location=None, channel=None,
                      waveform_tag='*', starttime=None, endtime=None):

        network = network or "*"
        station = station or "*"
        location = location or "*"
        channel = channel or "*"

        obj = self.ds.get_waveforms(network, station, location, channel,
                                    tag=waveform_tag, starttime=starttime,
                                    endtime=endtime)

        traces = []
        for item in obj:
            if isinstance(item, ObspyTrace):
                data = item.data
                stats = Stats()
                for key in item.stats.__dict__.keys():
                    stats.__dict__[key] = item.stats.__dict__[key]
                data.stats[key] = item.stats[key]
                trace = Trace(data=data, header=stats)
                traces.append(trace)

        return Stream(traces=traces)

    def get_all_waveforms(self, tags: Union[List[str], str] = []):
        """
        Retrieve specific waveforms from the ASDF dataset for a specific tag.
        :param tags: Tag describing the waveforms if None data for all tags will be
        retrieved.
        :type tags: str or list[str]
        :return: a dictionary of uquake.core.stream.Stream objects
        """

        stations = self.ds.waveforms.list()

        stream_dict = {}

        for station in stations:
            # station_code = station.replace('.', '_')
            sta = self.ds.waveforms[station_code]

            if tags:
                tags = sta.get_waveform_tags()
            elif isinstance(tag, str):
                tags = [tags]

            for tag in tags:
                if tag in stream_dict.keys():
                    stream_dict[tag] += sta[tag]
                else:
                    stream_dict[tag] = sta[tag]

        return stream_dict


def read_asdf(file_path, waveform_tag='default'):
    """
    Read a ASDF file and return a MicroseismicDataExchange object.
    :param file_path: path to file
    :type file_path: str
    :param waveform_tag: tag describing the waveforms to retrieve
    (e.g. 'raw', 'processed') default = 'default'
    :type waveform_tag: str
    :return: MicroseismicDataExchange object
    """
    return MicroseismicDataExchange.read(file_path, waveform_tag=waveform_tag)


def generate_unique_names(n):
    """
    Generate n number of unique 5-character names comprising only lower and upper case
    letters.

    :param n: The number of unique names to generate.
    :return: A list of n unique 5-character names.
    """
    names = set()  # Set to store unique names

    while len(names) < n:
        name = ''.join(random.choices(string.ascii_letters, k=5))
        names.add(name)

    return list(names)
