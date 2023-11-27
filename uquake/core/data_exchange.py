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
import string
import pyasdf
from typing import List, Union
import re
from uquake.io.data_exchange.zarr import (read_zarr, write_zarr, get_inventory,
                                          get_catalog)
import numpy as np
import io


def validate_station_code(code):
    # Replace non-alphanumeric characters (except spaces) with an underscore
    valid_code = re.sub(r'[^\w\s]', '_', code.upper())
    return valid_code


# station_name = "7000L SWRM/Crush"
# valid_station_code = validate_and_correct_station_code(station_name)
# print(f"Valid station code: {valid_station_code}")


class MicroseismicDataExchange(object):
    """
    A class to handle the exchange of microseismic data (stream, catalog, and inventory)
    using a composite format including the streams, the catalog and inventory.

    This object handles the reading and writing of the data to and from a files in
    various formats including ASDF, Zarr and eventually TileDB.


    :ivar stream: The ObsPy/uQuake Stream object representing the seismic waveform data.
    :type stream: Stream, optional
    :ivar catalog: The ObsPy/uQuake Catalog object representing the event data.
    :type catalog: Catalog, optional
    :ivar inventory: The ObsPy/uQuake Inventory object representing the station and
    channel metadata.
    :type inventory: Inventory, optional
    """

    def __init__(self, stream: Stream = None, catalog: Catalog = None,
                 inventory: Inventory = None):
        """
        Initializes the MicroseismicDataExchange object.

        :param stream: The ObsPy/uQuake Stream object representing the seismic waveform
        data.
        :type stream: Stream, optional
        :param catalog: The ObsPy/uQuake Catalog object representing the event data.
        :type catalog: Catalog, optional
        :param inventory: The ObsPy/uQuake Inventory object representing the station and
        channel metadata.
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

    def remove_instrument(self, station_code, location_code):
        """
        Remove a station from the inventory and the stream
        :param station_code: station code
        :return:
        """

        new_stations = []
        new_traces = []
        for station in self.inventory[0]:
            new_channels = []
            for channel in station:
                if (channel.location_code == location_code) & \
                        (station.code == station_code):
                    continue
                new_channels.append(channel)

            if len(new_channels) != 0:
                station.channels = new_channels
                new_stations.append(station)

        self.inventory[0].stations = new_stations

        for tr in self.stream:
            if (tr.stats.station == station_code) \
                    & (tr.stats.location == location_code):
                continue
            new_traces.append(tr)

        self.stream.traces = new_traces

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
        :shuffle: Whether to shuffle the data when writing the ASDF file.
        Shuffling could increase the compression ratio.
        """

        event_type_lookup = EventTypeLookup()
        event_types = [event.event_type for event in self.catalog]
        for i, event in enumerate(self.catalog):
            if event_type_lookup.is_valid_uquakeml(event.event_type):
                self.catalog[i].event_type = \
                    event_type_lookup.convert_to_quakeml(self.catalog[i].event_type)
            else: # if event type is not valid, set it to unknown
                self.catalog[i].event_type = 'other event'

        asdf_handler = ASDFHandler(file_path, compression=compression, mode='a',
                                   shuffle=shuffle)

        for i in range(len(self.catalog[0].picks)):
            self.catalog[0].picks[i].waveform_id.station_code = \
                validate_station_code(self.catalog[0].picks[i].waveform_id.station_code)

        for i in range(len(self.inventory[0])):
            self.inventory[0][i].code = \
                validate_station_code(self.inventory[0][i].code)

        for i in range(len(self.stream)):
            self.stream[i].stats.station = \
                validate_station_code(self.stream[i].stats.station)

        # import ipdb
        # ipdb.set_trace()

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

    def p_sv_sh_stream_from_hodogram(self, window_length=0.02):
        """
        create a P SV and SH stream oriented based on the polarity of the incoming
        wavefield
        """
        # Rotate the stream in "E", "N", "Z" if that is not yet the case
        st_zne = self.stream.copy().rotate('->ZNE', inventory=self.inventory)
        st_zne = st_zne.filter('highpass', freq=10)

        event = self.catalog[0]
        inventory = self.inventory
        if event.preferred_origin() is not None:
            origin = event.preferred_origin()
        else:
            origin = event.origins[-1]

        st_out = Stream()

        for arrival in origin.arrivals:
            if arrival.phase == 'S':
                continue
            pick = arrival.pick
            network = pick.waveform_id.network_code
            station = pick.waveform_id.station_code
            location = pick.waveform_id.location_code

            start_time = pick.time
            end_time = pick.time + window_length

            st_tmp = st_zne.copy().select(network=network, station=station,
                                          location=location).trim(
                starttime=start_time, endtime=end_time)

            if len(st_tmp) != 3:
                for tr in st_tmp:
                    st_out.traces.append(tr)
                continue

            wave_mat = []
            for component in ['E', 'N', 'Z']:
                tr = st_tmp.select(component=component)[0]
                wave_mat.append(tr.data)

            wave_mat = np.array(wave_mat)

            cov_mat = np.cov(wave_mat)

            eig_vals, eig_vects = np.linalg.eig(cov_mat)
            i_ = np.argsort(eig_vals)

            if arrival.phase == 'P':
                eig_vect = eig_vects[i_[-1]]
                linearity = (1 - np.linalg.norm(eig_vals[i_[:2]]) /
                             eig_vals[i_[2]])
                color = 'b'
            elif arrival.phase == 'S':
                eig_vect = eig_vects[i_[0]]
                linearity = (1 - eig_vals[i_[0]] /
                             np.linalg.norm(eig_vals[i_[1:]]))
                color = 'r'

            sta = inventory.select(network=network,
                                   station=station,
                                   location=location)[0][0]

            eig_vect /= np.linalg.norm(eig_vect)
            ray = origin.rays.select(network=network, station=station, location=location,
                                     phase='P')[0]



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
            sta = self.ds.waveforms[station]

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


class ZarrHandler:

    def __init__(self, file_path=None):
        self.file_path = file_path

    @staticmethod
    def write(file_path, mde: MicroseismicDataExchange):
        """
        Write a MicroseismicDataExchange object to a Zarr file.
        :param mde:
        :return:
        """

        write_zarr(file_path, mde)

    @staticmethod
    def read(file_path):
        """
        Read a MicroseismicDataExchange object from a Zarr file.
        :return:
        """

        data_dict = read_zarr(file_path)
        return MicroseismicDataExchange(**data_dict)

    def get_inventory(self):
        """
        Retrieve the seismic inventory from the Zarr dataset.
        :return: uQuake Inventory object
        """
        z = zarr.open(self.file_path, mode='r')
        return get_inventory(z)

    @staticmethod
    def get_catalog(self):
        """
        Retrieve the seismic catalog from the Zarr dataset.
        :return: uQuake Catalog object
        """

        z = zarr.open(self.filepath, mode='r')
        return get_catalog(z)

    def get_stream(self, networks: List[str] = [], stations: List[str] = [],
                   locations: List[str] = [], channels: List[str] = []):
        """
        Retrieve specific waveforms from the Zarr dataset.
        :param networks: List of network codes
        :param stations: List of station codes
        :param locations: List of location codes
        :param channels: List of channel codes
        :return: uquake.core.stream.Stream object
        """
        stream = Stream()

        stream_group = zarr.open_group(self.file_path / 'stream', mode='r')

        for network in stream_group:
            if networks and network not in networks:
                continue
            for station in stream_group[network]:
                if stations and station not in stations:
                    continue
                for location in stream_group[network][station]:
                    if locations and location not in locations:
                        continue
                    for channel, arr in stream_group[network][station][
                        location].items():
                        if channels and channel not in channels:
                            continue
                        tr = Trace(data=arr[:])
                        for key in ['network', 'station', 'location', 'channel',
                                    'sampling_rate', 'starttime', 'calib']:
                            tr.stats[key] = arr.attrs[key]
                        stream.append(tr)

        return stream





