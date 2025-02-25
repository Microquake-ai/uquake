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

# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: inventory.py
#  Purpose: Expansion of the obspy.core.inventory.inventory module
#   Author: uquake development team
#    Email: devs@uquake.org
#
# Copyright (C) 2016 uquake development team
# --------------------------------------------------------------------
"""
Expansion of the obspy.core.event module

:copyright:
    uquake development team (devs@uquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import io
from obspy.core import inventory, AttribDict
from obspy.core.inventory import *
import numpy as np
from obspy.core.inventory.util import (Equipment, Operator, Person,
                                       PhoneNumber, Site, _textwrap,
                                       _unified_content_strings)
from uquake.core.util.decorators import expand_input_format_compatibility
from uquake.core.coordinates import Coordinates, CoordinateSystem
from pathlib import Path

from .logging import logger
from uquake import __package_name__ as ns

import pandas as pd
from io import BytesIO
from .util.tools import lon_lat_x_y

from .util import ENTRY_POINTS
from pkg_resources import load_entry_point
from tempfile import NamedTemporaryFile
import os
from .util.requests import download_file_from_url
from uquake.core.util.attribute_handler import set_extra, get_extra, namespace
import hashlib
from enum import Enum
from pydantic import BaseModel
from typing import Union, Literal, Optional, List
from datetime import datetime
from uquake.core.util.decorators import deprecated


class Inventory(inventory.Inventory):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(self, *args, **kwargs)

    @classmethod
    def from_obspy_inventory_object(cls, obspy_inventory,
                                    xy_from_lat_lon=False):

        source = ns  # Network ID of the institution sending
        # the message.

        inv = cls([], ns)
        inv.networks = []
        for network in obspy_inventory.networks:
            inv.networks.append(Network.from_obspy_network(network,
                                                           xy_from_lat_lon=
                                                           xy_from_lat_lon))

        return inv

    @staticmethod
    def from_url(url):
        """
        Load an ObsPy inventory object from a URL.

        :param url: The URL to download the inventory file from.
        :type url: str
        :return: The loaded ObsPy inventory object.
        :rtype: obspy.core.inventory.Inventory
        """
        # Download the inventory file from the URL

        inventory_data = download_file_from_url(url)

        # Save the inventory data to a temporary file
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(inventory_data.read())
            temp_file.flush()

            # Load the inventory from the temporary file
            file_format = 'STATIONXML'  # Replace with the correct format
            inventory = read_inventory(temp_file.name, format=file_format)

        # Remove the temporary file after reading the inventory
        os.remove(temp_file.name)

        return inventory

    def write(self, path_or_file_obj, *args, format='stationxml', **kwargs):
        return super().write(path_or_file_obj, *args, format=format,
                             nsmap={'mq': namespace}, **kwargs)

    def get_station(self, sta):
        return self.select(sta)

    def get_channel(self, station=None, location=None, channel=None):
        return self.select(station=station, location=location, channel=channel)

    def select_instrument(self, instruments=None):
        if isinstance(instruments, list):
            for location in instruments:
                for obj_site in self.instruments:
                    if location.code == obj_site.code:
                        yield obj_site

        elif isinstance(instruments, str):
            location = instruments
            for obj_site in self.instruments:
                if location.code == obj_site.code:
                    return obj_site

    def to_bytes(self):

        file_out = BytesIO()
        self.write(file_out, format='stationxml')
        file_out.seek(0)
        return file_out.getvalue()

    @staticmethod
    def from_bytes(byte_string):
        file_in = io.BytesIO(byte_string)
        file_in.read()
        return read_inventory(file_in, format='stationxml')

    @staticmethod
    def read(path_or_file_obj, format='stationxml', **kwargs):
        return read_inventory(path_or_file_obj, format=format, **kwargs)

    def __eq__(self, other):
        return np.all(self.instruments == other.instruments)

    # def write(self, filename):
    #     super().write(self, filename, format='stationxml', nsmap={ns: ns})

    @property
    def instruments(self):
        instruments = []
        for network in self.networks:
            for station in network.stations:
                for instrument in station.instruments:
                    instruments.append(instrument)

        return np.sort(instruments)

    @property
    def short_ids(self):
        unique_ids = set()
        short_ids = []

        for network in self.networks:
            for station in network.stations:
                for instrument in station.instruments:
                    if len(instrument.code) > 6:
                        hash = hashlib.md5(instrument.code.encode()).hexdigest()[:5]
                        if hash + '0' not in unique_ids:
                            unique_ids.add(hash + '0')
                            short_ids.append(hash + '0')
                        else:
                            # First try appending numbers
                            found_unique = False
                            for i in range(1, 10):
                                potential_id = hash + str(i)
                                if potential_id not in unique_ids:
                                    unique_ids.add(potential_id)
                                    short_ids.append(potential_id)
                                    found_unique = True
                                    break

                            # If all numbers are used, start appending letters
                            if not found_unique:
                                for letter in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                                    potential_id = hash + letter
                                    if potential_id not in unique_ids:
                                        unique_ids.add(potential_id)
                                        short_ids.append(potential_id)
                                        break
        return short_ids

    def instrument_code_from_shortids(self, short_id):
        """
        return the instrument code from the short id
        :param short_id: a string representing the short id
        :return:
        """

        short_ids = self.short_ids

        for i, sid in enumerate(short_ids):
            if sid == short_id:
                return self.instruments[i].code


class Network(inventory.Network):
    __doc__ = inventory.Network.__doc__.replace('obspy', ns)

    extra_keys = ['vp', 'vs']

    def __init__(self, *args, **kwargs):

        if 'extra' not in self.__dict__.keys():  # hack for deepcopy to work
            self['extra'] = {}

        self.extra = AttribDict()

        # for extra_key in self.extra_keys:
        #     if extra_key in kwargs.keys():
        #         self.extra[extra_key] = kwargs.pop(extra_key)

        if 'vp' in kwargs.keys():
            self.vp = kwargs.pop('vp')
        if 'vs' in kwargs.keys():
            self.vs = kwargs.pop('vs')
        if 'units' in kwargs.keys():
            self.units = kwargs.pop('units')

        super().__init__(*args, **kwargs)

    def __setattr__(self, name, value):
        name = name.lower()
        if name in self.extra_keys:
            self.extra[name] = AttribDict({"value": f"{value}", "namespace": namespace})
        else:
            super().__setattr__(name, value)

    def __getattr__(self, item):
        item = item.lower()
        if item in self.extra_keys:
            try:
                return float(self.extra[item].value)
            except:
                return self.extra[item].value
        else:
            super().__getattr__(item)

    def __setitem__(self, key, value):
        self.__dict__['key'] = value

    @classmethod
    def from_obspy_network(cls, obspy_network, xy_from_lat_lon=False,
                           input_projection=4326, output_projection=None):

        net = cls(obspy_network.code)

        for key in obspy_network.__dict__.keys():
            if 'stations' in key:
                net.__dict__[key] = []
            else:
                try:
                    net.__dict__[key] = obspy_network.__dict__[key]
                except Exception as e:
                    logger.error(e)

        for i, station in enumerate(obspy_network.stations):
            net.stations.append(Station.from_obspy_station(station,
                                                           xy_from_lat_lon))

        return net

    def get_grid_extent(self, padding_fraction=0.1, ignore_stations=[],
                        ignore_sites=[]):
        """
        return the extents of a grid encompassing all sensors comprised in the
        network
        :param padding_fraction: buffer to add around the stations
        :param ignore_stations: list of stations to exclude from the
        calculation of the grid extents
        :param ignore_sites: list of location to exclude from the calculation of
        the grid extents
        :type ignore_stations: list
        :return: return the lower and upper corners
        :rtype: dict
        """

        xs = []
        ys = []
        zs = []

        coordinates = []
        for station in self.stations:
            if station.code in ignore_stations:
                continue
            for location in station.instruments:
                if location.code in ignore_sites:
                    continue
                coordinates.append(location.loc)

        min = np.min(coordinates, axis=0)
        max = np.max(coordinates, axis=0)
        # center = min + (max - min) / 2
        center = (min + max) / 2
        d = (max - min) * (1 + padding_fraction)

        c1 = center - d / 2
        c2 = center + d / 2

        return c1, c2

    @property
    def site_coordinates(self):
        coordinates = []
        for station in self.stations:
            coordinates.append(station.site_coordinates)

        return np.array(coordinates)

    @property
    def station_coordinates(self):
        return np.array([station.loc for station in self.stations])

    @property
    def instruments(self):
        instruments = []
        for station in self.stations:
            for instrument in station.instruments:
                instruments.append(instrument)
        return instruments

    @property
    def sensors(self):
        return self.instruments

    @property
    def instrument_coordinates(self):
        coordinates = []
        for instrument in self.instruments:
            coordinates.append(instrument.loc)
        return np.array(coordinates)

    @property
    def sensor_coordinates(self):
        return self.instrument_coordinates


class Station(inventory.Station):
    __doc__ = inventory.Station.__doc__.replace('obspy', ns)

    def __init__(self, *args, coordinates: Coordinates = Coordinates(0, 0, 0), **kwargs):

        if 'extra' not in self.__dict__.keys():  # hack for deepcopy to work
            self['extra'] = {}

        self.extra = AttribDict()

        if 'latitude' not in kwargs.keys():
            kwargs['latitude'] = 0
        if 'longitude' not in kwargs.keys():
            kwargs['longitude'] = 0
        if 'elevation' not in kwargs.keys():
            kwargs['elevation'] = 0

        # initialize the extra key

        if not hasattr(self, 'extra'):
            self.extra = AttribDict()

        super().__init__(*args, **kwargs)

        self.extra['coordinates'] = coordinates.to_extra_key(namespace=namespace)

    def __setattr__(self, name, value):
        if name == 'coordinates':
            self.extra[name] = value.to_extra_key(namespace=namespace)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, item):
        if item == 'coordinates':
            return Coordinates.from_extra_key(self.extra[item])
        else:
            super().__getattr__(item)

    def __setitem__(self, key, value):
        self.__dict__['key'] = value

    def __eq__(self, other):
        if not super().__eq__(other):
            return False

        return self.coordinates == other.coordinates

    @classmethod
    def from_obspy_station(cls, obspy_station, xy_from_lat_lon=False):

        #     cls(*params) is same as calling Station(*params):

        stn = cls(obspy_station.code, latitude=obspy_station.latitude,
                  longitude=obspy_station.longitude,
                  elevation=obspy_station.elevation)
        for key in obspy_station.__dict__.keys():
            try:
                stn.__dict__[key] = obspy_station.__dict__[key]
            except Exception as e:
                logger.error(e)

        if xy_from_lat_lon:
            if (stn.latitude is not None) and (stn.longitude is not None):

                x, y = lon_lat_x_y(
                    longitude=stn.longitude, latitude=stn.latitude)

                z = obspy_station.elevation

                stn.coordinates = Coordinates(
                    x, y, z, coordinate_system=CoordinateSystem.NEU)

            else:
                logger.warning(f'Latitude or Longitude are not'
                               f'defined for station {obspy_station.code}.')

                output_projection = 32725

        stn.channels = []

        for channel in obspy_station.channels:
            stn.channels.append(Channel.from_obspy_channel(channel,
                                                           xy_from_lat_lon=
                                                           xy_from_lat_lon))

        return stn

    @property
    def x(self):
        return self.coordinates.x

    @property
    def y(self):
        return self.coordinates.y

    @property
    def z(self):
        return self.coordinates.z

    @property
    def loc(self):
        return np.array(self.coordinates.loc)

    @property
    def instruments(self):
        location_codes = []
        channel_dict = {}
        instruments = []
        for channel in self.channels:
            location_codes.append(channel.location_code)
            channel_dict[channel.location_code] = []

        for channel in self.channels:
            channel_dict[channel.location_code].append(channel)

        for key in channel_dict.keys():
            instruments.append(Instrument(self, channel_dict[key]))

        return instruments

    @property
    def sensors(self):
        return self.instruments

    @property
    def instrument_coordinates(self):
        coordinates = []
        for instrument in self.instruments:
            coordinates.append(instrument.loc)
        return np.array(coordinates)

    @property
    def sensor_coordinates(self):
        return self.instrument_coordinates

    def __str__(self):
        contents = self.get_contents()

        x = self.x
        y = self.y
        z = self.z

        location_count = len(self.instruments)
        channel_count = len(self.channels)

        format_dict = {
            'code': self.code or 'N/A',
            'location_count': location_count or 0,
            'channel_count': channel_count or 0,
            'start_date': self.start_date or 'N/A',
            'end_date': self.end_date or 'N/A',
            'x': f"{x:.0f}" if x is not None else 0,
            'y': f"{y:.0f}" if y is not None else 0,
            'z': f"{z:.0f}" if z is not None else 0
        }

        ret = ("\tStation Code: {code}\n"
               "\tLocation Count: {location_count}\n"
               "\tChannel Count: {channel_count}\n"
               "\tDate range: {start_date} - {end_date}\n"
               "\tx: {x}, y: {y}, z: {z} m\n").format(**format_dict)

        if getattr(self, 'extra', None):
            if getattr(self.extra, 'x', None) and getattr(self.extra, 'y',
                                                          None):
                x = self.x
                y = self.y
                z = self.z
                ret = ("Station {station_name}\n"
                       "\tStation Code: {station_code}\n"
                       "\tLocation Count: {site_count}\n"
                       "\tChannel Count: {selected}/{total}"
                       " (Selected/Total)\n"
                       "\tDate range: {start_date} - {end_date}\n"
                       "\tEasting [x]: {x:.0f} m, Northing [y] m: {y:.0f}, "
                       "Elevation [z]: {z:.0f} m\n")

        ret = ret.format(
            station_name=contents["stations"][0],
            station_code=self.code,
            site_count=len(self.instruments),
            selected=self.selected_number_of_channels,
            total=self.total_number_of_channels,
            start_date=str(self.start_date),
            end_date=str(self.end_date) if self.end_date else "",
            restricted=self.restricted_status,
            alternate_code="Alternate Code: %s " % self.alternate_code if
            self.alternate_code else "",
            historical_code="Historical Code: %s " % self.historical_code if
            self.historical_code else "",
            x=x, y=y, z=z)
        ret += "\tAvailable Channels:\n"
        ret += "\n".join(_textwrap(
            ", ".join(_unified_content_strings(contents["channels"])),
            initial_indent="\t\t", subsequent_indent="\t\t",
            expand_tabs=False))

        return ret

    def __repr__(self):
        return self.__str__()


class Instrument:
    """
    This class is a container for grouping the channels into coherent entity
    that are Instruments. From the uquake package perspective a station is
    the physical location where data acquisition instruments are grouped.
    One or multiple instruments can be connected to a station.
    """

    def __init__(self, station, channels):

        location_codes = []
        for channel in channels:
            location_codes.append(channel.location_code)
            if len(np.unique(location_codes)) > 1:
                raise ValueError('the channels in the channel list should have a unique '
                                 'location code')

        self.location_code = location_codes[0]

        if len(np.unique(location_codes)) > 1:
            logger.error('the channels in the channel list should have a'
                         'unique location code')
            raise KeyError

        self.station = station
        self.channels = channels

    def __repr__(self):
        ret = f'\tInstrument {self.instrument_code}\n' \
              f'\tx: {self.x:.0f} m, y: {self.y:.0f} m z: {self.z:0.0f} m\n' \
              f'\tChannel Count: {len(self.channels)}'

        return ret

    def __str__(self):
        return self.instrument_code

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __getitem__(self, item):
        return self.channels[item]

    @property
    def loc(self):
        return np.array([self.x, self.y, self.z])

    @property
    def alternate_code(self):
        return self.channels[0].alternative_code

    @property
    def x(self):
        return self.channels[0].x

    @property
    def y(self):
        return self.channels[0].y

    @property
    def z(self):
        return self.channels[0].z

    @property
    def coordinates(self):
        return self.channels[0].coordinates

    @property
    def station_code(self):
        return self.station.code

    @property
    def instrument_code(self):
        return self.code

    @property
    def simplified_code(self):
        return f'{self.station_code}{self.location_code}'

    @property
    def code(self):
        return self.make_instrument_code(self.station_code, self.location_code)

    @property
    def label(self):
        return self.simplified_code

    @property
    def sensor_type_code(self):
        return self.channels[0].code[0:-1]

    @staticmethod
    def make_instrument_code(station_code, location_code):
        return f'{station_code}.{location_code}'

    @property
    def coordinate_system(self):
        return self.coordinates.coordinate_system


class Channel(inventory.Channel):
    defaults = {}

    __doc__ = inventory.Channel.__doc__.replace('obspy', ns)

    def __init__(self, code, location_code, active: bool = True, oriented: bool = False,
                 coordinates: Coordinates = Coordinates(0, 0, 0),
                 orientation_vector=None, **kwargs):

        latitude = kwargs.pop('latitude') if 'latitude' in kwargs.keys() else 0
        longitude = kwargs.pop('longitude') if 'longitude' in kwargs.keys() else 0
        elevation = kwargs.pop('elevation') if 'elevation' in kwargs.keys() else 0
        depth = kwargs.pop('depth') if 'depth' in kwargs.keys() else 0

        super().__init__(code, location_code, latitude, longitude,
                         elevation, depth, **kwargs)

        if 'extra' not in self.__dict__.keys():  # hack for deepcopy to work
            self.__dict__['extra'] = {}

        self.extra['coordinates'] = coordinates.to_extra_key(namespace=namespace)
        set_extra(self, 'active', active, namespace=namespace)
        set_extra(self, 'oriented', oriented, namespace=namespace)

        if orientation_vector is not None:
            # making the orientation vector (cosine vector) unitary
            orientation_vector = orientation_vector / np.linalg.norm(orientation_vector)
            self.set_orientation(orientation_vector)

    @classmethod
    def from_obspy_channel(cls, obspy_channel, xy_from_lat_lon=False,
                           output_projection=None, input_projection=4326):

        cha = cls(obspy_channel.code, obspy_channel.location_code,
                  latitude=obspy_channel.latitude,
                  longitude=obspy_channel.longitude,
                  elevation=obspy_channel.elevation,
                  depth=obspy_channel.depth)

        if hasattr(obspy_channel, 'extra'):
            for key in cha.extra.keys():
                if key not in obspy_channel.__dict__['extra'].keys():
                    cha.__dict__['extra'][key] = 0
                else:
                    cha.__dict__['extra'][key] = \
                        obspy_channel.__dict__['extra'][key]

        for key in obspy_channel.__dict__.keys():
            cha.__dict__[key] = obspy_channel.__dict__[key]

        if xy_from_lat_lon:
            if (cha.latitude is not None) and (cha.longitude is not None):

                x, y = lon_lat_x_y(longitude=cha.longitude, latitude=cha.latitude)

                z = cha.elevation

                coordinates = Coordinates(
                    x, y, z, coordinate_system=CoordinateSystem.NEU)
                cha.coordinates = coordinates

        return cha

    def __getattr__(self, item):
        if item == 'coordinates':
            return Coordinates.from_extra_key(self.extra['coordinates'])
        elif item in ('active', 'oriented'):
            return get_extra(item)
        else:
            if hasattr(super(), item):
                return getattr(super(), item)
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        if key == 'coordinates':
            self.extra[key] = value.to_extra_key(namespace=namespace)
        elif key in ('active', 'oriented'):
            set_extra(self, key, value, namespace=namespace)
        else:
            super().__setattr__(key, value)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__['key'] = value

    def __repr__(self):
        time_range = f"{self.start_date} - {self.end_date}" if self.start_date and \
                                                               self.end_date else 'N/A'

        if self.coordinates.coordinate_system == CoordinateSystem.ENU:

            attributes = {
                'Channel': self.code,
                'Location': self.location_code,
                'Time range': time_range,
                'Easting [x]': f"{self.x:0.0f} m" if self.x is not None else 'N/A',
                'Northing [y]': f"{self.y:0.0f} m" if self.y is not None else 'N/A',
                'Elevation (UP) [z]':
                    f"{self.z:0.0f} m" if self.z is not None else 'N/A',
                'Dip (degrees)': f"{self.dip:0.0f}" if self.dip is not None else 'N/A',
                'Azimuth (degrees)': f"{self.azimuth:0.0f}" if self.azimuth
                                                               is not None else 'N/A',
                'Response information': 'available' if self.response else 'not available'
            }

        elif self.coordinates.coordinate_system == CoordinateSystem.NED:
            attributes = {
                'Channel': self.code,
                'Location': self.location_code,
                'Time range': time_range,
                'Northing [x]': f"{self.x:0.0f} m" if self.x is not None else 'N/A',
                'Easting [y]': f"{self.y:0.0f} m" if self.y is not None else 'N/A',
                'Depth (Down) [z]': f"{self.z:0.0f} m" if self.z is not None else 'N/A',
                'Dip (degrees)': f"{self.dip:0.0f}" if self.dip is not None else 'N/A',
                'Azimuth (degrees)': f"{self.azimuth:0.0f}" if self.azimuth
                                                               is not None else 'N/A',
                'Response information': 'available' if self.response else 'not available'
            }

        elif self.coordinates.coordinate_system == CoordinateSystem.NEU:
            attributes = {
                'Channel': self.code,
                'Location': self.location_code,
                'Time range': time_range,
                'Northing [x]': f"{self.x:0.0f} m" if self.x is not None else 'N/A',
                'Easting [y]': f"{self.y:0.0f} m" if self.y is not None else 'N/A',
                'Elevation (up) [z]':
                    f"{self.z:0.0f} m" if self.z is not None else 'N/A',
                'Dip (degrees)': f"{self.dip:0.0f}" if self.dip is not None else 'N/A',
                'Azimuth (degrees)': f"{self.azimuth:0.0f}" if self.azimuth
                                                               is not None else 'N/A',
                'Response information': 'available' if self.response else 'not available'
            }

        elif self.coordinates.coordinate_system == CoordinateSystem.END:

            attributes = {
                'Channel': self.code,
                'Location': self.location_code,
                'Time range': time_range,
                'Easting [x]': f"{self.x:0.0f} m" if self.x is not None else 'N/A',
                'Northing [y]': f"{self.y:0.0f} m" if self.y is not None else 'N/A',
                'Depth (Down) [z]': f"{self.z:0.0f} m" if self.z is not None else 'N/A',
                'Dip (degrees)': f"{self.dip:0.0f}" if self.dip is not None else 'N/A',
                'Azimuth (degrees)': f"{self.azimuth:0.0f}" if self.azimuth
                                                               is not None else 'N/A',
                'Response information': 'available' if self.response else 'not available'
            }

        ret = "\n".join([f"{key}: {value}" for key, value in attributes.items()])
        return ret

    def __eq__(self, other):
        if not super().__eq__(other):
            return False

        return (self.coordinates == other.coordinates and
                self.active == other.active and
                self.oriented == other.oriented)

    def set_orientation(self, orientation_vector):
        """
        set the Azimuth and Dip from an orientation vector assuming the
        orientation vector provided is east, north, up.
        :param self:
        :param orientation_vector:
        :return:

        Azimuth is defined from the north direction and positive clockwise
        Dip is defined as the angle from the horizontal plane and positive down
        """

        azimuth, dip = self.calculate_azimuth_and_dip(orientation_vector)
        self.azimuth = azimuth
        self.dip = dip

    def calculate_azimuth_and_dip(self, orientation_vector):
        """
        calculate the Azimuth and Dip from an orientation vector assuming the
        orientation vector provided is east, north, up.
        :param orientation_vector:
        :return:

        Azimuth is defined from the north direction and positive clockwise
        Dip is defined as the angle from the horizontal plane and positive down
        """

        if self.coordinates.coordinate_system == CoordinateSystem.ENU:

            east = orientation_vector[0]
            north = orientation_vector[1]
            up = orientation_vector[2]

        elif self.coordinates.coordinate_system == CoordinateSystem.NED:

            north = orientation_vector[0]
            east = orientation_vector[1]
            up = - orientation_vector[2]

        elif self.coordinates.coordinate_system == CoordinateSystem.NEU:
            north = orientation_vector[0]
            east = orientation_vector[1]
            up = orientation_vector[2]

        elif self.coordinates.coordinate_system == CoordinateSystem.END:
            east = orientation_vector[0]
            north = orientation_vector[1]
            up = - orientation_vector[2]

        horizontal_length = np.linalg.norm([east, north])

        azimuth = np.arctan2(east, north) * 180 / np.pi
        if azimuth < 0:
            azimuth = 360 + azimuth

        dip = np.arctan2(-up, horizontal_length) * 180 / np.pi

        return azimuth, dip

    @property
    def orientation_vector(self):
        """
        Computes the orientation vector based on the current azimuth and dip values.

        The method first converts the azimuth and dip from degrees to radians.
        It then calculates
        the components of the vector (up, east, north) based on trigonometric
        relationships. The
        final orientation vector is dependent on the coordinate system of the instance
        (either ENU or NED).

        For ENU (East-North-Up), the vector is returned as [east, north, up].
        For NED (North-East-Down), it is returned as [north, east, -up].

        Returns:
            numpy.ndarray: A 3-element array representing the orientation vector in the
            specified coordinate system.
        """

        up = -np.sin(self.dip * np.pi / 180)
        east = np.sin(self.azimuth * np.pi / 180) * \
               np.cos(self.dip * np.pi / 180)
        north = np.cos(self.azimuth * np.pi / 180) * \
                np.cos(self.dip * np.pi / 180)

        if self.coordinates.coordinate_system == CoordinateSystem.ENU:
            ov = np.array([east, north, up])

        elif self.coordinates.coordinate_system == CoordinateSystem.NED:
            ov = np.array([north, east, -up])

        elif self.coordinates.coordinate_system == CoordinateSystem.NEU:
            ov = np.array([north, east, up])
        elif self.coordinates.coordinate_system == CoordinateSystem.END:
            ov = np.array([east, north, -up])
        # else:
        #     raise ValueError('coordinate system not supported')
        return ov

    @property
    def coordinate_system(self):
        return self.coordinates.coordinate_system

    @property
    def x(self):
        return self.coordinates.x

    @property
    def y(self):
        return self.coordinates.y

    @property
    def z(self):
        return self.coordinates.z

    @property
    def loc(self):
        return np.array([self.x, self.y, self.z])


def load_from_excel(file_name) -> Inventory:
    """
    Read in a multi-sheet excel file with network metadata sheets:
        Locations, Networks, Hubs, Stations, Components, Locations, Cables,
        Boreholes
    Organize these into a uquake Inventory object

    :param xls_file: path to excel file
    :type: xls_file: str
    :return: inventory
    :rtype: uquake.core.data.inventory.Inventory

    """

    df_dict = pd.read_excel(file_name, sheet_name=None)

    source = df_dict['Locations'].iloc[0]['code']
    # sender (str, optional) Name of the institution sending this message.
    sender = df_dict['Locations'].iloc[0]['operator']
    net_code = df_dict['Networks'].iloc[0]['code']
    net_descriptions = df_dict['Networks'].iloc[0]['name']

    contact_name = df_dict['Networks'].iloc[0]['contact_name']
    contact_email = df_dict['Networks'].iloc[0]['contact_email']
    contact_phone = df_dict['Networks'].iloc[0]['contact_phone']
    site_operator = df_dict['Locations'].iloc[0]['operator']
    site_country = df_dict['Locations'].iloc[0]['country']
    site_name = df_dict['Locations'].iloc[0]['name']
    location_code = df_dict['Locations'].iloc[0]['code']

    print("source=%s" % source)
    print("sender=%s" % sender)
    print("net_code=%s" % net_code)

    network = Network(net_code)
    inventory = Inventory([network], source)

    # obspy requirements for PhoneNumber are super specific:
    # So likely this will raise an error if/when someone changes the value in
    # Networks.contact_phone
    """
    PhoneNumber(self, area_code, phone_number, country_code=None,
    description=None):
        :type area_code: int
        :param area_code: The area code.
        :type phone_number: str
        :param phone_number: The phone number minus the country and
        area code. Must be in the form "[0-9]+-[0-9]+", e.g. 1234-5678.
        :type country_code: int, optional
        :param country_code: The country code.
    """

    import re
    phone = re.findall(r"[\d']+", contact_phone)
    area_code = int(phone[0])
    number = "%s-%s" % (phone[1], phone[2])
    phone_number = PhoneNumber(area_code=area_code, phone_number=number)

    person = Person(names=[contact_name], agencies=[site_operator],
                    emails=[contact_email], phones=[phone_number])
    operator = Operator(site_operator, contacts=[person])
    location = Instrument(name=site_name, description=site_name,
                          country=site_country)

    # Merge Stations+Components+Locations+Cables info into sorted stations +
    # channels dicts:

    df_dict['Stations']['station_code'] = df_dict['Stations']['code']
    df_dict['Locations']['sensor_code'] = df_dict['Locations']['code']
    df_dict['Components']['code_channel'] = df_dict['Components']['code']
    df_dict['Components']['sensor'] = df_dict['Components']['sensor__code']
    df_merge = pd.merge(df_dict['Stations'], df_dict['Locations'],
                        left_on='code', right_on='station__code',
                        how='inner', suffixes=('', '_channel'))

    df_merge2 = pd.merge(df_merge, df_dict['Components'],
                         left_on='sensor_code', right_on='sensor__code',
                         how='inner', suffixes=('', '_sensor'))

    df_merge3 = pd.merge(df_merge2, df_dict['Cable types'],
                         left_on='cable__code', right_on='code',
                         how='inner', suffixes=('', '_cable'))

    df_merge4 = pd.merge(df_merge3, df_dict['Location types'],
                         left_on='sensor_type__model', right_on='model',
                         how='inner', suffixes=('', '_sensor_type'))

    df = df_merge4.sort_values(['sensor_code', 'location_code']).fillna(0)

    # Need to sort by unique station codes, then look through 1-3 channels
    # to add
    stn_codes = set(df['sensor_code'])
    stations = []

    for code in stn_codes:
        chan_rows = df.loc[df['sensor_code'] == code]
        row = chan_rows.iloc[0]
        station = {}
        # Set some keys explicitly
        #     from ipdb import set_trace; set_trace()
        station['code'] = '{}'.format(row['sensor_code'])
        station['x'] = row['location_x_channel']
        station['y'] = row['location_y_channel']
        station['z'] = row['location_z_channel']
        station['loc'] = np.array(
            [station['x'], station['y'], station['z']])
        station['long_name'] = "{}.{}.{:02d}".format(row['network__code'],
                                                     row['station_code'],
                                                     row['location_code'])

        # MTH: 2019/07 Seem to have moved from pF to F on Cables sheet:
        station['cable_capacitance_pF_per_meter'] = row['c'] * 1e12

        # Set the rest (minus empty fields) directly from spreadsheet names:
        renamed_keys = {'sensor_code', 'location_x', 'location_y',
                        'location_z', 'name'}

        # These keys are either redundant or specific to channel, not station:
        remove_keys = {'code', 'id_channel', 'orientation_x',
                       'orientation_y', 'orientation_z', 'id_sensor',
                       'enabled_channel', 'station_id', 'id_cable'}
        keys = row.keys()
        empty_keys = keys[pd.isna(row)]
        keys = set(keys) - set(empty_keys) - renamed_keys - remove_keys

        for key in keys:
            station[key] = row[key]

        # Added keys:
        station['motion'] = 'VELOCITY'

        if row['sensor_type'].upper() == 'ACCELEROMETER':
            station['motion'] = 'ACCELERATION'

        # Attach channels:
        station['channels'] = []

        for index, rr in chan_rows.iterrows():
            chan = {}
            chan['cmp'] = rr['code_channel_sensor'].upper()
            chan['orientation'] = np.array([rr['orientation_x'],
                                            rr['orientation_y'],
                                            rr['orientation_z']])
            chan['x'] = row['location_x_channel']
            chan['y'] = row['location_y_channel']
            chan['z'] = row['location_z_channel']
            chan['enabled'] = rr['enabled']
            station['channels'].append(chan)

        stations.append(station)

    # from ipdb import set_trace; set_trace()

    # Convert these station dicts to inventory.Station objects and attach to
    # inventory.network:
    station_list = []

    for station in stations:
        # This is where namespace is first employed:
        station = Station.from_station_dict(station, site_name)
        station.location = location
        station.operators = [operator]
        station_list.append(station)

    network.stations = station_list

    return inventory


@expand_input_format_compatibility
def read_inventory(path_or_file_object, format='STATIONXML',
                   xy_from_lat_lon=False, *args, **kwargs) -> Inventory:
    """
    Read inventory file
    :param path_or_file_object: the path to the inventory file or a file object
    :param format: the format
    :param xy_from_lat_lon: if True convert populate the XY field by converting
    the latitude and longitude to UTM
    :param args: see obspy.core.inventory.read_inventory for more information
    :param kwargs: see obspy.core.inventory.read_inventory for more information
    :return: an inventory object
    :rtype: ~uquake.core.inventory.Inventory
    """


    if type(path_or_file_object) is Path:
        path_or_file_object = str(path_or_file_object)

    # del kwargs['xy_from_lat_lon']

    if (format not in ENTRY_POINTS['inventory'].keys()) or \
            (format.upper() == 'STATIONXML'):

        obspy_inv = inventory.read_inventory(path_or_file_object, *args, format=format,
                                             **kwargs)

        return Inventory.from_obspy_inventory_object(
            obspy_inv, xy_from_lat_lon=xy_from_lat_lon)

    else:
        format_ep = ENTRY_POINTS['inventory'][format]

        read_format = load_entry_point(format_ep.dist.key,
                                       'obspy.io.%s' %
                                       format_ep.name, 'readFormat')

        return expand_input_format_compatibility(
            read_format(path_or_file_object, **kwargs))

    # else:


# def read_inventory(filename, format='STATIONXML', **kwargs):
#     if isinstance(filename, Path):
#         filename = str(filename)
#
#     if format in ENTRY_POINTS['inventory'].keys():
#         format_ep = ENTRY_POINTS['inventory'][format]
#         read_format = load_entry_point(format_ep.dist.key,
#                                        'obspy.plugin.inventory.%s' %
#                                        format_ep.name, 'readFormat')
#
#         return Inventory(inventory=read_format(filename, **kwargs))
#
#     else:
#         return inventory.read_inventory(filename, format=format,
#                                                **kwargs)


read_inventory.__doc__ = inventory.read_inventory.__doc__.replace(
    'obspy', ns)
