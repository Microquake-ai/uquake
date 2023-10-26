# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: event.py
#  Purpose: Expansion of the obspy.core.event module
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

import base64
import io

import numpy as np
import obspy.core.event as obsevent
from obspy.core.event import *
from obspy.core.event import ResourceIdentifier
from copy import deepcopy
from base64 import b64encode, b64decode
import pickle
from uquake.waveform.mag_utils import calc_static_stress_drop
from pathlib import Path
from uquake.core.coordinates import Coordinates, CoordinateSystem
from uquake.core.util.attribute_handler import (_set_attr_handler, pop_keys_matching,
                                                namespace, _init_handler)
import json
from typing import List
from enum import Enum


class RayCollection(list):

    def append(self, item):
        if not isinstance(item, Ray):
            raise TypeError("Only Ray objects can be appended to RayCollection")
        super().append(item)

    def __repr__(self):
        return f"RayCollection({super().__repr__()})"

    def to_json(self):
        out_list = []
        for ray in self:
            out_list.append(ray.to_json())
        return out_list

    @classmethod
    def from_json(cls, encoded_rays):
        out_list = cls()
        for encoded_ray in encoded_rays:
            out_list.append(Ray.from_json(encoded_ray))
        return out_list


class Phase(Enum):

    P = 'P'
    S = 'S'

    def __str__(self):
        return str(self.value)


class Ray(object):
    """
    Initializes a Ray object that represents a seismic raypath in a 3D medium.

    :param nodes: List of nodes defining the raypath geometry. Default is an empty list.
    :type nodes: list, optional
    :param waveform_id: Identifier for the associated waveform. Default is an empty
    WaveformStreamID.
    :type waveform_id: WaveformStreamID, optional
    :param arrival_id: Identifier for the associated seismic arrival. Default is None.
    :type arrival_id: ResourceIdentifier, optional
    :param phase: Type of seismic wave, either "P" for primary or "S" for secondary.
    Default is None.
    :type phase: str, optional
    :param travel_time: Travel time from the source to the observation location, in
    seconds.
    Default is None.
    :type travel_time: float, optional
    :param earth_model_id: Identifier for the Earth model used in ray tracing. Default is
    None.
    :type earth_model_id: ResourceIdentifier, optional
    :param coordinate_system: Coordinate system in which the ray nodes are defined.
    Default North East Down (NED).
    :type coordinate_system: CoordinateSystem, optional
    """

    def __init__(self, nodes: list,
                 waveform_id: WaveformStreamID = WaveformStreamID(),
                 arrival_id: ResourceIdentifier = None,
                 phase: str = 'P',
                 travel_time: float = None,
                 takeoff_angle: float = None,
                 azimuth: float = None,
                 velocity_model_id: ResourceIdentifier = None,
                 coordinate_system: CoordinateSystem = CoordinateSystem('NED')):

        self.nodes = np.array(nodes)
        self.waveform_id = waveform_id
        self.arrival_id = arrival_id
        self.phase = Phase(phase)
        self.travel_time = travel_time
        self.resource_id = obsevent.ResourceIdentifier()
        self.velocity_model_id = velocity_model_id
        self.coordinate_system = coordinate_system
        self.takeoff_angle = takeoff_angle
        self.azimuth = azimuth

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        # Compare nodes (assuming the same shape and numerical values)
        if not np.array_equal(self.nodes, other.nodes):
            return False

        # Compare other attributes
        if self.arrival_id != other.arrival_id or \
                self.phase != other.phase or \
                self.travel_time != other.travel_time or \
                self.takeoff_angle != other.takeoff_angle or \
                self.azimuth != other.azimuth or \
                self.velocity_model_id != other.velocity_model_id or \
                self.coordinate_system != other.coordinate_system or \
                self.waveform_id != other.waveform_id:
            return False

        return True

    def __setattr__(self, key, value):
        if key == 'phase':
            if value is None:
                self.__dict__[key] = value
            else:
                self.__dict__[key] = value
        else:
            self.__dict__[key] = value

    @property
    def length(self):
        if len(self.nodes) < 2:
            return 0

        length = 0
        for k, node1 in enumerate(self.nodes[0:-1]):
            node2 = self.nodes[k + 1]
            length += np.linalg.norm(node1 - node2)

        return length

    @property
    def toa(self):
        return self.takeoff_angle

    @property
    def az(self):
        return self.azimuth

    @property
    def location(self):
        return self.waveform_id.location if self.waveform_id is not None else None

    @property
    def station(self):
        return self.waveform_id.station_code

    @property
    def location(self):
        return self.waveform_id.location_code

    @property
    def network(self):
        return self.waveform_id.network_code

    @property
    def back_azimuth(self):
        self.baz

    @property
    def earth_model_id(self):
        return self.velocity_model_id

    def _vector(self, index1, index2):
        """Get the vector between two nodes based on coordinate system."""
        v = self.nodes[index2] - self.nodes[index1]
        if self.coordinate_system == CoordinateSystem.NED:
            return np.array([v[0], v[1], -v[2]])
        return v

    @property
    def baz(self):
        """Back-azimuth in degrees."""
        if len(self.nodes) < 2:
            return None
        v = self._vector(-2, -1)
        baz = np.arctan2(v[1], v[0]) * 180 / np.pi
        return (baz + 360) % 360

    @property
    def incidence_angle(self):
        """Incidence angle in degrees."""
        if len(self.nodes) < 2:
            return None
        v = self._vector(-2, -1)
        h = np.sqrt(v[0] ** 2 + v[1] ** 2)
        ia = np.arctan2(h, v[2]) * 180 / np.pi
        return ia

    def to_pick(self, origin_time):
        time = origin_time + self.travel_time
        waveform_id = WaveformStreamID(network_code=self.network,
                                       station_code=self.station,
                                       location_code=self.location)

        method_id = ResourceIdentifier('predicted from rays')
        return Pick(time=time, waveform_id=waveform_id, method_id=method_id,
                    back_azimuth=self.back_azimuth, phase_hint=self.phase,
                    evaluation_mode='automatic',
                    evaluation_status='preliminary')

    def __len__(self):
        return len(self.nodes)

    def __str__(self):

        stream_code = f'{self.waveform_id.network_code}.' \
                      f'{self.waveform_id.station_code}.' \
                      f'{self.waveform_id.location_code}'

        txt = \
            f"""
         stream code: {stream_code}
          arrival id: {self.arrival_id}
               phase: {self.phase}
          length (m): {self.length}
     number of nodes: {len(self.nodes)}
            """
        return txt

    def __repr__(self):
        return self.__str__()

    def to_json(self):
        obj_dict = {
            'nodes': self.nodes.tolist(),
            'waveform_id': {
                'network_code': self.waveform_id.network_code,
                'station_code': self.waveform_id.station_code,
                'channel_code': self.waveform_id.channel_code,
                'location_code': self.waveform_id.location_code,
                'resource_uri': str(self.waveform_id.resource_uri)
            },
            'arrival_id': str(self.arrival_id),
            'phase': str(self.phase),
            'travel_time': self.travel_time,
            'resource_id': str(self.resource_id),
            'velocity_model_id': str(self.velocity_model_id),
            'coordinate_system': str(self.coordinate_system),
            'takeoff_angle': self.takeoff_angle,
            'azimuth': self.azimuth
        }
        return json.dumps(obj_dict)

    @classmethod
    def from_json(cls, json_str):
        obj_dict = json.loads(json_str)
        nodes = np.array(obj_dict['nodes'])
        waveform_id_dict = obj_dict['waveform_id']
        waveform_id = WaveformStreamID(network_code=waveform_id_dict['network_code'],
                                       station_code=waveform_id_dict['station_code'],
                                       channel_code=waveform_id_dict['channel_code'],
                                       location_code=waveform_id_dict['location_code'],
                                       resource_uri=ResourceIdentifier(
                                           waveform_id_dict['resource_uri']))
        arrival_id = ResourceIdentifier(obj_dict['arrival_id'])
        phase = Phase(obj_dict['phase'])
        travel_time = obj_dict['travel_time']
        velocity_model_id = ResourceIdentifier(obj_dict['velocity_model_id'])
        coordinate_system = CoordinateSystem(obj_dict['coordinate_system'])
        takeoff_angle = obj_dict['takeoff_angle']
        azimuth = obj_dict['azimuth']

        return cls(nodes, waveform_id, arrival_id, phase, travel_time,
                   takeoff_angle, azimuth, velocity_model_id, coordinate_system)


class Catalog(obsevent.Catalog):
    extra_keys = []

    __doc__ = obsevent.Catalog.__doc__.replace('obspy', 'uquake')

    def __init__(self, obspy_obj=None, **kwargs):
        # _init_handler(self, obspy_obj, **kwargs)
        if obspy_obj and len(kwargs) > 0:
            raise AttributeError("Initialize from either \
                                  obspy_obj or kwargs, not both")

        if obspy_obj:

            for key in obspy_obj.__dict__.keys():
                if key == 'events':
                    event_type_lookup = EventTypeLookup()
                    events = []
                    for event in obspy_obj:
                        uquake_event = Event(obspy_obj=event)
                        if event_type_lookup.is_valid_quakeml(uquake_event.event_type):
                            uquake_event.event_type = \
                                event_type_lookup.inverse_lookup_table[event.event_type]
                        events.append(uquake_event)
                    self.events = events
                else:
                    self.__dict__[key] = obspy_obj.__dict__[key]

        else:
            super(type(self), self).__init__(
                **kwargs)

    def __setattr__(self, name, value):
        super(type(self), self).__setattr__(name, value)

    def write(self, fileobj, format='quakeml', **kwargs):
        event_type_lookup = EventTypeLookup()
        type_lookup = event_type_lookup.lookup_table
        for event in self.events:
            if event_type_lookup.is_valid_uquakeml(event.event_type):
                event.event_type = type_lookup[event.event_type]
            for ori in event.origins:
                for ar in ori.arrivals:
                    if 'extra' in ar.keys():
                        del ar.extra

        result = obsevent.Catalog.write(self, fileobj, format=format, **kwargs)

        for event in self.events:
            for ori in event.origins:
                ars = []
                for ar in ori.arrivals:
                    ars.append(Arrival(obspy_obj=ar))
                ori.arrivals = ars

        return result

    def duplicate(self):
        """
        this function duplicates a catalog object, this function does not
        duplicate picks. It creates an object containing multiple event
        containing each one origin and one magnitude.
        :return: a new catalog object
        """

        new_events = []
        for event in self.events:
            if len(event.origins) == 0:
                new_origins = []
                preferred_origin_id = None

            else:
                new_origins = [event.preferred_origin() or event.origins[-1]]

                preferred_origin_id = ResourceIdentifier()
                new_origins[0].resource_id = preferred_origin_id
                new_origins[0].arrivals = []
                new_rays = []
                for ray in new_origins[0].rays:
                    ray.resource_id = ResourceIdentifier()
                    new_rays.append(ray)
                new_origins[0].rays = new_rays

            if len(event.magnitudes) == 0:
                new_magnitudes = []
                preferred_magnitude_id = None
            else:
                new_magnitudes = [event.preferred_magnitude()
                                  or event.magnitudes[-1]]

                preferred_magnitude_id = ResourceIdentifier()
                new_magnitudes[0].resource_id = preferred_magnitude_id
                new_magnitudes[0].origin_id = preferred_origin_id

            new_event = Event(origins=new_origins, magnitudes=new_magnitudes)
            new_event.preferred_origin_id = preferred_origin_id
            new_event.preferred_magnitude_id = preferred_magnitude_id

            new_events.append(new_event)

            return Catalog(events=new_events)

    def copy(self):
        return deepcopy(self)


class EventTypeLookup(object):

    def __init__(self):
        self.lookup_table = {'earthquake/large event': 'earthquake',
                             'seismic event': 'induced or triggered event',
                             'offsite event': 'atmospheric event',
                             'rock burst': 'rock burst',
                             'fall of ground/rockfall': 'cavity collapse',
                             'blast': 'explosion',
                             'blast sequence': 'accidental explosion',
                             'development blast': 'industrial explosion',
                             'production blast': 'mining explosion',
                             'far away blast/open pit blast': 'quarry blast',
                             'offsite blast': 'nuclear explosion',
                             'paste firing': 'chemical explosion',
                             'calibration blast': 'controlled explosion',
                             'other blast/slashing': 'experimental explosion',
                             'mid-shift blast/slash blast': 'industrial explosion',
                             'raise bore': 'hydroacoustic event',
                             'crusher noise': 'road cut',
                             'orepass noise': 'collapse',
                             'drilling noise': 'acoustic noise',
                             'electrical noise': 'thunder',
                             'scaling noise': 'anthropogenic event',
                             'mechanical noise': 'crash',
                             'test pulse': 'sonic boom',
                             'unidentified noise/other noise': 'other event',
                             'duplicate': 'boat crash',
                             'unknown': 'plane crash',
                             'tap test/test': 'avalanche'}

    @property
    def inverse_lookup_table(self):
        inverse_lookup_table = {}
        lookup_table = self.lookup_table
        for key in lookup_table.keys():
            inverse_lookup_table[lookup_table[key]] = key

        return inverse_lookup_table

    def convert_from_quakeml(self, quakeml_type):
        return self.lookup_table[quakeml_type]

    def convert_to_quakeml(self, uquakem_type):
        return self.inverse_lookup_table[uquakem_type]

    @property
    def valid_quakeml_types(self):
        return self.inverse_lookup_table.keys()

    @property
    def valid_uquakeml_types(self):
        return self.lookup_table.keys()

    def is_valid_quakeml(self, event_type):
        return event_type in self.valid_quakeml_types

    def is_valid_uquakeml(self, event_type):
        return event_type in self.valid_uquakeml_types


class Event(obsevent.Event):

    # _format keyword is actualy a missing obspy default

    __doc__ = obsevent.Event.__doc__.replace('obspy', 'uquake')

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)

    def __setattr__(self, name, value):
        _set_attr_handler(self, name, value)

    def __str__(self):
        return "Event:\t%s\n\n%s" % (
            self.short_str(),
            "\n".join(super(Event, self).__str__().split("\n")[1:]))

    def short_str(self):
        out = ''

        if self.origins:
            og = self.preferred_origin() or self.origins[-1]

            if og.x is None:
                x = 0
            else:
                x = og.x
            if og.y is None:
                y = 0
            else:
                y = og.y
            if og.z is None:
                z = 0
            else:
                z = og.z

            out += '%s | %0.2f, %0.2f, %0.2f | %s' % (og.time,
                                                      x, y, z,
                                                      og.evaluation_mode)

        if self.magnitudes:
            magnitude = self.preferred_magnitude() or self.magnitudes[-1]
            out += ' | %0.2f %-2s' % (magnitude.mag,
                                      magnitude.magnitude_type)

        return out

        self.picks += picks

    def write(self, fileobj, **kwargs):
        for ori in self.origins:
            arrivals = []
            for ar in ori.arrivals:
                if 'extra' in ar.keys():
                    del ar.extra

        return obsevent.Event.write(self, fileobj, **kwargs)

    def plot_focal_mechanism(self):
        pass

    def append_origin_as_preferred_origin(self, new_origin: Origin):
        self.origins.append(new_origin)
        self.preferred_origin_id = new_origin.resource_id


class UncertaintyPointCloud(object):
    def __init__(self, locations: List[float], probabilities: List[float],
                 coordinate_system: CoordinateSystem = CoordinateSystem('NED')):
        if isinstance(locations, np.ndarray):
            locations = locations.tolist()
        if isinstance(probabilities, np.ndarray):
            probabilities = probabilities.tolist()
        self.locations = locations
        self.probabilities = probabilities
        self.coordinate_system = coordinate_system

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.probabilities != other.probabilities:
            return False
        if self.locations != other.locations:
            return False
        if self.coordinate_system != other.coordinate_system:
            return False

        return True

    def to_json(self):
        out_dict = {}
        for key in self.__dict__.keys():
            if key == 'coordinate_system':
                out_dict[key] = str(self.coordinate_system)
            else:
                out_dict[key] = self.__dict__[key]

        return json.dumps(out_dict)

    @classmethod
    def from_json(cls, json_string):
        in_dict = json.loads(json_string)
        if 'coordinate_system' in in_dict.keys():
            in_dict['coordinate_system'] = CoordinateSystem(
                in_dict['coordinate_system'])
        return cls(**in_dict)


class Origin(obsevent.Origin):
    original_doc = obsevent.Origin.__doc__.replace('obspy', 'uquake')
    original_doc = original_doc.replace('earth_model_id', 'velocity_model_id')

    # Locate the latitude section
    start_idx = original_doc.find(':type latitude:')
    end_idx = original_doc.find(':type', original_doc.find(
        ':param latitude:'))  # Locate the next ':type' after latitude

    # Define the new 'coordinates' parameter section
    new_section = """:type coordinates: :class:`~uquake.core.coordinates.Coordinates`
    :param coordinates: Spatial coordinates of the event origin.
    """

    # Replace the latitude section with the new 'coordinates' section
    original_doc = original_doc[:start_idx] + new_section + original_doc[end_idx:]

    # Remove unwanted sections
    sections_to_remove = ['latitude', 'longitude', 'longitude_error', 'depth',
                          'depth_error', 'depth_type', 'elevation',
                          'epicenter_fixed']
    for section in sections_to_remove:
        start_idx = original_doc.find(f':type {section}')
        end_idx = original_doc.find(':type', original_doc.find(
            f':param {section}') + 1)  # Locate the next ':type' after section

        if start_idx != -1:  # Only remove if the section exists
            original_doc = original_doc[:start_idx] + original_doc[end_idx:]

    __doc__ = original_doc

    extra_keys = ['rays', 'coordinates', 'uncertainty_point_cloud']

    def __init__(self, coordinates: Coordinates = Coordinates(0, 0, 0),
                 rays: RayCollection = [],
                 uncertainty_point_cloud: UncertaintyPointCloud = None,
                 obspy_obj=None, **kwargs):

        _init_handler(self, obspy_obj, **kwargs)

        if 'velocity_model_id' in kwargs.keys():
            kwargs['earth_model_id'] = kwargs.pop('velocity_model_id')
        if coordinates:
            self.coordinates = coordinates
        if rays:
            self.rays = rays
        if uncertainty_point_cloud is not None:
            self.uncertainty_point_cloud = uncertainty_point_cloud

    def __setattr__(self, name, value):
        _set_attr_handler(self, name, value)

    def __getattr__(self, item):
        if item == 'coordinates':
            return Coordinates.from_json(self.__dict__['extra'][item])
        elif item == 'rays':
            return RayCollection.from_json(self.__dict__['extra'][item])
        elif item == 'uncertainty_point_cloud':
            return UncertaintyPointCloud.from_json(self.__dict__['extra'][item])
        else:
            return self.__dict__[item]

    @staticmethod
    def __encode_rays__(rays: List):
        encoded_rays = []
        for ray in rays:
            encoded_rays.append(ray.to_json())
        return encoded_rays

    def __decode_rays__(self):
        decoded_rays = []
        if self.__encoded_rays__ is None:
            return
        for encoded_ray in self.__encoded_rays__:
            decoded_rays.append(Ray.from_json(encoded_ray))

        return decoded_rays

    @staticmethod
    def __encode_uncertainty_point_cloud__(
            uncertainty_point_cloud: UncertaintyPointCloud):
        return uncertainty_point_cloud.to_json()

    def __decode_uncertainty_point_cloud__(self):
        return UncertaintyPointCloud.from_json(self.__encoded_uncertainty_point_cloud__)

    @staticmethod
    def __encode_coordinates__(value):
        if not isinstance(value, Coordinates):
            raise TypeError(f'the value provided must be of instance '
                            f'{type(Coordinates)}')

        return value.to_json()

    def __decode_coordinates__(self):
        if self.__cartesian_coordinates__ is None:
            return
        else:
            return Coordinates.from_json(self.__cartesian_coordinates__)

    def get_arrival_id(self, phase, station_code):
        arrival_id = None
        for arrival in self.arrivals:
            if (arrival.phase == phase) and (arrival.get_sta() ==
                                             station_code):
                arrival_id = arrival.resource_id

        return arrival_id

    def append_ray(self, item):
        if self.rays is None:
            self.rays = [item]
        else:
            self.rays = self.rays + [item]

    @property
    def rms_residual(self):
        if len(self.arrivals) == 0:
            return None
        residuals = [arrival.time_residual ** 2 for arrival in self.arrivals]
        return np.sqrt(np.mean(residuals))

    @property
    def loc(self):
        return np.array([self.x, self.y, self.z])

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
    def easting(self):
        return self.coordinates.easting

    @property
    def northing(self):
        return self.coordinates.northing

    @property
    def down(self):
        return self.coordinates.down

    @property
    def depth(self):
        return self.coordinates.down

    @property
    def up(self):
        return self.coordinates.up

    @property
    def elevation(self):
        return self.coordinates.up

    @property
    def coordinate_system(self):
        return self.coordinates.coordinate_system

    @property
    def location_uncertainty(self):
        if self.origin_uncertainty is None:
            return None
        else:
            if self.origin_uncertainty.confidence_ellipsoid is None:
                return None
            return self.origin_uncertainty.confidence_ellipsoid \
                .semi_major_axis_length

    def get_origin(self):
        if self.preferred_origin_id is not None:
            return self.preferred_origin_id.get_referred_object()

    def get_all_magnitudes_for_origin(self, cat):
        magnitudes = []

        for event in cat:
            for mag in event.magnitudes:
                if mag.origin_id.id == self.resource_id.id:
                    magnitudes.append(mag)

        return magnitudes

    def __str__(self, **kwargs):
        string = f"""
       resource_id: {self.resource_id}
         time(UTC): {self.time}
                 x: {self.x}
                 y: {self.y}
                 z: {self.z}
       uncertainty: {self.location_uncertainty} (m)
   evaluation_mode: {self.evaluation_mode}
 evaluation_status: {self.evaluation_status}
                ---------
          arrivals: {len(self.arrivals):d}
        """
        return string

    def get_incidence_baz_angles(self, site_code, phase):
        baz = None
        inc = None
        for ray in self.rays:
            if (ray.site_code == site_code) and (ray.phase == phase):
                baz = ray.back_azimuth
                inc = ray.incidence_angle
                break
        return baz, inc

    def get_ray_station_phase(self, site_code, phase):
        out_ray = None
        for ray in self.rays:
            if (ray.site_code == site_code) and (ray.phase == phase):
                out_ray = ray
                break
        return out_ray

    def distance_station(self, site_code, phase='P'):
        ray = self.get_ray_station_phase(self, site_code, phase)
        if ray is None:
            return None

        return ray.length


class Magnitude(obsevent.Magnitude):
    __doc__ = obsevent.Magnitude.__doc__.replace('obspy', 'uquake')

    extra_keys = ['corner_frequency',
                  'corner_frequency_p', 'corner_frequency_s',
                  'corner_frequency_error', 'corner_frequency_p_error',
                  'corner_frequency_s_error'
                  'moment_magnitude', 'moment_magnitude_uncertainty']

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)

    def __setattr__(self, name, value):
        _set_attr_handler(self, name, value)

    @classmethod
    def from_dict(cls, input_dict, origin_id=None, obspy_obj=None, **kwargs):
        out_cls = cls(obspy_obj=obspy_obj, **input_dict,
                      origin_id=origin_id, **kwargs)
        out_cls.mag = input_dict['moment_magnitude']
        out_cls.magnitude_type = 'Mw'

        return out_cls

    @property
    def static_stress_drop_mpa(self):
        ssd = None
        if self.magnitude_type == "Mw":
            if self.mag and self.corner_frequency_hz:
                ssd = calc_static_stress_drop(self.mag,
                                              self.corner_frequency_hz)
        return ssd

    @property
    def apparent_stress(self):
        app_stress = None
        if self.magnitude_type == 'Mw':
            if self.energy_joule and self.mag:
                app_stress = 2 * self.energy_joule / self.potency_m3
        return app_stress

    @property
    def seismic_moment(self):
        seismic_moment = None
        if self.magnitude_type == 'Mw':
            seismic_moment = 10 ** (3 * (self.mag + 6.03) / 2)
        return seismic_moment

    # @seismic_moment.setter
    # def seismic_moment(self, val):
    #     self.mag = val
    #     self.magnitude_type = 'Mw'

    @property
    def potency_m3(self):
        potency = None
        mu = 29.5e9
        if self.magnitude_type == 'Mw':
            if self.mag:
                potency = self.seismic_moment / mu
        return potency

    def __str__(self, **kwargs):

        es_ep = None
        if self.energy_p_joule and self.energy_s_joule:
            es_ep = self.energy_s_joule / self.energy_p_joule

        string = f"""
             resource_id: {self.resource_id.id}     
               Magnitude: {self.mag}
          Magnitude type: {self.magnitude_type}
   Corner frequency (Hz): {self.corner_frequency_hz}
 Radiated Energy (joule): {self.energy_joule}
                   Es/Ep: {es_ep}
          Seismic moment: {self.seismic_moment}
       Source volume(m3): {self.potency_m3}
Static stress drop (MPa): {self.static_stress_drop_mpa}
     Apparent stress(Pa): {self.apparent_stress}
         evaluation mode: {self.evaluation_mode}
       evaluation status: {self.evaluation_status}
        """

        return string


class Pick(obsevent.Pick):
    original_doc = obsevent.Pick.__doc__.replace('obspy', 'uquake')

    insert_position = original_doc.find(".. note::")

    new_doc_addition = """
        :type snr: float, optional
        :param snr: Signal-to-Noise Ratio.
        :type planarity: float, optional
        :param planarity: Measure of planarity of the waveform.
        :type linearity: float, optional
        :param linearity: Measure of linearity of the waveform.

        **Extensions to original obsevent.Pick class above**
    """

    # Insert new documentation at the identified position
    new_doc = original_doc[:insert_position] + new_doc_addition + original_doc[
                                                                  insert_position:]

    extra_keys = ['snr', 'planarity', 'linearity']

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)

    def __setattr__(self, name, value):
        _set_attr_handler(self, name, value)

    def __str__(self, **kwargs):
        string = """
         stream_id: %s
              time: %s
             phase: %s
               snr: %s
   evaluation_mode: %s
 evaluation_status: %s
       resource_id: %s
        """ \
                 % (self.stream_id, self.time.strftime("%Y/%m/%d %H:%M:%S.%f"),
                    self.phase_hint,  self.snr,
                    self.evaluation_mode,
                    self.evaluation_status, self.resource_id)
        return string

    def get_sta(self):
        if self.waveform_id is not None:
            return self.waveform_id.station_code

    @property
    def location(self):
        if self.waveform_id is not None:
            location = f'{self.waveform_id.station_code}.' \
                       f'{self.waveform_id.location_code}'
            return location

    @property
    def stream_id(self):
        return self.waveform_id.stream_id


class WaveformStreamID(obsevent.WaveformStreamID):
    __doc__ = obsevent.WaveformStreamID.__doc__.replace('obspy', 'uquake')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def location(self):
        return self.location_code

    @property
    def stream_id(self):
        return f'{self.network_code}.{self.station_code}.{self.location_code}.' \
               f'{self.channel_code}'


class Arrival(obsevent.Arrival):
    __doc__ = obsevent.Arrival.__doc__.replace('obspy', 'uquake')
    __doc__ = __doc__.replace(':param distance: Epicentral distance. Unit: deg',
                              ':param distance: Hypocentral distance. Unit: m')
    __doc__ = __doc__.replace(':param earth_model_id: Earth model',
                              ':param velocity_model_id: Velocity model')

    __doc__ = __doc__.replace(':type earth_model_id:',
                              ':type velocity_model_id:')

    extra_keys = []

    def __init__(self, obspy_obj=None, **kwargs):
        if 'velocity_model_id' in kwargs.keys():
            velocity_model_id = kwargs['velocity_model_id']
            kwargs.pop('velocity_model_id')
            kwargs['earth_model_id'] = velocity_model_id
        _init_handler(self, obspy_obj, **kwargs)

    def __setattr__(self, name, value):
        if name == 'velocity_model_id':
            self.earth_model_id = value
        super().__setattr__(name, value)

    def __repr__(self):
        out_str = f"""
       resource_id: {self.resource_id}
           pick_id: {self.get_pick().resource_id}
             phase: {self.phase}
           azimuth: {self.azimuth} (deg)
          distance: {self.distance} (m)
     takeoff_angle: {self.takeoff_angle} (deg)
     time_residual: {self.time_residual * 1000} (ms)
       time_weight: {self.time_weight}
        """
        return out_str

    @staticmethod
    def calculate_time_residual(observed: float, predicted: float):
        return observed - predicted

    @property
    def polarity(self):
        return self.get_pick().polarity

    @property
    def pick(self):
        return self.get_pick()

    @property
    def time(self):
        return self.pick.time

    @property
    def channel(self):
        return self.pick.waveform_id.channel_code

    @property
    def location(self):
        return self.pick.waveform_id.location_code

    @property
    def station(self):
        return self.pick.waveform_id.station_code

    @property
    def network(self):
        return self.pick.waveform_id.network_code

    @property
    def velocity_model_id(self):
        return self.earth_model_id

    def get_pick(self):
        if self.pick_id is not None:
            return self.pick_id.get_referred_object()

    def get_sta(self):
        if self.pick_id is not None:
            pick = self.pick_id.get_referred_object()
            return pick.get_sta()

    def get_station(self):
        return self.get_sta()


def get_arrival_from_pick(arrivals, pick):
    """
      return arrival corresponding to pick

      :param arrivals: list of arrivals
      :type arrivals: list of either obspy.core.event.origin.Arrival
                      or uquake.core.event.origin.Arrival
      :param pick: P or S pick
      :type pick: either obspy.core.event.origin.Pick
                      or uquake.core.event.origin.Pick
      :return arrival
      :rtype: obspy.core.event.origin.Arrival or
              uquake.core.event.origin.Arrival
    """

    arrival = None
    for arr in arrivals:
        if arr.pick_id == pick.resource_id:
            arrival = arr
            break

    return arrival


def read_events(filename, **kwargs):
    if isinstance(filename, Path):
        filename = str(filename)
    # converting the obspy object into uquake objects

    cat = obsevent.read_events(filename, **kwargs)
    mq_catalog = Catalog(obspy_obj=cat)

    return mq_catalog


def _init_from_obspy_object(uquake_obj, obspy_obj):
    """
    When initializing uquake object from obspy_obj
    checks attributes for lists of obspy objects and
    converts them to equivalent uquake objects.
    """

    class_equiv = {obsevent.event: Event,
                   obsevent.Pick: Pick,
                   obsevent.Arrival: Arrival,
                   obsevent.Origin: Origin,
                   obsevent.Magnitude: Magnitude,
                   obsevent.WaveformStreamID: WaveformStreamID}

    for key, val in obspy_obj.__dict__.items():
        itype = type(val)
        if itype in class_equiv:
            uquake_obj.__setattr__(key, class_equiv[itype](val))

        elif itype == list:
            out = []
            for item in val:
                itype = type(item)
                if itype in class_equiv:
                    out.append(class_equiv[itype](item))
                else:
                    out.append(item)
            uquake_obj.__setattr__(key, out)
        else:
            uquake_obj.__setattr__(key, val)


def break_down(event):
    origin = event.origins[0]
    print("break_down: Here's what obspy reads:")
    print(origin)
    print("origin res id: %s" % origin.resource_id.id)
    print("id(origin): %s" % id(origin))
    print("id(origin.resource_id):%s" % id(origin.resource_id))
    ref_obj = origin.resource_id.get_referred_object()
    print("id(ref_obj):%s" % id(ref_obj))

    return


# MTH: this could(should?) be moved to waveforms/pick.py ??
def make_pick(time, phase='P', wave_data=None, snr=None, mode='automatic',
              status='preliminary', method_string=None, resource_id=None):
    this_pick = Pick()
    this_pick.time = time
    this_pick.phase_hint = phase
    this_pick.evaluation_mode = mode
    this_pick.evaluation_status = status

    this_pick.method = method_string
    this_pick.snr = snr

    if wave_data is not None:
        this_pick.waveform_id = WaveformStreamID(
            network_code=wave_data.stats.network,
            station_code=wave_data.stats.station,
            location_code=wave_data.stats.location,
            channel_code=wave_data.stats.channel)

        this_pick.trace_id = wave_data.get_id()

    return this_pick
