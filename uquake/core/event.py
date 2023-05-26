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
import warnings

import numpy as np
import obspy.core.event as obsevent
from obspy.core.event import *
from obspy.core.util import AttribDict
from copy import deepcopy
from base64 import b64encode, b64decode
from io import BytesIO
import pickle
from uquake.waveform.mag_utils import calc_static_stress_drop
from .logging import logger
from pathlib import Path

debug = False


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
                             'low quality event': 'cavity collapse',
                             'rock burst': 'rock burst',
                             'fall of ground/rockfall': 'mine collapse',
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
    extra_keys = ['_format', 'ACCEPTED', 'ASSOC_SEISMOGRAM_NAMES',
                  'AUTO_PROCESSED',
                  'BLAST', 'CORNER_FREQUENCY', 'DYNAMIC_STRESS_DROP',
                  'ENERGY', 'ENERGY_P', 'ENERGY_S', 'EVENT_MODIFICATION_TIME',
                  'EVENT_NAME', 'EVENT_TIME_FORMATTED', 'EVENT_TIME_NANOS',
                  'LOCAL_MAGNITUDE', 'LOCATION_RESIDUAL', 'LOCATION_X',
                  'LOCATION_Y', 'LOCATION_Z', 'MANUALLY_PROCESSED',
                  'NUM_ACCEPTED_TRIGGERS', 'NUM_TRIGGERS', 'POTENCY',
                  'POTENCY_P', 'POTENCY_S', 'STATIC_STRESS_DROP', 'TAP_TEST',
                  'TEST', 'TRIGGERED_SITES', 'USER_NAME', 'network']

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


class Origin(obsevent.Origin):
    __doc__ = obsevent.Origin.__doc__.replace('obspy', 'uquake')
    extra_keys = ['x', 'y', 'z', 'x_error', 'y_error', 'z_error', 'scatter',
                  '__encoded_rays__', 'author']

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)

    def __setattr__(self, name, value):

        if name == 'rays':
            self.__encoded_rays__ = self.__encode_rays__(self, value)
        # elif name == 'encoded_rays':
        #     self.__encoded_rays__ = value
        else:
            _set_attr_handler(self, name, value)

    def __getattr__(self, item):
        if item == 'rays':
            try:
                return self.__decode_rays__(self)
            except pickle.UnpicklingError:
                self.__encoded_rays__ = eval(self.__encoded_rays__)
                return self.__decode_rays__(self)

        else:
            return self.__dict__[item]

    @staticmethod
    def __encode_rays__(self, rays):
        return b64encode(pickle.dumps(rays))

    @staticmethod
    def __decode_rays__(self):
        if self.__encoded_rays__ is None:
            return
        try:
            return pickle.loads(b64decode(self.__encoded_rays__))
        except Exception as e:
            self.__encoded_rays__ = eval(self.__encoded_rays__)
            return pickle.loads(b64decode(self.__encoded_rays__))

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

    extra_keys = ['energy_joule', 'energy_p_joule', 'energy_p_std',
                  'energy_s_joule', 'energy_s_std', 'corner_frequency_hz',
                  'corner_frequency_p_hz', 'corner_frequency_s_hz',
                  'corner_frequency_error',
                  'time_domain_moment_magnitude',
                  'frequency_domain_moment_magnitude',
                  'moment_magnitude', 'moment_magnitude_uncertainty',
                  'seismic_moment', 'potency_m3', 'source_volume_m3',
                  'apparent_stress', 'static_stress_drop_mpa',
                  'quick_magnitude', 'error']

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
    __doc__ = obsevent.Pick.__doc__.replace('obspy', 'uquake')
    extra_keys = ['method', 'snr', 'trace_id', 'author', 'linearity',
                  'planarity', 'azimuth', 'dip']

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)
        if obspy_obj:
            wid = self.waveform_id
            self.trace_id = "%s.%s.%s.%s" % (wid.network_code,
                                             wid.station_code,
                                             wid.location_code,
                                             wid.channel_code)

    def __setattr__(self, name, value):
        _set_attr_handler(self, name, value)

    def __str__(self, **kwargs):
        string = """
          trace_id: %s
              time: %s
             phase:[%s]
            method: %s [%s]
   evaluation_mode: %s
 evaluation_status: %s
       resource_id: %s
        """ \
                 % (self.trace_id, self.time.strftime("%Y/%m/%d %H:%M:%S.%f"),
                    self.phase_hint, self.method, self.snr,
                    self.evaluation_mode,
                    self.evaluation_status, self.resource_id)
        return string

    def get_sta(self):
        if self.waveform_id is not None:
            return self.waveform_id.station_code

    @property
    def site(self):
        if self.waveform_id is not None:
            site = f'{self.waveform_id.station_code}.{self.waveform_id.location_code}'
            return site

    @property
    def site_code(self):
        return self.site


class WaveformStreamID(obsevent.WaveformStreamID):
    __doc__ = obsevent.WaveformStreamID.__doc__.replace('obspy', 'uquake')
    extra_keys = []

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)

    @property
    def site_code(self):
        if self.station_code is not None:
            if self.location_code is None:
                return self.station_code
            else:
                return f'{self.station_code}.{self.location_code}'

        return

    @property
    def site(self):
        return self.site_code


class Arrival(obsevent.Arrival):
    __doc__ = obsevent.Arrival.__doc__.replace('obspy', 'uquake')

    extra_keys = ['ray', 'backazimuth', 'inc_angle',
                  'peak_vel', 'tpeak_vel', 't1', 't2', 'pulse_snr',
                  'peak_dis', 'tpeak_dis', 'max_dis', 'tmax_dis',
                  'dis_pulse_width', 'dis_pulse_area', 'smom', 'fit',
                  'tstar', 'hypo_dist_in_m', 'vel_flux', 'vel_flux_Q',
                  'energy', 'fmin', 'fmax', 'traces']

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)

    def __setattr__(self, name, value):
        if name == 'polarity':
            if value in ['positive', 'negative', 'undecidable']:
                self.get_pick().polarity = value
            elif value == -1:
                self.get_pick().polarity = 'negative'
            elif value == 1:
                self.get_pick().polarity = 'positive'
            else:
                self.get_pick().polarity = 'undecidable'

        else:
            _set_attr_handler(self, name, value)

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
        if self.get_pick().polarity == 'positive':
            return 1.0
        elif self.get_pick().polarity == 'negative':
            return -1.0
        else:
            return None

    @property
    def pick(self):
        return self.get_pick()

    @property
    def site_code(self):
        return self.pick.site

    @property
    def site(self):
        return self.pick.site

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

    def get_pick(self):
        if self.pick_id is not None:
            return self.pick_id.get_referred_object()

    def get_sta(self):
        if self.pick_id is not None:
            pick = self.pick_id.get_referred_object()
            return pick.get_sta()


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


def _init_handler(self, obspy_obj, **kwargs):
    """
    Handler to initialize uquake objects which
    inherit from ObsPy class. If obspy_obj is none,
    Kwargs is expected to be a mix of obspy kwargs
    and uquake kwargs specified by the hardcoded
    extra_keys.
    """

    if obspy_obj and len(kwargs) > 0:
        raise AttributeError("Initialize from either \
                              obspy_obj or kwargs, not both")

    # default initialize the extra_keys args to None
    self['extra'] = {}
    [self.__setattr__(key, None) for key in self.extra_keys]

    if obspy_obj:
        _init_from_obspy_object(self, obspy_obj)

        if 'resource_id' in obspy_obj.__dict__.keys():
            rid = obspy_obj.resource_id.id
            self.resource_id = ResourceIdentifier(id=rid,
                                                  referred_object=self)
    else:
        extra_kwargs = pop_keys_matching(kwargs, self.extra_keys)
        super(type(self), self).__init__(**kwargs)  # init obspy_origin args
        [self.__setattr__(k, v) for k, v in extra_kwargs.items()]  # init
        # extra_args


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


def _set_attr_handler(self, name, value, namespace='UQUAKE'):
    """
    Generic handler to set attributes for uquake objects
    which inherit from ObsPy objects. If 'name' is not in
    default keys then it will be set in self['extra'] dict. If
    'name' is not in default keys but in the self.extra_keys
    then it will also be set as a class attribute. When loading
    extra keys from quakeml file, those in self.extra_keys will
    be set as attributes.
    """

    #  use obspy default setattr for default keys
    if name in self.defaults.keys():
        self.__dict__[name] = value
        if isinstance(self.__dict__[name], ResourceIdentifier):
            self.__dict__[name] = ResourceIdentifier(id=value.id)
        # super(type(self), self).__setattr__(name, value)
    elif name in self.extra_keys:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self[name] = value
        if type(value) is np.ndarray:
            value = "npy64_" + array_to_b64(value)
        elif type(value) is str:
            if "npy64_" in value:
                value.replace("npy64_", "")
                b64_to_array(value)
        self['extra'][name] = {'value': value, 'namespace': namespace}
    # recursive parse of 'extra' args when constructing uquake from obspy
    elif name == 'extra':
        if 'extra' not in self:  # hack for deepcopy to work
            self['extra'] = {}
        for key, adict in value.items():
            if key in self.extra_keys:
                self.__dict__[key] = parse_string_val(adict.value)
            else:
                self['extra'][key] = adict
    else:
        raise KeyError(name)


def _set_attr_handler2(self, name, value, namespace='UQUAKE'):
    """
    Generic handler to set attributes for uquake objects
    which inherit from ObsPy objects. If 'name' is not in
    default keys then it will be set in self['extra'] dict. If
    'name' is not in default keys but in the self.extra_keys
    then it will also be set as a class attribute. When loading
    extra keys from quakeml file, those in self.extra_keys will
    be set as attributes.
    """

    #  use obspy default setattr for default keys
    if name in self.defaults.keys():
        super(type(self), self).__setattr__(name, value)
    # recursive parse of extra args when constructing uquake from obspy
    elif name == 'extra':
        if 'extra' not in self:  # hack for deepcopy to work
            self['extra'] = {}
        for key, adict in value.items():
            self.__setattr__(key, parse_string_val(adict.value))
    else:  # branch for extra keys
        if name in self.extra_keys:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self[name] = value
        if type(value) is np.ndarray:
            value = "npy64_" + array_to_b64(value)
        self['extra'][name] = {'value': value, 'namespace': namespace}


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def pop_keys_matching(dict_in, keys):
    # Move keys from dict_in to dict_out
    dict_out = {}
    for key in keys:
        if key in dict_in:
            dict_out[key] = dict_in.pop(key)
    return dict_out


def array_to_b64(array):
    output = io.BytesIO()
    np.save(output, array)
    content = output.getvalue()
    encoded = base64.b64encode(content).decode('utf-8')
    return encoded


def b64_to_array(b64str):
    arr = np.load(io.BytesIO(base64.b64decode(b64str)))
    return arr


def parse_string_val(val, arr_flag='npy64_'):
    """
    Parse extra args in quakeML which are all stored as string.
    """
    if val is None:  # hack for deepcopy ignoring isfloat try-except
        val = None
    elif type(val) == AttribDict:
        val = val
    elif isfloat(val):
        val = float(val)
    elif str(val) == 'None':
        val = None
    elif val[:len(arr_flag)] == 'npy64_':
        val = b64_to_array(val[len(arr_flag):])
    return val


class RayCollection:
    def __init__(self, rays=[]):
        self.__encoded_rays__ = self.__encode_rays__(self, rays)
        self.origin_id = None

    def __setattr__(self, key, value):
        if key == 'rays':
            self.__encoded_rays__ = self.__encode_rays__(self, value)
        else:
            self.__dict__[key] = value

    def __getattr__(self, item):
        if item == 'rays':
            return self.__decode_rays__(self)
        else:
            return self.__dict__[item]

    @staticmethod
    def __encode_rays__(self, rays):
        return b64encode(pickle.dumps(rays))

    @staticmethod
    def __decode_rays__(self):
        return pickle.loads(b64decode(self.__encoded_rays__))

    def append(self, item):
        self.rays = self.rays + [item]


class Ray(object):

    def __init__(self, nodes: list = [], site_code: str = None,
                 arrival_id: ResourceIdentifier = None,
                 phase: str = None, azimuth: float = None,
                 takeoff_angle: float = None,
                 travel_time: float = None,
                 earth_model_id: ResourceIdentifier = None,
                 network: str = None):
        """
        :param nodes: ray nodes
        :param site_code: site code
        :param arrival_id: the ResourceIdentifier of the arrival associated to
        the ray
        :param phase: seismic phase ("P" or "S")
        :param azimuth: Azimuth in degrees
        :param takeoff_angle: takeoff angle in degrees
        :param travel_time: travel time between the source and the site in
        second
        :param earth_model_id: velocity model ResourceIdentifier
        :param network:
        :type network: str
        """

        self.nodes = np.array(nodes)
        self.site_code = site_code
        self.arrival_id = arrival_id
        self.phase = phase
        self.azimuth = azimuth
        self.takeoff_angle = takeoff_angle
        self.travel_time = travel_time
        self.resource_id = obsevent.ResourceIdentifier()
        self.earth_model_id = earth_model_id
        self.network = network

    def __setattr__(self, key, value):
        if key == 'phase':
            if value is None:
                self.__dict__[key] = value
            else:
                self.__dict__[key] = value.upper()
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
    def baz(self):
        # back_azimuth
        baz = None
        if len(self.nodes) > 2:
            v = self.nodes[-2] - self.nodes[-1]
            baz = np.arctan2(v[0], v[1]) * 180 / np.pi
        return baz

    @property
    def station(self):
        return self.site_code.split('.')[0]

    @property
    def location(self):
        return self.site_code.split('.')[1]

    @property
    def back_azimuth(self):
        self.baz

    @property
    def incidence_angle(self):
        ia = None
        if len(self.nodes) > 2:
            v = self.nodes[-2] - self.nodes[-1]
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

        site_code = f'{self.network}.{self.site_code[0:4]}.' \
                    f'{self.site_code[4:]}'

        txt = \
            f"""
       site code: {site_code}
        arrival id: {self.arrival_id}
             phase: {self.phase}
        length (m): {self.length}
   number of nodes: {len(self.nodes)}
            """
        return txt

    def __repr__(self):
        return self.__str__()


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
