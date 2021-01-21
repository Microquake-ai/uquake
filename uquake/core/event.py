# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: event.py
#  Purpose: Expansion of the obspy.core.event module
#   Author: microquake development team
#    Email: devs@microquake.org
#
# Copyright (C) 2016 microquake development team
# --------------------------------------------------------------------
"""
Expansion of the obspy.core.event module

:copyright:
    microquake development team (devs@microquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""


from .handlers import _set_attr_handler, _init_handler

import numpy as np
import obspy.core.event as obsevent
from obspy.core.event import WaveformStreamID, ResourceIdentifier
from copy import deepcopy
from base64 import b64encode, b64decode
import pickle
from ..waveform.mag_utils import calc_static_stress_drop

debug = False


def datetime_to_epoch_sec(dtime):
    return (dtime - datetime(1970, 1, 1)) / timedelta(seconds=1)


class Catalog(obsevent.Catalog):

    extra_keys = []

    __doc__ = obsevent.Catalog.__doc__.replace('obspy', 'microquake')

    def __init__(self, obspy_obj=None, **kwargs):
        if obspy_obj and len(kwargs) > 0:
            raise AttributeError("Initialize from either \
                                  obspy_obj or kwargs, not both")

        if obspy_obj:

            for key in obspy_obj.__dict__.keys():
                if key == 'events':
                    events = []
                    for event in obspy_obj:
                        events.append(Event(obspy_obj=event))
                    self.events = events
                else:
                    self.__dict__[key] = obspy_obj.__dict__[key]

        else:
            super(type(self), self).__init__(
                **kwargs)

    def __setattr__(self, name, value):
        super(type(self), self).__setattr__(name, value)

    def write(self, fileobj, format='quakeml', **kwargs):
        for event in self.events:
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


class Event(obsevent.Event):

    # _format keyword is actualy a missing obspy default
    extra_keys = ['_format', 'ACCEPTED', 'ASSOC_SEISMOGRAM_NAMES', 'AUTO_PROCESSED',
                  'BLAST', 'CORNER_FREQUENCY', 'DYNAMIC_STRESS_DROP',
                  'ENERGY', 'ENERGY_P', 'ENERGY_S', 'EVENT_MODIFICATION_TIME',
                  'EVENT_NAME', 'EVENT_TIME_FORMATTED', 'EVENT_TIME_NANOS',
                  'LOCAL_MAGNITUDE', 'LOCATION_RESIDUAL', 'LOCATION_X',
                  'LOCATION_Y', 'LOCATION_Z', 'MANUALLY_PROCESSED',
                  'NUM_ACCEPTED_TRIGGERS', 'NUM_TRIGGERS', 'POTENCY',
                  'POTENCY_P', 'POTENCY_S', 'STATIC_STRESS_DROP', 'TAP_TEST',
                  'TEST', 'TRIGGERED_SITES', 'USER_NAME', 'network']

    __doc__ = obsevent.Event.__doc__.replace('obspy', 'microquake')

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
            og = self.preferred_origin() or self.origins[0]
            out += '%s | %s, %s, %s | %s' % (og.time, og.x, og.y, og.z,
                                             og.evaluation_mode)

        if self.magnitudes:
            magnitude = self.preferred_magnitude() or self.magnitudes[0]
            out += ' | %s %-2s' % (magnitude.mag,
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


class Origin(obsevent.Origin):
    __doc__ = obsevent.Origin.__doc__.replace('obspy', 'microquake')
    extra_keys = ['x', 'y', 'z', 'x_error', 'y_error', 'z_error', 'scatter',
                  'interloc_vmax', 'interloc_time', '__encoded_rays__']

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)

    def __setattr__(self, name, value):

        if name == 'rays':
            self.__encoded_rays__ = self.__encode_rays__(self, value)
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
    def uncertainty(self):
        if self.origin_uncertainty is None:
            return None
        else:
            return self.origin_uncertainty.confidence_ellipsoid\
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
        string = """
       resource_id: %s
              time: %s
                 x: %s
                 y: %s
                 z: %s
       uncertainty: %s
   evaluation_mode: %s
 evaluation_status: %s
                ---------
          arrivals: %d Elements
        """ \
            % (self.resource_id, self.time.strftime("%Y/%m/%d %H:%M:%S.%f"),
               self.x, self.y, self.z, self.uncertainty, self.evaluation_mode,
               self.evaluation_status,
               len(self.arrivals))
        return string

    def get_incidence_baz_angles(self, station_code, phase):
        baz = None
        inc = None
        for ray in self.rays:
            if (ray.station_code == station_code) and (ray.phase == phase):
                baz = ray.back_azimuth
                inc = ray.incidence_angle
                break
        return baz, inc

    def get_ray_station_phase(self, station_code, phase):
        out_ray = None
        for ray in self.rays:
            if (ray.station_code == station_code) and (ray.phase == phase):
                out_ray = ray
                break
        return out_ray

    def distance_station(self, station_code, phase='P'):
        ray = self.get_ray_station_phase(self, station_code, phase)
        if ray is None:
            return None

        return ray.length


class Magnitude(obsevent.Magnitude):
    __doc__ = obsevent.Magnitude.__doc__.replace('obspy', 'microquake')

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
            seismic_moment = 10 ** (3 * (self.mag + 6.02) / 2)
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

        string = """
             resource_id: {}     
               Magnitude: {}
          Magnitude type: {}
   Corner frequency (Hz): {}
 Radiated Energy (joule): {}
                   Es/Ep: {}
          Seismic moment: {}
       Source volume(m3): {}
Static stress drop (MPa): {}
     Apparent stress(Pa): {}
         evaluation_mode: {}
        """.format(self.resource_id.id, self.mag,
                   self.magnitude_type, self.corner_frequency_hz,
                   self.energy_joule, es_ep, self.seismic_moment, self.potency_m3,
                   self.static_stress_drop_mpa, self.apparent_stress,
                   self.evaluation_mode)

        return string


class Pick(obsevent.Pick):
    __doc__ = obsevent.Pick.__doc__.replace('obspy', 'microquake')
    extra_keys = ['method', 'snr', 'trace_id']

    def __init__(self, obspy_obj=None, **kwargs):
        _init_handler(self, obspy_obj, **kwargs)
        # MTH  - this seems to have been left out ??
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
               self.phase_hint, self.method, self.snr, self.evaluation_mode,
               self.evaluation_status, self.resource_id)
        return string

    def get_sta(self):
        if self.waveform_id is not None:
            return self.waveform_id.station_code


class Arrival(obsevent.Arrival):
    __doc__ = obsevent.Arrival.__doc__.replace('obspy', 'microquake')

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

    @property
    def polarity(self):
        if self.get_pick().polarity == 'positive':
            return 1.0
        elif self.get_pick().polarity == 'negative':
            return -1.0
        else:
            return None

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
                      or microquake.core.event.origin.Arrival
      :param pick: P or S pick
      :type pick: either obspy.core.event.origin.Pick
                      or microquake.core.event.origin.Pick
      :return arrival
      :rtype: obspy.core.event.origin.Arrival or
              microquake.core.event.origin.Arrival
    """

    arrival = None
    for arr in arrivals:
        if arr.pick_id == pick.resource_id:
            arrival = arr
            break

    return arrival


def read_events(*args, **kwargs):

    # converting the obspy object into microquake objects
    cat = obsevent.read_events(*args, **kwargs)
    mq_catalog = Catalog(obspy_obj=cat)

    if mq_catalog[0].preferred_origin():
        if mq_catalog[0].preferred_origin().__encoded_rays__:
            mq_catalog[0].preferred_origin().__encoded_rays__ = eval(
                mq_catalog[0].preferred_origin().__encoded_rays__)

    return mq_catalog


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


class Ray:

    def __init__(self, nodes=[]):
        self.nodes = np.array(nodes)
        self.station_code = None
        self.arrival_id = None
        self.phase = None
        self.azimuth = None
        self.takeoff_angle = None
        self.travel_time = None
        self.resource_id = obsevent.ResourceIdentifier()

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
            baz = np.arctan2(v[0], v[1])
        return baz

    @property
    def back_azimuth(self):
        self.baz

    @property
    def incidence_angle(self):
        ia = None
        if len(self.nodes) > 2:
            v = self.nodes[-2] - self.nodes[-1]
            h = np.sqrt(v[0] ** 2 + v[1] ** 2)
            ia = np.arctan2(h, v[2])
        return ia

    def __len__(self):
        return self.length

    def __str__(self):
        txt = \
            f"""
      station code: {self.station_code}
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
