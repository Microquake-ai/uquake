# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: core.py
#  Purpose: Expansion of the obspy.core.stream module
#   Author: uquake development team
#    Email: devs@uquake.org
#
# Copyright (C) 2016 uquake development team
# --------------------------------------------------------------------
"""
Expansion of the obspy.core.stream module

:copyright:
    uquake development team (devs@uquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from abc import ABC
from io import BytesIO

import numpy as np
import obspy.core.stream as obsstream
from pkg_resources import load_entry_point

from .trace import Trace
from .util import ENTRY_POINTS, tools
from .logging import logger
from pathlib import Path


class Stream(obsstream.Stream, ABC):
    __doc__ = obsstream.Stream.__doc__.replace('obspy', 'uquake')

    def __init__(self, stream=None, **kwargs):
        super(Stream, self).__init__(**kwargs)

        if stream:
            traces = []

            for tr in stream.traces:
                traces.append(Trace(trace=tr))

            self.traces = traces

    def composite(self):
        """
        returns a new stream object containing composite trace for all station.
        The amplitude of the composite traces are the norm of the amplitude of
        the trace of all component and the phase of the trace (sign) is the
        sign of the first components of a given station.
        :param st: a stream object
        :type st: ~uquake.core.stream.Stream
        :rtype: ~uquake.core.stream.Stream

        """

        return composite_traces(self)

    def select(self, **kwargs):
        if 'site' in kwargs.keys():
            trs = [tr for tr in self.traces if tr.stats.site == kwargs['site']]
        else:
            return super().select(**kwargs)

        st_tmp = Stream(traces=trs)

        kwargs_tmp = {}
        for key in kwargs.keys():
            if key == 'site':
                continue
            kwargs_tmp[key] = kwargs[key]

        return st_tmp.select(**kwargs_tmp)

    def as_array(self, wlen_sec=None, taplen=0.05):
        t0 = np.min([tr.stats.starttime for tr in self])
        sr = self[0].stats.sampling_rate

        if wlen_sec is not None:
            npts_fix = int(wlen_sec * sr)
        else:
            npts_fix = int(np.max([len(tr.data) for tr in self]))

        return tools.stream_to_array(self, t0, npts_fix, taplen=taplen), sr, t0

    def chan_groups(self):
        chanmap = self.channel_map()
        groups = [np.where(sk == chanmap)[0] for sk in np.unique(chanmap)]

        return groups

    def channel_map(self):
        stations = np.array([tr.stats.station for tr in self])
        unique = np.unique(stations)
        unique_dict = dict(zip(unique, np.arange(len(unique))))
        chanmap = np.array([unique_dict[chan] for chan in stations], dtype=int)

        return chanmap

    def write(self, filename, format='MSEED', **kwargs):

        from six import string_types
        f = filename

        if isinstance(filename, string_types):
            if filename.endswith('gz'):
                import gzip
                f = gzip.open(filename, 'w')
            elif filename.endswith('bz2'):
                import bz2
                f = bz2.BZ2File(filename, 'w')
            elif filename.endswith('zip'):
                print('Zip protocol is not supported')

        st_out = self.copy()

        return obsstream.Stream.write(st_out, f, format, **kwargs)

    write.__doc__ = obsstream.Stream.write.__doc__.replace('obspy',
                                                           'uquake')

    def write_bytes(self):
        buf = BytesIO()
        self.write(buf, format='MSEED')

        return buf.getvalue()

    def valid(self, **kwargs):
        return is_valid(self, return_stream=True)

    def concat(self, comp_st):

        c = (comp_st is not None)

        if c:
            for i, (t1, t2) in enumerate(zip(comp_st.traces, self.traces)):
                self.detrend_norm(t2)
                comp_st.traces[i] = t1.__add__(t2, method=1, fill_value=0)
        else:
            for t in self:
                self.detrend_norm(t)

            comp_st = self

        return comp_st

    @property
    def unique_stations(self):
        return np.sort(np.unique([tr.stats.station for tr in self]))

    @property
    def unique_sites(self):
        return np.sort(np.unique([tr.stats.site for tr in self]))

    @property
    def stations(self):
        return self.unique_stations

    @property
    def sites(self):
        return self.unique_sites

    def zpad_names(self):
        for tr in self.traces:
            tr.stats.station = tr.stats.station.zfill(3)
        self.sort()

    def zstrip_names(self):
        for tr in self.traces:
            tr.stats.station = tr.stats.station.lstrip('0')

    def distance_time_plot(self, event, site, scale=20, freq_min=100,
                           freq_max=1000):
        """
        plot traces that have
        :param event: event object
        :param site: site object
        :param scale: vertical size of pick markers and waveform
        :return: plot handler
        """

        st = self.copy()
        st.detrend('demean')
        st.taper(max_percentage=0.01)
        st.filter('bandpass', freqmin=freq_min, freqmax=freq_max)

        import matplotlib.pyplot as plt
        import numpy as np

        # initializing the plot

        ax = plt.subplot(111)

        if event.preferred_origin():
            origin = event.preferred_origin()
        elif event.origins:
            origin = event.origins[0]
        else:
            return

        event_location = origin.loc

        # find the earliest start time and latest end time
        start_time = None
        end_time = None

        for tr in st:
            if not start_time:
                start_time = tr.stats.starttime
                end_time = tr.stats.endtime

            if tr.stats.starttime < start_time:
                start_time = tr.stats.starttime

            if tr.stats.endtime > end_time:
                end_time = tr.stats.endtime

        for tr in st:
            station_code = tr.stats.station
            # search for arrival
            station = site.select(station_code).stations()[0]
            station_location = station.loc
            distance = np.linalg.norm(event_location - station_location)
            p_pick = None
            s_pick = None
            data = (tr.data / np.max(np.abs(tr.data))) * scale
            time_delta = tr.stats.starttime - start_time
            time = np.arange(0, len(data)) / tr.stats.sampling_rate + \
                   time_delta

            for arrival in origin.arrivals:
                if arrival.get_pick().waveform_id.station_code == station_code:
                    distance = arrival.distance

                    if arrival.phase == 'P':
                        p_pick = arrival.get_pick().time - start_time
                    elif arrival.phase == 'S':
                        s_pick = arrival.get_pick().time - start_time

            ax.plot(time, data + distance, 'k')

            if p_pick:
                ax.vlines(p_pick, distance - scale, distance + scale, 'r')

            if s_pick:
                ax.vlines(s_pick, distance - scale, distance + scale, 'b')

            plt.xlabel('relative time (s)')
            plt.ylabel('distance from event (m)')

    @staticmethod
    def create_from_json_traces(traces_json_list):
        traces = []
        # for tr_json in traces_json_list:

        for i, tr_json in enumerate(traces_json_list):
            stats = tr_json['stats']
            tr = Trace.create_from_json(tr_json)
            traces.append(tr)

        return Stream(traces=traces)

    def to_traces_json(self):
        traces = []

        for tr in self:
            trout = tr.to_json()
            traces.append(trout)

        return traces

    def plot(self, *args, **kwargs):
        """
        see Obspy stream.plot()
        """
        from ..imaging.waveform import WaveformPlotting
        waveform = WaveformPlotting(stream=self, *args, **kwargs)

        return waveform.plotWaveform(*args, **kwargs)


# from uquake.core import read, read_events
# from spp.utils import application
# app = application.Application()
# site = app.get_stations()
# st = read('2018-11-08T10:21:49.898496Z.mseed', format='mseed')
# cat = read_events('test.xml')
# evt = cat[0]
# st = st.composite()


def is_valid(st_in, return_stream=False, STA=0.005, LTA=0.1, min_num_valid=5):
    """
        Determine if an event is valid or return valid traces in a  stream
        :param st_in: stream
        :type st_in: uquake.core.stream.Stream
        :param return_stream: return stream of valid traces if true else return
        true if the event is valid
        :type return_stream: bool
        :param STA: short term average used to determine if an event is valid
        :type STA: float
        :param LTA: long term average
        :type LTA: float
        :param min_num_valid: minimum number of valid traces to declare the
        event valid
        :type min_num_valid: int
        :rtype: bool or uquake.core.stream.Stream
    """

    from scipy.ndimage.filters import gaussian_filter1d
    from obspy.signal.trigger import recursive_sta_lta

    st = st_in.copy()
    st.detrend('demean').detrend('linear')
    trstd = []
    trmax = []
    trs_out = []
    st_comp = composite_traces(st)

    for tr in st_comp:
        if not np.any(tr.data):
            continue
        sampling_rate = tr.stats.sampling_rate
        trstd.append(np.std(tr.data))
        trmax.append(np.max(np.abs(tr.data)))
        nsta = int(STA * sampling_rate)
        nlta = int(LTA * sampling_rate)
        cft = recursive_sta_lta(np.array(tr.data), nsta, nlta)
        sfreq = tr.stats['sampling_rate']
        sigma = sfreq / (2 * np.pi * 100)
        cft = gaussian_filter1d(cft, sigma=sigma, mode='reflect')
        try:
            mx = np.r_[True, cft[1:] > cft[:-1]] & \
                 np.r_[cft[:-1] > cft[1:], True]
        except Exception as e:
            logger.error(e)
            continue

        i1 = np.nonzero(mx)[0]
        i2 = i1[cft[i1] > np.max(cft) / 2]

        tspan = (np.max(i2) - np.min(i2)) / sampling_rate

        ratio = np.max(np.abs(tr.data)) / np.std(tr.data)

        accept = True

        if len(i2) < 3:
            if ratio < 4:
                accept = False

        elif len(i2) >= 4:
            accept = False
        # else:
        #     if ratio < 4:
        #         accept = False

        if tspan > 0.1:
            accept = False

        if (len(i2) == 2) and (tspan > 0.01) and (tspan < 0.1):
            if ratio > 5:
                accept = True

        if accept:
            for tr_accepted in st_in.select(station=tr.stats.station):
                trs_out.append(tr_accepted)

    st_out = Stream(traces=trs_out)

    if return_stream:
        return st_out
    else:
        if len(st.unique_stations()) >= min_num_valid:
            return True
        else:
            return False


def check_for_dead_trace(tr):
    eps = 1e-6
    data = tr.data.copy()
    mean = np.mean(data)
    max = np.max(data) - mean
    min = np.min(data) - mean
    # print('%s: mean:%f max:%f min:%f' % (tr.get_id(), mean, max, min))

    if max < eps and np.abs(min) < eps:
        return 1
    else:
        return 0


def composite_traces(st_in):
    """
    Requires length and sampling_rates equal for all traces
    returns a new stream object containing composite trace for all station.
    The amplitude of the composite traces are the norm of the amplitude of
    the trace of all component and the phase of the trace (sign) is the sign
    of the first components of a given station.
    :param st_in: a stream object
    :type st_in: ~uquake.core.stream.Stream
    :rtype: ~uquake.core.stream.Stream

    """

    trsout = []

    st = st_in.copy()
    st.detrend('demean')

    for site in st.unique_sites:
        trs = st.select(site=site)

        if len(trs) == 1:
            trsout.append(trs[0].copy())

            continue

        npts = len(trs[0].data)
        buf = np.zeros(npts, dtype=trs[0].data.dtype)

        for tr in trs:
            dat = tr.data
            buf += (dat - np.mean(dat)) ** 2

        buf = np.sign(trs[0].data) * np.sqrt(buf)
        stats = trs[0].stats.copy()
        ch = st_in.traces[0].stats.channel
        if len(ch) > 1:
            prefix = ch[:-1]
        stats.channel = f'{prefix}C'
        trsout.append(Trace(data=buf.copy(), header=stats))

    return Stream(traces=trsout)


def read(filename, format='MSEED', **kwargs):
    if isinstance(filename, Path):
        filename = str(filename)

    if format in ENTRY_POINTS['waveform'].keys():
        format_ep = ENTRY_POINTS['waveform'][format]
        read_format = load_entry_point(format_ep.dist.key,
                                       'obspy.plugin.waveform.%s' %
                                       format_ep.name, 'readFormat')

        st = Stream(stream=read_format(filename, **kwargs))

        # making sure the channel names are upper case
        trs = []
        for tr in st:
            tr.stats.channel = tr.stats.channel.upper()
            trs.append(tr.copy())

        st.traces = trs

        return st
    else:
        return Stream(stream=obsstream.read(filename, format=format, **kwargs))


read.__doc__ = obsstream.read.__doc__.replace('obspy', 'uquake')
