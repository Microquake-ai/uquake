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
# Filename: core.py
#  Purpose: Expansion of the obspy.core.trace module
#   Author: uquake development team
#    Email: devs@uquake.org
#
# Copyright (C) 2016 uquake development team
# --------------------------------------------------------------------
"""
Expansion of the obspy.core.trace module

:copyright:
    uquake development team (devs@uquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from abc import ABC
import numpy as np
from obspy.core.trace import Trace as ObspyTrace
from obspy.core.trace import Stats as ObspyStats
from obspy import UTCDateTime
from obspy.core.event import WaveformStreamID
from obspy.core.trace import AttribDict
from uquake.core.util.decorators import update_doc

from .event import Pick
from .util import tools


@update_doc
class Stats(ObspyStats, ABC):
    __doc__ = ObspyStats.__doc__.replace('obspy', 'uquake')

    def __init__(self, stats=None, **kwargs):
        super().__init__(**kwargs)

        if stats:
            for item in stats.__dict__.keys():
                self.__dict__[item] = stats.__dict__[item]

    @property
    def site(self):
        return f'{self.station}.{self.location}'

    def __str__(self):
        """
        Return better readable string representation of Stats object.
        """
        priorized_keys = ['network', 'station', 'location', 'channel',
                          'starttime', 'endtime', 'sampling_rate', 'delta',
                          'npts', 'calib']
        return self._pretty_str(priorized_keys)


class Trace(ObspyTrace, ABC):
    def __init__(self, trace=None, **kwargs):
        super(Trace, self).__init__(**kwargs)

        if trace:
            self.stats = Stats(stats=trace.stats)
            self.data = trace.data

        elif 'header' in kwargs.keys():
            self.stats = Stats(stats=kwargs['header'])

        elif 'stats' in kwargs.keys():
            self.stats = Stats(stats=kwargs['stats'])

        else:
            self.stats = Stats()

    @property
    def sr(self):
        return self.stats.sampling_rate

    @property
    def ppv(self):
        return np.max(np.abs(self.data))

    @property
    def ppa(self):
        return None

    def time_to_index(self, time):
        return int((time - self.stats.starttime) * self.sr)

    def index_to_time(self, index):
        return self.stats.starttime + (index / self.sr)

    def times(self):
        sr = self.stats.sampling_rate

        return np.linspace(0, len(self.data) / sr, len(self.data))

    def plot(self, **kwargs):
        from uquake.imaging.waveform import WaveformPlotting
        from uquake.core.stream import Stream
        waveform = WaveformPlotting(stream=Stream(traces=[self], **kwargs))

        return waveform.plotWaveform()

    def make_pick(self, pick_time, wlen_search,
                  stepsize, snr_wlens, phase_hint=None):

        ipick = self.time_to_index(pick_time)
        sr = self.stats.sampling_rate
        stepsize_samp = int(stepsize * sr)
        snr_wlens_samp = (snr_wlens * sr).astype(int)
        wlen_search_samp = int(wlen_search * sr)

        new_pick, snr = tools.repick_using_snr(self.data, ipick,
                                               wlen_search_samp, stepsize_samp,
                                               snr_wlens_samp)

        waveform_id = WaveformStreamID(channel_code=self.stats.channel,
                                       station_code=self.stats.station)

        pick = Pick(time=self.index_to_time(newpick), waveform_id=waveform_id,
                    phase_hint=phase_hint, evaluation_mode='automatic',
                    evaluation_status='preliminary', method='snr', snr=snr)

        return pick

    def time_within(self, utime, edge_buf=0.0):
        within = True

        if (utime - edge_buf) < self.stats.starttime:
            within = False
        elif (utime + edge_buf) > self.stats.endtime:
            within = False

        return within

    @classmethod
    def create_from_json(cls, trace_json_object):

        trace_json_object['stats']['starttime'] = UTCDateTime(
            trace_json_object['stats']['starttime'])
        trace_json_object['stats']['endtime'] = UTCDateTime(
            trace_json_object['stats']['endtime'])

        trc = cls(header=trace_json_object['stats'], data=np.array(
            trace_json_object['data'], dtype='float32'))

        return trc

    def to_json(self):
        trace_dict = dict()
        trace_dict['stats'] = dict()

        for key in self.stats.keys():
            if isinstance(self.stats[key], UTCDateTime):
                trace_dict['stats'][key] = self.stats[key].isoformat()
            elif isinstance(self.stats[key], AttribDict):
                trace_dict['stats'][key] = self.stats[key].__dict__
            else:
                trace_dict['stats'][key] = self.stats[key]

        trace_dict['data'] = self.data.tolist()

        return trace_dict
