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

import numpy as np
from obspy.core.trace import Trace as ObspyTrace
from obspy.core.trace import Stats as ObspyStats
from obspy import UTCDateTime
from obspy.core.trace import AttribDict


class Trace(ObspyTrace):
    def __init__(self, trace=None, **kwargs):
        super(Trace, self).__init__(**kwargs)

        if trace:
            self.stats = Stats(**trace.stats.__dict__)
            self.data = trace.data

    @property
    def site(self):
        return self.stats.station + self.stats.location

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

    # def plot(self, **kwargs):
    #     from obspy.imaging.waveform import WaveformPlotting
    #     waveform = WaveformPlotting(stream=self, **kwargs)
    #
    #     return waveform.plotWaveform()

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

    def plot(self, **kwargs):
        from ..imaging.waveform import WaveformPlotting
        waveform = WaveformPlotting(stream=self, **kwargs)

        return waveform.plotWaveform()


class Stats(ObspyStats):
    def __init__(self, **kwargs):
        super().__init__()
        for key in kwargs.keys():
            self.__dict__[key] = kwargs[key]

    @property
    def site(self):
        return f'{self.station}{self.location}'



