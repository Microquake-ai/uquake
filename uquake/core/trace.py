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
import obspy.core.trace as obstrace
from obspy import UTCDateTime
from obspy.core.trace import AttribDict


class Trace(obstrace.Trace):
    def __init__(self, trace=None, **kwargs):
        super(Trace, self).__init__(**kwargs)

        if trace:
            self.stats = trace.stats
            self.data = trace.data

    @property
    def sr(self):
        return self.stats.sampling_rate

    def ppv(self):
        return np.max(np.abs(self.data))

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
