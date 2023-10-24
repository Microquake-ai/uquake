# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: core.py
# Purpose: plugin for reading and writing various waveform format expending
# the number of format readable.
#   Author: uquake development team
#    Email: devs@uquake.org
#
# Copyright (C) 2016 uquake development team
# --------------------------------------------------------------------
"""
plugin for reading and writing various waveform format expending
# the number of format readable.

:copyright:
    uquake development team (devs@uquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from datetime import datetime, timedelta
from glob import glob
from struct import unpack

# from io import BytesIO
import numpy as np
from dateutil.parser import parse
from obspy import UTCDateTime
from obspy.core.trace import Stats
from obspy.core.util.decorator import uncompress_file as uncompress
from pytz import timezone

from loguru import logger
from ...core import Stream, Trace, read
from ...core.util.tools import datetime_to_epoch_sec


def decompose_mseed(mseed_bytes, mseed_reclen=4096):
    """
    Return dict with key as epoch starttime and val
    as concatenated mseed blocks which share that
    starttime.
    """
    starts = np.arange(0, len(mseed_bytes), mseed_reclen)
    dchunks = {}

    for start in starts:
        end = start + mseed_reclen
        chunk = mseed_bytes[start:end]
        dt = mseed_date_from_header(chunk)
        key = int(datetime_to_epoch_sec(dt) * 1000)

        if key not in dchunks:
            dchunks[key] = b''
        dchunks[key] += chunk

    return dchunks


def mseed_date_from_header(block4096):
    """

    :param block4096: a binary string
    :return starttime in UTCDateTime:

    mseed bytes 20 to 30 are date in header
    YYYY The year with the century (e.g., 1987)
    DDD  The julian day of the year (January 1 is 001)
    HH   The hour of the day UTC (00—23)
    MM   The minute of the day (00—59)
    SS   The seconds (00—60; use 60 only to note leap seconds)
    FFFF The fraction of a second (to .0001 seconds resolution)
    """

    vals = unpack('>HHBBBBH', block4096[20:30])
    year, julday, hour, minute, sec, _, sec_frac = vals
    tstamp = '%0.4d,%0.3d,%0.2d:%0.2d:%0.2d.%0.4d' % (year, julday, hour,
                                                      minute, sec, sec_frac)
    dt = datetime.strptime(tstamp, '%Y,%j,%H:%M:%S.%f')

    return UTCDateTime(dt)


def read_IMS_ASCII(path, net='', **kwargs):
    """
    read a IMS_ASCII seismogram from a single station
    :param path: path to file
    :return: uquake.core.Stream
    """

    data = np.loadtxt(path, delimiter=',', skiprows=1)
    stats = Stats()

    with open(path) as fid:
        field = fid.readline().split(',')

    stats.sampling_rate = float(field[1])
    timetmp = datetime.fromtimestamp(float(field[5])) \
              + timedelta(
        seconds=float(field[6]) / 1e6)  # trigger time in second

    trgtime_UTC = UTCDateTime(timetmp)
    stats.starttime = trgtime_UTC - float(field[10]) / stats.sampling_rate
    stats.npts = len(data)

    stats.station = field[8]
    stats.network = net

    traces = []
    component = np.array(['X', 'Y', 'Z'])
    std = np.std(data, axis=0)
    mstd = np.max(std)

    for k, dt in enumerate(data.T):
        stats.channel = '%s' % (component[k])
        traces.append(Trace(data=np.array(dt), header=stats))

    return Stream(traces=traces)


# def write_ims_ascii(stream, filename, **kwargs):
#     """
#     write IMS ASCII format
#     :param stream: data stream
#     :type stream: ~uquake.core.Stream.stream
#     :param filename: filename/path
#     :type filename: str
#     """
#
#
#     for tr in stream:
#
#         decimated_sampling = tr.stats.sampling_rate
#         sampling_rate = tr.stats.sampling_rate
#         decimation_factor = 0
#         trigger_time = tr.stats.starttime.datetime.timestamp
#         trigger_time_second = np.floor(trigger_time)
#         trigger_time_usecond = (trigger_time - trigger_time_second) * 1e6
#
#     header_line = f'x_24,{decimated_sampling},{sampling_rate},' \
#                   f'{decimation_factor},100.0,{trigger_time_second},' \
#                   f'{trigger_time_usecond},0,}'
#
#
#


@uncompress
def read_ESG_SEGY(fname, inventory=None, **kwargs):
    """
    read data produced by ESG and turn them into a valid stream with network,
    station and component information properly filled
    :param fname: the filename
    :param inventory: a location object containing location information
    :type inventory: ~uquake.core.inventory.Inventory
    :return: ~uquake.core.stream.Stream
    """

    if inventory is None:
        return

    st = read(fname, format='SEGY', unpack_trace_headers=True, **kwargs)

    trs = []
    missed_traces = 0
    for tr in st:
        tr_y = tr.stats.segy.trace_header.group_coordinate_x
        tr_x = tr.stats.segy.trace_header.group_coordinate_y

        h_distances = []
        locations = inventory.networks[0].locations

        component = np.abs(tr.stats.segy.trace_header.trace_identification_code
                           - 14)

        if component == 13:
            component = 0

        for location in inventory.networks[0].locations:
            h_distance = np.linalg.norm([location.x - tr_x, location.y - tr_y])
            h_distances.append(h_distance)
        if np.min(h_distances) > 1:
            missed_traces += 1
            continue

        i = np.argmin(h_distances)
        location = locations[i]

        channel_code = f'{location.channels[0].code[0:2]}{component:0.0f}'

        tr_stats = Stats()
        tr_stats.network = network
        tr_stats.station = location.station.code
        tr_stats.location = location.location_code
        tr_stats.channel = channel_code
        tr_stats.sampling_rate = tr.stats.sampling_rate

        msec_starttime = tr.stats.segy.trace_header.lag_time_A
        usec_starttime = tr.stats.segy.trace_header.lag_time_B

        usecs = msec_starttime / 1000. + usec_starttime / 1.0e6
        tr_stats.starttime = tr.stats.starttime + usecs
        tr_stats.npts = len(tr.data)

        trs.append(Trace(data=tr.data, header=tr_stats))

    return Stream(traces=trs)


@uncompress
def read_TEXCEL_CSV(filename, **kwargs):
    """
    Reads a texcel csv file and returns a uquake Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        uquake :func:`~uquake.core.stream.read` function, call this
        instead.
    :param filename: the path to the file
    :param kwargs:
    :return: ~uquake.core.stream.Stream
    """

    with open(filename) as fle:
        x = []
        y = []
        z = []

        for k, line in enumerate(fle):
            if k == 0:
                if 'MICROPHONE' in line:
                    offset = 9
                else:
                    offset = 8
            # header

            if k < 2:
                continue

            val = line.strip().split(',')

            # relative time

            if k == 3:
                rt0 = timedelta(seconds=float(val[0]))

            elif k == 6:
                station = str(eval(val[offset]))

            elif k == 7:
                date = val[offset]

            elif k == 8:
                date_time = date + " " + val[offset]
                datetime = parse(date_time)
                starttime = datetime + rt0

            elif k == 9:
                location = val[offset]

            elif k == 10:
                location = val[offset]

            elif k == 17:

                sensitivity_x = float(val[offset])
                sensitivity_y = float(val[offset + 1])
                sensitivity_z = float(val[offset + 2])

            elif k == 18:
                range_x = float(val[offset])
                range_y = float(val[offset + 1])
                range_z = float(val[offset + 2])

            elif k == 19:
                trigger_x = float(val[offset])
                trigger_y = float(val[offset + 1])
                trigger_z = float(val[offset + 2])

            elif k == 20:
                si_x = float(val[offset])
                si_y = float(val[offset + 1])
                si_z = float(val[offset + 2])

            elif k == 21:
                sr_x = float(val[offset])
                sr_y = float(val[offset + 1])
                sr_z = float(val[offset + 2])

            x.append(float(val[1]))
            y.append(float(val[2]))
            z.append(float(val[3]))

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        stats = Stats()
        stats.network = location
        stats.delta = si_x / 1000.0
        stats.npts = len(x)
        stats.location = location
        stats.station = station
        stats.starttime = UTCDateTime(starttime)

        stats.channel = 'radial'
        tr_x = Trace(data=x / 1000.0, header=stats)

        stats.delta = si_y / 1000.0
        stats.channel = 'transverse'
        tr_y = Trace(data=y / 1000.0, header=stats)

        stats.delta = si_z / 1000.0
        stats.channel = 'vertical'
        tr_z = Trace(data=z / 1000.0, header=stats)

    return Stream(traces=[tr_x, tr_y, tr_z])
