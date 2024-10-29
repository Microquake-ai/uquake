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
from ...core.inventory import Inventory
from ...core.util.tools import datetime_to_epoch_sec

from typing import List
from .protobuf import uquake_pb2, convert_utc_to_grpc_timestamp
from ...core.util.signal import WhiteningMethod, GaussianWhiteningParams


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

def write_one_bit(stream: Stream, filename: str, whiten: bool = True,
                 whitening_method: WhiteningMethod = WhiteningMethod.Gaussian,
                 whitening_params: GaussianWhiteningParams = GaussianWhiteningParams(),
                 correct_response: bool = False, inventory: Inventory = None,
                 pre_filt: List = None,
                 output: str = 'VEL', water_level: float = None, **kwargs):
    if isinstance(stream, Trace):
        stream = Stream(traces=[stream])

    if correct_response and (inventory is None):
        logger.warning('an inventory object must be provided to correct the response')
        correct_response = False

    if correct_response:
        stream.attach_response(inventory)
        stream.remove_response(
            inventory, pre_filt=pre_filt, output=output, water_level=water_level
        )

    traces = []
    networks = []
    stations = []
    locations = []
    channels = []
    sampling_rates = []
    starttimes = []
    endtimes = []
    nsamples = []
    data = []

    for tr in stream:
        if hasattr(tr, 'is_one_bit'):
            if tr.is_one_bit:
                if whiten:
                    logger.warning('the data are already one bit encoded. '
                                   'whitening will be skipped...')
        elif whiten:
            tr.whiten(method = whitening_method,
                      params = whitening_params)
        data.append(np.packbits(tr.data >= 0).tobytes())
        networks.append(tr.stats.network)
        stations.append(tr.stats.station)
        locations.append(tr.stats.location)
        channels.append(tr.stats.channel)
        sampling_rates.append(int(tr.stats.sampling_rate))
        starttimes.append(convert_utc_to_grpc_timestamp(tr.stats.starttime))
        endtimes.append(convert_utc_to_grpc_timestamp(tr.stats.endtime))
        nsamples.append(int(tr.stats.npts))

    OneBit = uquake_pb2.OneBit(
        networks=networks, stations=stations, locations=locations, channels=channels,
        sampling_rates=sampling_rates, starttimes=starttimes,
        endtimes=endtimes, nsamples=nsamples, data=data
    )

    with open(filename, 'wb') as file_out:
        file_out.write(OneBit.SerializeToString())


def read_one_bit(filename, **kwargs) -> Stream:
    # Initialize an empty Stream object
    stream = Stream()

    # Read and parse the OneBit data from file
    one_bit = uquake_pb2.OneBit()
    with open(filename, 'rb') as file_in:
        one_bit.ParseFromString(file_in.read())

    # Iterate through each trace in the OneBit message
    for i in range(len(one_bit.networks)):
        # Extract metadata for each trace
        network = one_bit.networks[i]
        station = one_bit.stations[i]
        location = one_bit.locations[i]
        channel = one_bit.channels[i]
        sampling_rate = one_bit.sampling_rates[i]
        starttime = one_bit.starttimes[i]
        endtime = one_bit.endtimes[i]
        nsamples = one_bit.nsamples[i]

        # Convert protobuf Timestamp to UTCDateTime for start and end times
        start_utc = UTCDateTime(starttime.seconds + starttime.nanos * 1e-9)
        end_utc = UTCDateTime(endtime.seconds + endtime.nanos * 1e-9)

        # Unpack the bit-packed data
        packed_data = np.frombuffer(one_bit.data[i], dtype=np.uint8)
        unpacked_data = np.unpackbits(packed_data)

        # Normalize the unpacked data back to original form (convert 0 to -1 if necessary)
        trace_data = np.where(unpacked_data == 0, -1, 1)  # Example: 0 -> -1, 1 -> 1

        # Create Trace object with the data and metadata
        trace = Trace(data=trace_data)
        trace.stats.network = network
        trace.stats.station = station
        trace.stats.location = location
        trace.stats.channel = channel
        trace.stats.sampling_rate = sampling_rate
        trace.stats.starttime = start_utc
        trace.stats.nsamples = nsamples
        # trace.stats.endtime = end_utc
        trace.is_one_bit = True

        # Append trace to the stream
        stream.append(trace)

    return stream


# from scipy.ndimage import gaussian_filter
# def whiten_spectrum(trace: Trace, smoothing_kernel: int = 5):
#     tr_out = trace.copy()
#     data = tr_out.data
#     data_fft = np.fft.fft(data)
#     smooth_spectrum = gaussian_filter(np.abs(data_fft), sigma=smoothing_kernel)
#     data_fft /= smooth_spectrum
#     tr_out.data = np.real(np.fft.ifft(data_fft))
#     return tr_out
