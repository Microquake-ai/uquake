# -*- coding: utf-8; -*-
#
# (c) 2016 uquake development team
#
# This file is part of the uquake library
#
# uquake is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# uquake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with uquake.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from ..core import Stream, Trace
from obspy.realtime.signal import kurtosis
from scipy.signal import detrend
from ..core.logging import logger
from ..core.event import make_pick, Pick
from ..core.inventory import Inventory
from ..core.util.tools import copy_picks_to_dict
from ..core import UTCDateTime


def measure_polarity(st, catalog, site, average_time_window=1e-3,
                     lp_filter_freq=100):
    """
    Measure the P- and S-wave polarities
    :param st: seismograms
    :type st: either obspy.core.Stream or uquake.core.Stream
    :param catalog: catalog object
    :type catalog: uquake.core.Catalog
    :param site: sensor information
    :type site: uquake.core.station.Site
    :param average_time_window: time window over which the polarity is measured (s)
    :type average_time_window: float
    :param lp_filter_freq: frequency of the low pass filter to apply for
    polarity picking
    :rparam: returns a copy of the input catalog
    :rtype: uquake.core.Stream
    """
    cat_out = catalog.copy()

    for iev, ev in enumerate(cat_out.events):
        for ipick, pick in enumerate(ev.picks):
            sta_code = pick.waveform_id.station_code
            ch_code = pick.waveform_id.channel_code
            trs = st.select(station=sta_code, channel=ch_code)
            trs = trs.detrend('linear').detrend('demean')
            sta = site.select(station=sta_code, channel=ch_code)
            if not sta.networks[0].stations:
                continue
            if (len(sta.networks) > 1) or (len(sta.networks[0].stations) > 1):
                logger.warning("MeasurePolarity: multiple station selected "
                               "from site. Only the first will be used")
            sta = sta.networks[0].stations[0]
            motion_type = sta.motion_type
            if motion_type == 'acceleration':
                trs = trs.integrate().detrend('linear').detrend(
                    'demean').integrate()
            elif motion_type == 'velocity':
                trs.integrate()
            elif motion_type == 'displacement':
                pass
            else:
                logger.warning("MeasurePolarity: motion_type not set for "
                               "sensor %s... Displacement will be assumed"
                               % stname)

            trs.filter('lowpass', freq=lp_filter_freq)
            if len(trs) > 1:
                logger.warning("number of trace for station %s and channel %s"
                               "is greater than 1. Only the first trace "
                               "will be used" % (
                                   sta_code, ch_code))
            tr = trs.traces[0]

            sp_s = int(
                (pick.time - tr.stats.starttime) * tr.stats.sampling_rate)
            sp_e = int(
                (sp_s + average_time_window * tr.stats.sampling_rate)) + 1
            pol = np.sign(np.mean(tr.data[sp_s:sp_e]) - tr.data[sp_s])

            if pol > 0:
                cat_out.events[iev].picks[ipick].polarity = "positive"
            elif pol < 0:
                cat_out.events[iev].picks[ipick].polarity = "negative"
            else:
                cat_out.events[iev].picks[ipick].polarity = "undecidable"

    return cat_out


def _measure_pick_polarity_tr(tr, pick_time, signal_type='Acceleration',
                              nsamp_avg=10):
    """Mesure pick polarity
    The pick polarity is measured on the displacement trace looking at the
    difference
    between the amplitude at the pick time and the sign of the average
    amplitude
    for the 10 samples following the pick time

    :param tr: signal trace seismogram
    :type tr: obspy Trace
    :param pick_time: pick time
    :type pick_time: obspy UTCDateTime
    :param signal_type: Type of signal
    :type signal_type: str [accepted values are 'Acceleration', 'Velocity', and 'Displacement']
    :param nsamp_avg: number of sample on which to calculate the average difference
    :type nsamp_avg: int
    :returns: signa of signal polarity (1 or -1)
    :rtype: int
    """

    if signal_type not in ['Acceleration', 'Velocity', 'Displacement']:
        # raise error and exit funciton
        pass

    starttime = tr.stats.starttime
    sampling_rate = tr.stats.sampling_rate
    sample_pick = int(pick_time - starttime * sampling_rate)

    polarity = np.sign(np.mean(tr.data[sample_pick + 1:sample_pick +
                                                       nsamp_avg + 2] -
                               tr.data[sample_pick]))

    return polarity


def pick_uncertainty(tr, pick_time, snr_window=10):
    """Estimate the pick uncertainty, dt, based on the following equation
    dt = 1 / (fm * log_2(1 + SNR ** 2)
    where fm is the middle frequency and SNR, the signal to noise ratio in dB.

    :param tr: signal trace seismogram
    :type tr: obspy Trace
    :param pick_time: pick time
    :type pick_time: obspy UTCDateTime
    :param snr_window: length of window in ms centered on pick_time to
    calculate SNR.
        Noise and signal energy are calculated over the first and second half
        of the pick window, respectively.
    :type snr_window: float
    :returns: pick uncertainty in second
    :rtype: float
    """

    fm = 100  # central frequency in Hz
    # fm = centralfrequency(tr, pick_time, window)
    # tr is obspy tr, pick_time same, window window from pick over wich
    # signal frequencies are calculated
    st = Stream(traces=[tr])
    snr = calculate_snr(st, pick_time, snr_window)
    return 1 / (fm * np.log(1 + snr ** 2) / np.log(2))


def snr_repicker(st, picks, start_search_window, end_search_window,
                 search_resolution, pre_pick_window_len=1e-3,
                 post_pick_window_len=20e-3, trace_padding=30e-3):
    """
    Function to improve the picks based on the SNR.
    :param st: seismogram containing a seismic event
    :type st: :py:class:`obspy.core.stream.Stream`
    :param picks: list of uquake.core.event.Pick object
    picks
    :type picks: uquake.core.event.Catalog
    :param start_search_window: start of the search window relative to
    provided picks in seconds
    :type start_search_window: float
    :param end_search_window: end of the search window relative to the
    provided picks in seconds
    :type end_search_window: float
    :param search_resolution: resolution of the search space in seconds
    :type search_resolution: float
    :param pre_pick_window_len: length of the window before the presumed pick
    :type pre_pick_window_len: float
    :param post_pick_window_len: length of the window after the presumed pick
    :type post_pick_window_len: float
    :param trace_padding: buffer from trace start and end time excluded from
    search window
    :returns:  Tuple comprising 1) a :py:class:`uquake.core.event.Catalog`
    a new catalog containing a single event with a list of picks and 2) the SNR
    """

    snr_dt = np.arange(start_search_window, end_search_window,
                       search_resolution)

    st.detrend('demean')

    o_picks = []
    snrs = []

    pre_window_length = pre_pick_window_len
    post_window_length = post_pick_window_len

    for pick in picks:
        network = pick.waveform_id.network_code
        station = pick.waveform_id.station_code
        location = pick.waveform_id.location_code

        tr = st.select(network=network, station=station,
                       location=location).copy()[0]
        if tr is None:
            continue

        earliest_time = pick.time + start_search_window
        latest_time = pick.time + end_search_window

        if earliest_time is None or earliest_time < tr.stats.starttime + \
                trace_padding:
            earliest_time = tr.stats.starttime + trace_padding

        if latest_time is None or latest_time > tr.stats.endtime - \
                trace_padding:
            latest_time = tr.stats.endtime - trace_padding

        tau = []
        for dt in snr_dt:
            taut = pick.time + dt

            if earliest_time <= taut <= latest_time:
                tau.append(taut)

        if not tau:
            continue
        if tau[0] <= tr.stats.starttime:
            logger.error("Too early! tau[0]=%s <= tr.st=%s"
                         % (tau[0], tr.stats.starttime))
            logger.error("earliest_time:%s latest_time:%s"
                         % (earliest_time, latest_time))

            continue
        if tau[-1] >= tr.stats.endtime:
            logger.error("Too late! tau[-1]=%s >= tr.et=%s"
                         % (tau[-1], tr.stats.endtime))
            continue

        tau = np.array(tau)
        indices = (tau - tr.stats.starttime) * tr.stats.sampling_rate
        tmp = np.array([(taut, index,
                         calculate_snr(tr, taut,
                                       pre_wl=pre_window_length,
                                       post_wl=post_window_length))
                        for taut, index in zip(tau, indices)])

        alpha = 0

        index = np.argmax(tmp[:, 2])
        pick_time = tmp[index, 0]

        snr = calculate_snr(tr, pick_time, pre_wl=pre_window_length,
                            post_wl=post_window_length)

        # logger.debug("%s: sta:%s [%s] time_diff:%0.3f SNR:%.2f" %
        #              (function_name, station, phase, pick.time -
        #               pick_time, snr))

        method_string = 'snr_picker preWl=%.3g postWl=%.3g alpha=%.1f' % \
                        (pre_window_length, post_window_length, alpha)
        o_picks.append(make_pick(pick_time, phase=pick.phase_hint,
                                 wave_data=tr, snr=snr,
                                 method_string=method_string,
                                 resource_id=pick.resource_id))
        snrs.append(snr)

    return snrs, o_picks


def linearity_ensemble_re_picker(st, picks, start_search_window,
                                 end_search_window,
                                 start_refined_search_window,
                                 end_refinde_search_window,
                                 refined_window_search_resolution,
                                 linearity_calc_window_len):
    """
        Function to improve the picks based on the linearity or planarity
        for an ensemble of traces.
        :param st: waveforms containing a seismic event
        :type st: :py:class:`uquake.core.stream.Stream`
        :param picks: list of :class:`uquake.core.event.Pick` object
        picks
        :type picks: list of :py:class:`uquake.core.event.Pick`
        :param start_search_window: start of the search window relative to
        provided picks in seconds
        :type start_search_window: float
        :param end_search_window: end of the search window relative to the
        provided picks in seconds
        :param start_refined_search_window: start of the refined search
        window in seconds
        :type: float
        :param end_refined_search_window: end of the refined search window in
        seconds
        :type: float
        :type end_search_window: float
        :param refined_window_search_resolution: resolution of the search
        space in seconds
        :type refined_window_search_resolution: float
        :param linearity_calc_window_len: length of the window after the
        presumed pick used to calculate the linearity
        :param trace_padding: buffer from trace start and end time excluded
        from search window
        :returns:  Tuple comprising 1) a :py:class:`uquake.core.event.Catalog`
        a new catalog containing a single event with a list of picks and 2)
        the SNR
        """



def snr_ensemble_re_picker(st, picks, start_search_window, end_search_window,
                           start_refined_search_window,
                           end_refined_search_window,
                           refined_window_search_resolution,
                           snr_calc_pre_pick_window_len=1e-3,
                           snr_calc_post_pick_window_len=20e-3,
                           trace_padding=30e-3):
    """
    Function to improve the picks based on the SNR for an ensemble of traces.
    :param st: waveforms containing a seismic event
    :type st: :py:class:`uquake.core.stream.Stream`
    :param picks: list of :class:`uquake.core.event.Pick` object
    picks
    :type picks: list of :py:class:`uquake.core.event.Pick`
    :param start_search_window: start of the search window relative to
    provided picks in seconds
    :type start_search_window: float
    :param end_search_window: end of the search window relative to the
    provided picks in seconds
    :param start_refined_search_window: start of the refined search window in
    seconds
    :type: float
    :param end_refined_search_window: end of the refined search window in
    seconds
    :type: float
    :type end_search_window: float
    :param refined_window_search_resolution: resolution of the search space in
    seconds
    :type refined_window_search_resolution: float
    :param snr_calc_pre_pick_window_len: length of the window before the
    presumed pick
    :type snr_calc_pre_pick_window_len: float
    :param snr_calc_post_pick_window_len: length of the window after the
    presumed pick
    :type snr_calc_post_pick_window_len: float
    :param trace_padding: buffer from trace start and end time excluded from
    search window
    :returns:  Tuple comprising 1) a :py:class:`uquake.core.event.Catalog`
    a new catalog containing a single event with a list of picks and 2) the SNR
    """

    snr_dt = np.arange(start_refined_search_window, end_refined_search_window,
                       refined_window_search_resolution)

    refined_window_size = end_refined_search_window - \
                          start_refined_search_window

    biases = np.arange(start_search_window, end_search_window,
                       refined_window_size)

    st.detrend('demean')

    pre_window_length = snr_calc_pre_pick_window_len
    post_window_length = snr_calc_post_pick_window_len

    snrs_ensemble = []
    picks_ensemble = []
    for bias in biases:
        o_picks = []
        snrs = []
        for pick in picks:
            network = pick.waveform_id.network_code
            station = pick.waveform_id.station_code
            location = pick.waveform_id.location_code

            st_tmp = st.select(network=network, station=station,
                               location=location).copy()

            if len(st_tmp) == 0:
                continue

            tr = st_tmp.composite()[0]

            if tr is None:
                continue

            phase = pick.phase_hint

            earliest_time = pick.time + bias + start_refined_search_window
            latest_time = pick.time + bias + end_refined_search_window

            if earliest_time is None or earliest_time < tr.stats.starttime + \
                    trace_padding:
                earliest_time = tr.stats.starttime + trace_padding

            if latest_time is None or latest_time > tr.stats.endtime - \
                    trace_padding:
                latest_time = tr.stats.endtime - trace_padding

            tau = []
            for dt in (snr_dt + bias):
                taut = pick.time + dt

                if earliest_time <= taut <= latest_time:
                    tau.append(taut)

            if not tau:
                continue
            if tau[0] <= tr.stats.starttime:
                logger.error("Too early! tau[0]=%s <= tr.st=%s"
                             % (tau[0], tr.stats.starttime))
                logger.error("earliest_time:%s latest_time:%s"
                             % (earliest_time, latest_time))

                continue
            if tau[-1] >= tr.stats.endtime:
                logger.error("Too late! tau[-1]=%s >= tr.et=%s"
                             % (tau[-1], tr.stats.endtime))
                continue

            tau = np.array(tau)
            indices = (tau - tr.stats.starttime) * tr.stats.sampling_rate
            tmp = np.array([(taut, index,
                             calculate_snr(tr, taut,
                                           pre_wl=pre_window_length,
                                           post_wl=post_window_length))
                            for taut, index in zip(tau, indices)])

            alpha = 0

            index = np.argmax(tmp[:, 2])
            pick_time = tmp[index, 0]

            snr = calculate_snr(tr, pick_time, pre_wl=pre_window_length,
                                post_wl=post_window_length)

            method_string = 'snr_picker preWl=%.3g postWl=%.3g alpha=%.1f' % \
                            (pre_window_length, post_window_length, alpha)
            o_picks.append(make_pick(pick_time, phase=pick.phase_hint,
                                     wave_data=tr, snr=snr,
                                     method_string=method_string,
                                     resource_id=pick.resource_id))
            snrs.append(snr)

        snrs_ensemble.append(np.sum(snrs))
        picks_ensemble.append(o_picks)

    index = np.argmax(snrs_ensemble)
    output_picks = picks_ensemble[index]

    return snrs, output_picks


def extract_trace_segment(st: Stream, pick_time: UTCDateTime,
                          window_length: float):
    """
        Extract a fragment of trace and return the waveform in a matrix
        :param st: waveforms recorded by a particular site
        :type st: :py:class:`uquake.core.stream.Stream`
        :param pick_time: pick time
        :type pick_time: :py:class:uquake.core.UTCDateTime
        :param window_length: length of the window in seconds over which the
        linearity is measured
        :return: the waveform data package as a numpy ndarray
        :rtype: :py:class:numpy.ndarray
        """
    # Assuming here that all the traces recorded at a specific sensor are
    # aligned
    st[0].stats.sampling_rate

    waveforms = []
    for tr in st:
        sampling_rate = tr.stats.sampling_rate
        window_length_sample = window_length * sampling_rate
        start_measurement_window = (pick_time - tr.stats.sampling_rate)
        start_measurement_window_sample = start_measurement_window * \
                                          sampling_rate

        end_measurement_window_sample = start_measurement_window_sample + \
                                        window_length_sample
        waveforms.append(tr.data[start_measurement_window_sample:
                                 end_measurement_window_sample])

    return np.array(waveforms)



def measure_linearity(st: Stream, pick_time: UTCDateTime,
                      window_length: float):
    """
    measure the linearity of the particule motion using PCA
    :param st: waveforms recorded by a particular site
    :type st: :py:class:`uquake.core.stream.Stream`
    :param pick_time: pick time
    :type pick_time: :py:class:uquake.core.UTCDateTime
    :param window_length: length of the window in seconds over which the
    linearity is measured
    :return: the linearity between 0 and 1. 1 representing a perfect linearity
    :rtype: float
    """




    pass


def measure_planarity():
    """
    measure the planarity of the particule motion using PCA
    :param st: waveforms recorded by a particular site
    :type st: :py:class:`uquake.core.stream.Stream`
    :param pick_time: pick time
    :type pick_time: :py:class:uquake.core.UTCDateTime
    :param window_length: length of the window in seconds over which the
    linearity is measured
    :return: the linearity between 0 and 1. 1 representing a perfect planarity
    :rtype: float
    """
    pass


def calculate_snr(trace, pick, pre_wl=1e-3, post_wl=10e-3):
    """
    input :
    trs - Obspy stream
    Pick Time - in Obspy UTCDateTime
    preWl - Length of pre-window in seconds
    postWl - Length of post-window in seconds

    output:
    SNR - Signal to noise ratio
    """

    tr = trace

    sr = tr.stats.sampling_rate
    st = tr.stats.starttime
    et = tr.stats.endtime
    ps = int((pick - st) * sr)
    n_pre = int(pre_wl * sr)
    n_post = int(post_wl * sr)

    if pick + post_wl > et:
        energy_s = np.var(tr.data[ps:])
    else:
        energy_s = np.var(tr.data[ps:ps + n_post])

    if pick - pre_wl < st:
        energy_n = np.var(tr.data[:ps])
    else:
        energy_n = np.var(tr.data[ps - n_pre:ps])

    if (energy_n == 0) | (energy_s == 0):
        return 0

    snr = 10 * np.log10(energy_s / energy_n)

    return snr


def calculate_energy(stream, pick, Wl=5e-3):
    sr = stream.traces[0].stats['sampling_rate']
    st = stream.traces[0].stats['starttime']
    Ps = int((pick - st) * sr)
    Nb = int(Wl * sr)

    EnergyS = np.sum(
        [np.var(tr.data[Ps - Nb / 4:Ps + 3 * Nb / 2]) for tr in stream])

    return EnergyS


def _CalculateCF1_3(tr1, BW=None, WS=1e-3):
    if not BW:
        BW = [100, 5000]
    tr = tr1.copy()
    tr.taper(max_percentage=0.5, type='cosine')
    tr.filter(type='bandpass', freqmin=BW[0], freqmax=BW[1])

    cf1 = kurtosis(tr, win=WS)
    cf1 /= np.max(np.abs(cf1))

    cf2 = np.zeros(cf1.shape)
    dcf1 = np.diff(cf1)
    dcf1[dcf1 < 0] = 0
    cf2[0] = cf1[0]

    for k in range(1, len(cf1)):
        cf2[k] = cf2[k - 1] + dcf1[k - 1]

    try:
        cf3 = detrend(cf2, type='linear')
    except:
        cf3 = np.zeros(cf2.shape)

    return cf1, cf2, cf3


def measure_incidence_angle(st: Stream, inventory: Inventory,
                            picks: list, window_length_second: float):

    station_location_list = list(set([(tr.stats.station, tr.stats.location)
                                      for tr in st]))

    st_rotated = st.rotate('->ZNE', inventory=inventory)

    for station, location in station_location_list:
        st_ = st_rotated.select(station=station, location=location)
        p_pick = None
        s_pick = None
        for pick in picks:
            if (pick.waveform_id.station_code == station) and \
               (pick.waveform_id.location_code == location):

                if pick.phase_hint == 'P':
                    p_pick = pick

                elif pick.phase_hint == 'S':
                    s_pick = pick

        if p_pick is not None:
            st_.trim(starttime=p_pick.time,
                     endtime=p_pick.time + window_length_second)

            



