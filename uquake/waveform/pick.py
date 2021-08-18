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
from obspy.signal.trigger import recursive_sta_lta, pk_baer, \
    classic_sta_lta, coincidence_trigger
from obspy.realtime.signal import kurtosis
from scipy.signal import detrend
from scipy.ndimage.filters import gaussian_filter1d
from ..core import event
from ..core.util.decorator import deprecated
from ..core.logging import logger

from ..core.event import make_pick
from ..core.util.tools import copy_picks_to_dict


def measure_polarity(st, catalog, site, average_time_window=1e-3,
                     hp_filter_freq=100):
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

            trs.filter('highpass', freq=hp_filter_freq)
            if len(trs) > 1:
                logger.warning("number of trace for station %s and channel %s"
                               "is greater than 1. Only the first trace will be used" % (
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
    The pick polarity is measured on the displacement trace looking at the difference
    between the amplitude at the pick time and the sign of the average amplitude
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
        Noise and signal energy are calculated over the first and second half of
        the pick window, respectively.
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


# noinspection PyProtectedMember
def STALTA_picker_refraction(st, nsta=1e-3, nlta=4e-2, fc=50, nc=2,
                             noise_mag=1e-25, SNR_window=5e-3):
    """The STA/LTA picker provides a first order good estimate of the arrival
    times for both the P- and S- wave. The STA/LTA picker uses
    :py:class:`obspy.signal.trigger.recursive_sta_lta`.

    :param st: seismogram containing a seismic event
    :type st: :py:class:`obspy.core.stream.Stream`
    :param nsta: Length of the short term average (STA) window in seconds.
    The value
        must be smaller than the minimum separation between the P- and S-
        wave arrival (explanation to be improved).
    :param nlta: Length of the long term average (LTA) window in seconds. The value
        must be long enough to capture well the noise but not too long (
        description to be improved)
    :param fc: Corner frequency of the Gaussian kernel used to smooth the
    LTA/STA
        function in Hz. This parameter must be chosen to limit the number of
        peaks in the STA/LTA function.
    :param nc: Number of desired onset. *** PARAMETER NOT USED ***
    :param noise_mag: Amplitude of the noise that is add to the original signal
    :param SNR_window: Length of window in seconds in which the SNR is
    calculated before and after the pick
    :returns:  :py:class:`obspy.core.event.Catalog` -- a new catalog
    containing a single event with a list of picks
    """

    # nphase = 1
    st.detrend('demean')
    stations = np.unique([tr.stats.station for tr in st])

    # for station in stations:
    opicks = []
    cfs = []
    snrs = []
    for station in stations:

        trs = st.select(station=station)
        # if len(trs) == 3:
        #   trs = _RotateP_S(trs)

        # else:
        #   continue

        sta = int(nsta * trs.traces[0].stats['sampling_rate'])
        lta = int(nlta * trs.traces[0].stats['sampling_rate'])

        # Noise added to avoid zeroed trace.
        rd = np.random.randn(len(trs.traces[0].data)) * noise_mag
        trstmp = [Trace(
            data=np.hstack(
                (tr.data[::-1].reshape(len(tr)) + rd[::-1].reshape(len(rd)),
                 tr.data.reshape(len(tr)) + rd.reshape(len(rd)))),
            header=tr.stats) for tr in trs]
        sttmp = Stream(traces=trstmp)

        # tr.data = tr.data + np.random.randn(len(tr.data)) * 1e-10

        StaLta = np.sum([classic_sta_lta(tr, sta, lta) for tr in sttmp],
                        axis=0)
        StaLta = StaLta[len(StaLta) / 2::]

        sfreq = trs.traces[0].stats['sampling_rate']
        sigma = sfreq / (2 * np.pi * fc)
        StaLtaf = gaussian_filter1d(StaLta, sigma=sigma, mode='reflect')

        picks = _Pick_STALTA_refraction(trs, StaLtaf)
        snrs = [calculate_snr(trs, p, SNR_window) for p in picks]

        cfs.append(StaLtaf)
        snrs.append(snrs)

        for k, p in enumerate(picks):
            if k == 0:
                opicks.append(make_pick(p, 'P', trs.traces[0], snrs[k]))
            elif k == 1:
                opicks.append(make_pick(p, 'S', trs.traces[0], snrs[k]))
            else:
                opicks.append(make_pick(p, '?', trs.traces[0], snrs[k]))

    catalog = event.Catalog()
    catalog.events.append(event.Event(picks=opicks))

    return catalog, cfs, snrs


def STALTA_picker(st, nphase=2, nsta=1e-3, nlta=4e-2, fc=50, nc=2,
                  noise_mag=1e-25, SNR_window=5e-3):
    """The STA/LTA picker provides a first order good estimate of the arrival
    times for both the P- and S- wave. The STA/LTA picker uses
    :py:class:`obspy.signal.trigger.recursive_sta_lta `.

    :param st: seismogram containing a seismic event
    :type st: :py:class:`obspy.core.stream.Stream`
    :param nphase: Number of phases to pick. The number of phase should
    represent the number of phase present in the signal. This parameter will
    be set-up to two for picking the P- and S-wave onset times.
    :param nsta: Length of the short term average (STA) window in seconds. The value
        must be smaller than the minimum separation between the P- and S-
        wave arrival (explanation to be improved).
    :param nlta: Length of the long term average (LTA) window in seconds. The value
        must be long enough to capture well the noise but not too long (
        description to be improved)
    :param fc: Corner frequency of the Gaussian kernel used to smooth the LTA/STA
        function in Hz. This parameter must be chosen to limit the number of
        peaks in the STA/LTA function (default 50 Hz).
    :param nc: Number of desired onset. *** PARAMETER NOT USED ***
    :param noise_mag: Amplitude of the noise to be added to the original signal
    :param SNR_window: Length of window in seconds in which the SNR is
    calculated before and after the pick
    :returns:  :py:class:`obspy.core.event.Catalog` -- a new catalog
    containing a single event with a list of picks
    """

    st.detrend('demean')
    stations = np.unique([tr.stats.station for tr in st])

    # for station in stations:
    opicks = []
    cfs = []
    snrs = []
    for station in stations:

        trs = st.select(station=station)
        # if len(trs) == 3:
        #   trs = _RotateP_S(trs)

        # else:
        #   continue

        sta = int(nsta * trs.traces[0].stats['sampling_rate'])
        lta = int(nlta * trs.traces[0].stats['sampling_rate'])

        # Noise added to avoid zeroed trace.
        rd = np.random.randn(len(trs.traces[0].data)) * noise_mag
        trstmp = [Trace(
            data=np.hstack(
                (tr.data[::-1].reshape(len(tr)) + rd[::-1].reshape(len(rd)),
                 tr.data.reshape(len(tr)) + rd.reshape(len(rd)))),
            header=tr.stats) for tr in trs]
        sttmp = Stream(traces=trstmp)

        # tr.data = tr.data + np.random.randn(len(tr.data)) * 1e-10

        StaLta = np.sum([recursive_sta_lta(tr, sta, lta) for tr in sttmp],
                        axis=0)
        StaLta = StaLta[len(StaLta) / 2::]

        sfreq = trs.traces[0].stats['sampling_rate']
        sigma = sfreq / (2 * np.pi * fc)
        StaLtaf = gaussian_filter1d(StaLta, sigma=sigma, mode='reflect')

        picks = _Pick_STALTA(trs, StaLtaf, nphase)
        SNRs = [calculate_snr(trs, p, SNR_window) for p in picks]

        # print picks[0], (picks - trs[0].stats.starttime) * trs[0].stats.sampling_rate, SNRs

        cfs.append(StaLtaf)
        snrs.append(SNRs)

        for k, p in enumerate(picks):
            if k == 0:
                opicks.append(make_pick(p, 'P', trs.traces[0], SNRs[k]))
            elif k == 1:
                opicks.append(make_pick(p, 'S', trs.traces[0], SNRs[k]))
            else:
                opicks.append(make_pick(p, '?', trs.traces[0], SNRs[k]))

    catalog = event.Catalog()
    catalog.events.append(event.Event(picks=opicks))

    return catalog, cfs, snrs


def kurtosis_picker(st, picks, freqmin=100, freqmax=1000, pick_freqs=None,
                    kurtosis_window=None, CF3_tol=10e-3, SNR_window=5e-3):
    """Kurtosis picker adapted for microseismic event processing
    from ``Baillard et al. 2014``.

    :param st: seismogram containing a seismic event
    :type st: :py:class:`obspy.core.stream.Stream`
    :param cat: catalog containing the event with previous picks
    :type cat: obspy.core.event.Catalog
    :param freqmin: Low end of frequency band used to prefilter the seismograms.
        This value depends on the type of sensor.
        The default value is optimized for 2:3 kHz accelerometers.
    :param freqmax: High end of frequency band used to prefilter the seismograms.
        This value depends on the type of sensor.
        The default value is optimized for 2:3 kHz accelerometers.
    :param pick_freqs: The smoothing frequencies for the smoothing
        window applied to the CF3 function (this is used instead of Ns).
    :param kurtosis_window: Windows lengths used to calculate CF3 up to P-S delay
    :param CF3_tol: Maximum time to move a pick
    :param SNR_window: Length of window in seconds in which the SNR is calculated before and after the pick
    :returns:  :py:class:`obspy.core.event.Catalog` -- a new catalog containing a single event with a list of picks
    """

    '''
    if not cat.events:
        return None

    evt = cat.events[0]

    if not evt['picks']:
        return None

    prevPicks = evt['picks']
    '''
    prevPicks = copy_picks_to_dict(picks)

    # # clip the signal to the existing picks
    # if prevPicks:
    #   p_early = UTCDateTime(2100, 1, 1)
    #   p_late = UTCDateTime(1970, 1, 1)

    #   for picks in prevPicks:
    #       if picks['time'] < p_early:
    #           p_early = picks['time']
    #       if picks['time'] > p_late:
    #           p_late = picks['time']

    #   p_early -= 25e-3
    #   p_late += 25e-3
    #   st.trim(starttime=p_early, endtime=p_late)

    if pick_freqs is None:
        pick_freqs = np.linspace(50, 1000, 20)
    if kurtosis_window is None:
        kurtosis_window = np.array([1, 2, 3]) * 1e-3

    st.detrend('demean')
    stations = np.unique([tr.stats.station for tr in st])

    opicks = []
    cfs = []
    snrs = []
    for station in stations:
        # find the existing P and S picks for the current station
        trs = st.select(station=station)
        # print('kurtosis: sta:%s ntr=%d' % (station, len(trs)))
        # if len(trs) < 3:
        # continue
        CF3 = np.zeros(len(trs.traces[0].data))

        for ws in kurtosis_window:
            for tr in trs:
                cf1, cf2, cf3 = _CalculateCF1_3(tr, WS=ws,
                                                BW=[freqmin, freqmax])
                CF3 += cf3

        cfs.append(CF3)

        # TODO:
        # Picker should probably work with no previous picks.
        # If the current station has not been picked before, a new pick will not be generated.
        # If the old pick has only either P or S, then only a new corresponding P or S will be generated, not both.
        if not prevPicks:
            continue

        for phase in prevPicks[station]:
            oldPick = prevPicks[station][phase]

            # for oldPick in prevPicks:
            # if oldPick['waveform_id'].station_code != station:
            # continue

            pick = _Pick_CF3(trs, CF3, oldPick['time'], pick_freqs, CF3_tol)
            SNR = calculate_snr(trs, pick, SNR_window)
            snrs.append(SNR)
            opicks.append(
                make_pick(pick, oldPick['phase_hint'], trs.traces[0], SNR))

            print('kurtosis: sta:%s pha:%s old_pick:%s new:%s' % (
            station, phase, oldPick.time, opicks[-1].time))

    # catalog = event.Catalog()
    # catalog.events.append(event.Event(picks=opicks))

    # return catalog, cfs, snrs
    return opicks


def snr_picker(st, picks, snr_dt=None, snr_window=(1e-3, 20e-3), filter=None):
    """
    Function to improve the picks based on the SNR.
    :param st: seismogram containing a seismic event
    :type st: :py:class:`obspy.core.stream.Stream`
    :param picks: list of uquake.core.event.Pick object
    picks
    :type picks: uquake.core.event.Catalog
    :param snr_dt: Window in which the picks will be improved.
    :param snr_window: Length of window in seconds in which the SNR is calculated
    before and after the pick
    :type snr_window: (tuple)
    :returns:  Tuple comprising 1) a :py:class:`uquake.core.event.Catalog`
    a new catalog containing a single event with a list of picks and 2) the SNR
    """

    function_name = 'snr_picker'

    filter_p = False
    filter_s = False
    if filter == 'S':
        filter_p = True
    elif filter == 'P':
        filter_s = True

    previous_picks = copy_picks_to_dict(picks)

    if snr_dt is None:
        snr_dt = np.linspace(-5e-3, 5e-3, 20)

    st.detrend('demean')
    stations = np.unique([tr.stats.station for tr in st])

    opicks = []
    snrs = []

    pre_window_length = snr_window[0]
    post_window_length = snr_window[1]

    for station in stations:

        tr = st.select(station=station).composite()[0]

        if station not in previous_picks:
            logger.warning('SNR_detect: station:[%s] has no previous picks'
                           % station)
            continue

        for phase in previous_picks[station]:

            if filter_p and phase == 'P':
                continue
            elif filter_s and phase == 'S':
                continue

            earliest_time = latest_time = None
            if phase == 'S' and 'P' in previous_picks[station]:
                delta = 1 / 2 * (previous_picks[station]['S'].time -
                                 previous_picks[station]['P'].time)
                earliest_time = previous_picks[station]['S'].time - delta

            elif phase == 'S' and 'P' not in previous_picks[station]:
                earliest_time = previous_picks[station]['S'].time - \
                                pre_window_length
            elif phase == 'P' and 'S' in previous_picks[station]:
                delta = 1 / 2 * (previous_picks[station]['S'].time -
                                 previous_picks[station]['P'].time)
                latest_time = previous_picks[station]['P'].time + delta
            elif phase == 'P' and 'S' not in previous_picks[station]:
                latest_time = previous_picks[station]['P'].time + \
                              post_window_length

            old_pick = previous_picks[station][phase]

            if earliest_time is None or earliest_time < tr.stats.starttime + \
                    .03:
                earliest_time = tr.stats.starttime + .03

            if latest_time is None or latest_time > tr.stats.endtime - .06:
                latest_time = tr.stats.endtime - .06

            tau = []
            for dt in snr_dt:
                taut = old_pick.time + dt

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
            tmp = np.array([(taut, index, calculate_snr(tr, taut,
                                                        pre_wl=pre_window_length,
                                                        post_wl=post_window_length))
                            for taut, index in zip(tau, indices)])

            # MTH: this is a hack to try to force the solution close to the oldPick
            alpha = 0
            """
            alpha = 10.
            for i, foo in enumerate(tmp):
                time = foo[0]
                snr = foo[1]
                dt = np.abs(old_pick.time - foo[0])
                #dt = np.abs(oldPick['time'] - foo[0])
                scale = np.exp(-alpha * dt)
                tmp[i,1] *= np.exp(-alpha * dt)
                #print(time, dt, snr, scale, snr*scale, tmp[i,1])
            """

            index = np.argmax(tmp[:, 2])
            pick_time = tmp[index, 0]

            import matplotlib.pyplot as plt

            snr = calculate_snr(tr, pick_time, pre_wl=pre_window_length,
                                post_wl=post_window_length)

            # plt.plot(tmp[:, 1], tmp[:, 2] / np.max(tmp[:, 2]))
            # plt.plot(tr.data / np.max(tr.data))
            # plt.axvline(tmp[index, 1], color='r', ls='--')
            # plt.xlim([tmp[0, 1], tmp[-1, 1]])
            # plt.show()

            # from ipdb import set_trace; set_trace()
            logger.debug("%s: sta:%s [%s] time_diff:%0.3f SNR:%.2f" %
                         (function_name, station, phase, old_pick.time -
                          pick_time, snr))

            method_string = 'snr_picker preWl=%.3g postWl=%.3g alpha=%.1f' % \
                            (pre_window_length, post_window_length, alpha)
            opicks.append(make_pick(pick_time, phase=old_pick.phase_hint,
                                    wave_data=tr, snr=snr,
                                    method_string=method_string,
                                    resource_id=old_pick.resource_id))
            snrs.append(snr)

    return snrs, opicks


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


def _Pick_STALTA(st, stalta, nphase):
    # Finding the two largest maximum of the smoothed STALTA
    # function. The two largest maximum are used as starting pick
    # values.

    sr = st.traces[0].stats['sampling_rate']
    starttime = st.traces[0].stats['starttime']
    endtime = st.traces[0].stats['endtime']

    buf = (
                      endtime - starttime) * 0.0001  # picks cannot be in the first and last 1% of the stream

    mx = np.r_[True, stalta[1:] > stalta[:-1]] & np.r_[
        stalta[:-1] > stalta[1:], True]

    i1 = np.nonzero(mx)[0]

    i2 = np.argsort(stalta[i1])[::-1]
    # i2 = i1

    # EnergySs = np.array([calculate_energy(st, starttime + i1[k] / sr, Wl=5e-3) for k in i2])

    # Eratio = EnergySs / np.max(EnergySs)
    # ie = np.nonzero(Eratio > 0.05)[0]
    # i2 = i2[ie]

    picks = []
    for k in range(0, nphase):
        try:
            picks.append(i1[i2[k]])
        except:
            pass
            # logger.warning("_Pick_STALTA: station=%s phase=%d FAILED!" % (st.traces[0].stats.station, k))
            # picks.append(int((endtime - starttime)*sr))

    # if len(picks) < nphase:
    #   logger.warning("_Pick_STALTA: Not all phases were picked")

    picks = np.sort([starttime + p / sr for p in picks])
    picks = np.array(
        [p for p in picks if ((starttime + buf) < p < (endtime - buf))])

    return picks


def _Pick_STALTA_refraction(st, stalta):
    # Finding the two largest maximum of the smoothed STALTA
    # function. The two largest maximum are used as starting pick
    # values.

    sr = st.traces[0].stats['sampling_rate']
    starttime = st.traces[0].stats['starttime']
    # endtime = st.traces[0].stats['endtime']

    mx = np.r_[True, stalta[1:] > stalta[:-1]] & np.r_[
        stalta[:-1] > stalta[1:], True]

    i1 = np.nonzero(mx)[0]

    i2 = 0  # np.argsort(stalta[i1])[::-1]
    # i2 = i1

    # EnergySs = np.array([calculate_energy(st,starttime+i1[k]/sr,Wl = 5e-3) for k in i2])

    # Eratio = EnergySs/np.max(EnergySs)
    # ie = np.nonzero(Eratio > 0.05)[0]
    # i2 = i2[ie]

    picks = [i1[i2]]

    picks = np.sort([starttime + p / sr for p in picks])

    return picks


# This should work with only one pick at a time regardless whether it is a P or S pick.
def _Pick_CF3(st, cf3, iniPick, f=np.linspace(50, 1000, 20), tol=10e-3):
    """
    input :
    st - stream object
    cf3 - characteristic function number 3
    f - list of frequency (used instead of the Ns parameter)
    tol - maximum time to move a pick
    """

    sr = st.traces[0].stats['sampling_rate']
    ST = st.traces[0].stats['starttime']
    # first = True
    CF3_2 = np.hstack((cf3, cf3[::-1])) - cf3[0]
    CF3_2TR = Trace(data=CF3_2, header=st.traces[0].stats)  # ??

    Pick = int((iniPick - ST) * sr)  # initial pick in sample
    Pick_stalta = Pick

    for freq in f:
        # sigma = sfreq/(2*np.pi*freq)
        CF3_2TR_F = CF3_2TR.copy()
        CF3_2TR_F.filter('lowpass', freq=freq)
        # CF3_f = gaussian_filter1d(CF3,sigma=sigma,mode='reflect')
        CF3_f = CF3_2TR_F.data
        # CF3_f = CF3_f[:len(CF3_f) / 2]
        CF3_f = CF3_f[:int(len(CF3_f) / 2)]
        s = np.r_[True, CF3_f[1:] < CF3_f[:-1]] & np.r_[
            CF3_f[:-1] < CF3_f[1:], True]
        indices = np.nonzero(s)[0]
        CF4 = np.zeros(CF3_f.shape)
        for i in indices[0:-1]:
            CF4[i] = CF3_f[i] - CF3_f[i + 1]

        indices = np.nonzero(CF4)[0]

        try:
            pick_tmp = indices[
                np.argmin(np.abs(Pick - indices[indices <= Pick_stalta]))]
        except:
            pick_tmp = Pick

        if np.abs(pick_tmp - Pick_stalta) <= (tol * sr):
            Pick = pick_tmp

    # Return pick time in samples
    pick = ST + Pick / sr
    return pick


@deprecated
def triggersByGroup(st, trigger_type="recstalta", group="station", thr_on=3,
                    thr_off=2,
                    thr_coincidence_sum=1, sta=0.01, lta=1):
    # Accentuate peaks - bug below, skip for now
    # for i,tr in enumerate(st):
    #   st2.traces[0].data = tr.data**2*np.sign(tr.data)

    gp = np.array([])

    if group == "all":
        # one trigger for all stations
        trig = coincidence_trigger(trigger_type, thr_on, thr_off, st,
                                   thr_coincidence_sum, sta=sta, lta=lta,
                                   details=True)

    elif group == "station":
        # one trigger per station
        trig = []

        SensorList = []
        for tr in st:
            SensorList.append(tr.stats.station)
        SensorList = np.unique(SensorList)

        for k, S in enumerate(SensorList):
            st2 = st.select(station=S)
            try:
                trigtmp = coincidence_trigger(trigger_type, thr_on, thr_off,
                                              st2, thr_coincidence_sum,
                                              sta=sta, lta=lta, details=True)
            except:
                trigtmp = []

            # plot - debug
            # for tr in st2:
            #   cft = classic_sta_lta(tr.data, int(sta * tr.stats.sampling_rate), int(lta * tr.stats.sampling_rate))
            #   plotTrigger(tr, cft, thr_on, thr_off)
            # rpdb.set_trace()

            if k == 0:
                trig = trigtmp
            else:
                trig = trig + trigtmp

            gp = np.hstack((gp, np.ones(len(trigtmp)) * k))
    else:

        # individual recursive_sta_lta for each trace
        trig = []
        pass

    return trig, gp


@deprecated
def associateTriggers(trig, gp, tolerance=25e-3):
    tme = np.sort([trg['time'] for trg in trig])

    trigger = []

    # at least 2 sensors need to be in the same tolerance window
    indices2 = np.argsort([trg['time'] for trg in trig])
    gp = gp[indices2]
    trig = np.array(trig)
    trig = trig[indices2]

    k = 0
    while k < len(tme):
        indices = np.nonzero((tme - tme[k] > 0) & (tme - tme[k] < tolerance))[
            0]
        if len(indices) > 0:
            k = indices[-1]
            gp2 = gp[indices]
            if len(np.unique(gp2)) > 1:
                trigs = trig[indices]
                best_trig = np.argmax([t['cft_peak_wmean'] for t in trigs])
                best_trig = trigs[best_trig]

                data = {'mean_time': best_trig['time'],
                        'data': trigs}

                trigger.append(data)
        else:
            k += 1

    return trigger


@deprecated
def picksFromTriggers(st, trg, method="by_triggers", tolerance=20e-3,
                      clip_stream=True, filter_stream=True):
    picks = []

    if method == "all":
        SensorList = []
        for tr in st:
            SensorList.append(tr.stats.station)

        SensorList = np.unique(SensorList)
        for S in SensorList:
            st2 = st.select(station=S)

            picks2 = compute_picks(st2, trg, tolerance, clip_stream,
                                   filter_stream)
            if picks2 is not None:
                picks = np.hstack((picks, picks2))

    elif method == "by_triggers":
        for tg in trg:
            SensorList = np.unique(tg['stations'])
            for S in SensorList:
                st2 = st.select(station=S)

                picks2 = compute_picks(st2, tg, tolerance, clip_stream,
                                       filter_stream)
                if picks2 is not None:
                    picks = np.hstack((picks, picks2))

    return picks


@deprecated
def compute_picks(st2, trg, tolerance=20e-3, clip_stream=True,
                  filter_stream=True):
    picks = []

    tg_time = trg['time']
    if clip_stream:
        st = st2.trim(tg_time - tolerance, tg_time + tolerance)
        if filter_stream:
            st = st.filter_stream(lf=100, hf=1000, copy=False)
    else:
        if filter_stream:
            st = st2.filter_stream(lf=100, hf=1000, copy=True)
        else:
            st = st2

    starttime = st.traces[0].stats.starttime
    # starttime2 = st2.traces[0].stats.starttime
    sr = st.traces[0].stats.sampling_rate
    station_type = st.traces[0].stats.station_type

    try:
        if len(st) < 3:
            data = st.traces[0].data ** 2 * np.sign(st.traces[0].data)
        else:
            data = (st.traces[0].data ** 2 +
                    st.traces[1].data ** 2 +
                    st.traces[2].data ** 2) * \
                   np.sign(st.traces[0].data)
        i = np.argmax(np.abs(data))
        data = data / data[i]

    except:
        return

    if 'A' in station_type:
        # Parameters originally estimated for Northparkes TBM project (which had a sampling rate of around 5000) -> Correction x2
        pickBaer = pk_baer(reltrc=data, samp_int=1, tdownmax=2, tupevent=40,
                           thr1=10, thr2=20, preset_len=10, p_dur=10)
        # pickBaer = pk_baer(reltrc=data, samp_int=1, tdownmax=16, tupevent=60, thr1=10, thr2=20, preset_len=20, p_dur=20)
    else:
        # Parameters originally estimated for Northparkes TBM project (which had a sampling rate of around 2500) -> Correction x4
        pickBaer = pk_baer(reltrc=data, samp_int=1, tdownmax=4, tupevent=10,
                           thr1=10, thr2=20, preset_len=10, p_dur=10)
        # pickBaer = pk_baer(reltrc=data, samp_int=1, tdownmax=32, tupevent=40, thr1=10, thr2=20, preset_len=10, p_dur=10)

    # plot - debug
    # if st.traces[0].stats.station == '14_A1':
    # trg_time = (trg['time'] - starttime)*sr
    # diff_samples = (starttime - starttime2)*sr
    # num_traces = len(st)
    # plt.close()
    # # for k, (tr, tr2) in enumerate(zip(st, st3.select(station=st.traces[0].stats.station))):
    # for k, (tr, tr2) in enumerate(zip(st, st2)):
    #   trace.plot_traces(tr, k, num_traces*2, pickSample=[trg_time, pickBaer[0]])
    #   trace.plot_traces(tr2, k+num_traces, num_traces*2, pickSample=[trg_time+diff_samples, pickBaer[0]+diff_samples])
    # plt.show()

    if pickBaer[0] < 10:  # pick can't occur in the first N samples
        return

    Noise = np.var(data[0:pickBaer[0]])
    Signal = np.var(data[pickBaer[0]:pickBaer[0] + 50])
    # SNR = 10*np.log((Signal-Noise)/(Noise+1e-10))
    SNR = 10 * np.log10(Signal / (Noise + 1e-10))
    # print SNR, pickBaer

    if SNR < 3:
        return

    t = starttime + pickBaer[0] / sr

    for tr in st:
        this_pick = event.Pick()
        this_pick.time = t
        this_pick.phase_hint = 'P'
        this_pick.waveform_id = event.WaveformStreamID(
            network_code=tr.stats.network,
            station_code=tr.stats.station,
            location_code=tr.stats.location,
            channel_code=tr.stats.channel)
        this_pick.evaluation_mode = 'automatic'
        # this_pick.creation_info = creation_info
        if 'E' in pickBaer[1]:
            this_pick.onset = 'emergent'
        if 'I' in pickBaer[1]:
            this_pick.onset = 'impulsive'
        this_pick.evaluation_status = 'preliminary'

        if len(st) < 3:
            if 'U' in pickBaer[1]:
                this_pick.polarity = 'positive'
            elif 'D' in pickBaer[1]:
                this_pick.polarity = 'negative'
        else:
            d_0 = tr.data[pickBaer[0]]
            d_1 = tr.data[pickBaer[0] + 3]
            if d_1 > d_0:
                this_pick.polarity = 'positive'
            else:
                this_pick.polarity = 'negative'

        this_pick.SNR = SNR

        picks.append(this_pick)

    return picks


def eventCategorization_polarity(catalog, site):
    """
    determine the event category by looking at the polarity of the first motion.
    It is assumed that blast will generate mostly positive first motion whereas seismic event
    will generate mixte first motion

    :param st: seismograms
    :type st: obspy.core.Stream or uquake.core.Stream
    :param catalog: events catalog
    :type catalog: obspy.core.event.Catalog or uquake.core.event.Catalog
    :param site: information on network
    :type site: uquake.core.data.station.Site
    """

    catalog = event.Catalog(cat=catalog)
    for evi, evt in enumerate(catalog.events):
        picks = evt.picks
        polarity = []
        for origin in evt.origins:
            evloc = np.array([origin.x, origin.y, origin.z])
            for pick in picks:
                if not pick.polarity:
                    continue
                if pick.polarity.lower() == "positive":
                    pick_polarity = 1
                elif pick.polarity.lower() == 'negative':
                    pick_polarity = -1
                else:
                    continue

                sta_code = pick.waveform_id.station_code
                station = site.stations(station=sta_code)[0]
                stloc = station.loc
                ev_st_vect = stloc - evloc
                for channel in station:
                    if not np.any(channel.orientation):
                        continue

                    polarity.append(
                        np.sign(np.dot(channel.orientation, ev_st_vect)))

        polarity = np.array(polarity)
        if len(polarity[polarity == 1]) >= 0.85 * len(polarity):
            catalog.events[evi].event_type = "mining explosion"
        else:
            catalog.events[evi].event_type = "induced or triggered event"

        catalog.events[evi].event_type_certainty = "suspected"

    return catalog
