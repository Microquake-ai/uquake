""" Waveform amplitude measurements

    This module contains a collection of functions for making
    measurements on the velocity and displacement waveforms
    that are later used to calculate moment magnitude, focal mechanism, etc

"""

# default logger
from uquake.helpers.logging import logger

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import rfft

from uquake.core.data.inventory import get_sensor_type_from_trace
from uquake.core.event import Pick
from uquake.core.stream import Stream
from uquake.core.util.tools import copy_picks_to_dict
from uquake.waveform.parseval_utils import npow2, unpack_rfft
from uquake.waveform.pick import calculate_snr


def measure_pick_amps(st_in, cat, phase_list=None,
                      triaxial_only=False,
                      **kwargs):
    """
    Attempt to measure velocity pulse parameters (polarity, peak vel, etc)
      and displacement pulse parameters (pulse width, area)
      for each arrival for each event preferred_origin in cat

    Measures are made on individual traces, saved to arrival.traces[trace id],
      and later combined to one measurement per arrival
      and added to the *arrival* extras dict

    :param st_in: velocity traces
    :type st_in: obspy.core.Stream or uquake.core.Stream
    :param cat: obspy.core.event.Catalog
    :type cat: list of obspy.core.event.Events or uquake.core.event.Events
    :param phase_list: ['P'], ['S'], or ['P', 'S'] - list of arrival phases
    to process
    :type phase_list: list
    :param triaxial_only: if True --> only keep 3-component observations (
    disp area) in arrival dict
    :type triaxial_only: boolean
    """

    fname = "measure_pick_amps"

    st = st_in.copy()

    cat = measure_velocity_pulse(st, cat, phase_list=phase_list, **kwargs)

    debug = False

    if 'debug' in kwargs:
        debug = kwargs['debug']

    cat = measure_displacement_pulse(st, cat, phase_list=phase_list,
                                     debug=debug)

    # Combine individual trace measurements (peak_vel, dis_pulse_area, etc)
    #    into one measurement per arrival:

    for event in cat:
        for phase in phase_list:
            origin = event.preferred_origin() if event.preferred_origin() \
                else event.origins[0]

            for arr in origin.arrivals:
                pk = Pick(arr.get_pick())

                if not pk:
                    continue
                sta = pk.get_sta()

                if arr.traces is not None:
                    dis_area = []
                    dis_width = []

                    for tr_id, v in arr.traces.items():

                        if v['polarity'] != 0 and tr_id[-1].upper() in ['Z',
                                                                        'P']:
                            arr.polarity = v['polarity']
                            arr.t1 = v['t1']
                            arr.t2 = v['t2']
                            arr.peak_vel = v['peak_vel']
                            arr.tpeak_vel = v['tpeak_vel']
                            arr.pulse_snr = v['pulse_snr']

                        if v['peak_dis'] is not None and tr_id[-1].upper() \
                                in ['Z', 'P']:
                            arr.peak_dis = v['peak_dis']
                            arr.max_dis = v['max_dis']
                            arr.tpeak_dis = v['tpeak_dis']
                            arr.tmax_dis = v['tmax_dis']

                        # But average vector quantities distributed over components:
                        if v['dis_pulse_area'] is not None:
                            dis_area.append(v['dis_pulse_area'])

                        if v['dis_pulse_width'] is not None:
                            dis_width.append(v['dis_pulse_width'])

                    # Here is where you could impose triaxial_only requirement
                    #  but this will filter out not only 1-chan stations, but
                    #  any stations where peak finder did not locate peak on
                    #  *all* 3 channels
                    # if triaxial_only and len(dis_area) == 3:

                    if triaxial_only and len(dis_area) == 3:
                        arr.dis_pulse_area = np.sqrt(np.sum(np.array(
                            dis_area) ** 2))
                    elif len(dis_area) > 0:
                        arr.dis_pulse_area = np.sqrt(np.sum(np.array(
                            dis_area) ** 2))

                    if len(dis_width) > 0:
                        arr.dis_pulse_width = np.mean(dis_width)

                else:
                    pass

    return cat.copy()


def measure_velocity_pulse(st,
                           cat,
                           phase_list=None,
                           pulse_min_width=.005,
                           pulse_min_snr_P=7,
                           pulse_min_snr_S=5,
                           debug=False,
                           ):
    """
    locate velocity pulse (zero crossings) near pick and measure peak amp,
        polarity, etc on it

    All measurements are added to the *arrival* extras dict

    :param st: velocity traces
    :type st: obspy.core.Stream or uquake.core.Stream
    :param cat: obspy.core.event.Catalog
    :type cat: list of obspy.core.event.Events or uquake.core.event.Events
    :param phase_list: ['P'], ['S'], or ['P', 'S'] - list of arrival phases
    to process
    :type phase_list: list
    :param pulse_min_width: Measured first pulse must be this wide to be
    retained
    :type pulse_min_width: float
    :param pulse_min_snr_P: Measure first P pulse must have snr greater
    than this
    :type pulse_min_snr_P: float
    :param pulse_min_snr_S: Measure first S pulse must have snr greater
    than this
    :type pulse_min_snr_S: float
    """

    fname = 'measure_velocity_pulse'

    traces_info = []

    if phase_list is None:
        phase_list = ['P']

    # Average of P,S min snr used for finding zeros
    min_pulse_snr = int((pulse_min_snr_P + pulse_min_snr_S) / 2)

    for event in cat:
        origin = event.preferred_origin() if event.preferred_origin() else \
            event.origins[-1]
        arrivals = origin.arrivals

        for arr in arrivals:

            phase = arr.phase

            if phase not in phase_list:
                continue

            pk = Pick(arr.get_pick())

            if pk is None:
                logger.error("%s: arr pha:%s id:%s --> Lost reference to "
                             "pick id:%s --> SKIP" %
                             (fname, arr.phase, arr.resource_id.id,
                              arr.pick_id.id))

                continue
            sta = pk.get_sta()

            trs = st.select(station=sta)

            if trs is None:
                logger.warning("%s: sta:%s has a [%s] arrival but no trace "
                               "in stream --> Skip" % (fname, sta, arr.phase))
                continue

            sensor_type = get_sensor_type_from_trace(trs[0])

            if sensor_type != "VEL":
                logger.info("%s: sta:%s sensor_type != VEL --> Skip"
                            % (fname, sta))
                continue

            arr.traces = {}

            for tr in trs:
                try:
                    tr.detrend("demean").detrend("linear")
                except Exception as e:
                    print(e)

                    continue
                data = tr.data.copy()
                ipick = int((pk.time - tr.stats.starttime) *
                            tr.stats.sampling_rate)

                polarity, vel_zeros = _find_signal_zeros(
                    tr, ipick,
                    nzeros_to_find=3,
                    min_pulse_width=pulse_min_width,
                    min_pulse_snr=min_pulse_snr,
                    debug=debug
                )

                dd = {}
                dd['polarity'] = 0
                dd['t1'] = None
                dd['t2'] = None
                dd['peak_vel'] = None
                dd['tpeak_vel'] = None
                dd['pulse_snr'] = None

                # A good pick will have the first velocity pulse located
                # between i1 and i2

                if vel_zeros is not None:
                    i1 = vel_zeros[0]
                    i2 = vel_zeros[1]
                    t1 = tr.stats.starttime + float(i1 * tr.stats.delta)
                    t2 = tr.stats.starttime + float(i2 * tr.stats.delta)

                    ipeak, peak_vel = _get_peak_amp(tr, i1, i2)
                    tpeak = tr.stats.starttime + float(ipeak * tr.stats.delta)

                    noise_npts = int(.01 * tr.stats.sampling_rate)
                    noise_end = ipick - int(.005 * tr.stats.sampling_rate)
                    noise = data[noise_end - noise_npts: noise_end]
                    noise1 = np.abs(np.mean(noise))
                    noise2 = np.abs(np.median(noise))
                    noise3 = np.abs(np.std(noise))
                    noise_level = np.max([noise1, noise2, noise3])

                    pulse_snr = np.abs(peak_vel / noise_level)
                    pulse_width = float((i2 - i1) * tr.stats.delta)

                    pulse_thresh = pulse_min_snr_P

                    if phase == 'S':
                        pulse_thresh = pulse_min_snr_S

                    if pulse_snr < pulse_thresh:
                        logger.debug("%s: tr:%s pha:%s t1:%s t2:%s "
                                     "pulse_snr=%.1f < thresh" %
                                     (fname, tr.get_id(), phase, t1, t2,
                                      pulse_snr))
                        polarity = 0

                    if pulse_width < pulse_min_width:
                        logger.debug("%s: tr:%s pha:%s t1:%s t2:%s "
                                     "pulse_width=%f < %f" %
                                     (fname, tr.get_id(), phase, t1, t2,
                                      pulse_width, pulse_min_width))
                        polarity = 0

                    dd['polarity'] = polarity
                    dd['peak_vel'] = peak_vel
                    dd['tpeak_vel'] = tpeak
                    dd['t1'] = t1
                    dd['t2'] = t2
                    dd['pulse_snr'] = pulse_snr

                else:
                    logger.debug("%s: Unable to locate zeros for tr:%s pha:%s"
                                 % (fname, tr.get_id(), phase))

                arr.traces[tr.get_id()] = dd
                dd['trace_id'] = tr.get_id()
                dd['arrival_id'] = arr.resource_id
                dd['event_id'] = event.resource_id
                dd['origin_id'] = origin.resource_id
                traces_info.append(dd)

            # Process next phase in phase_list

        # Process tr in st

    # Process next event in cat

    return cat


def measure_displacement_pulse(st,
                               cat,
                               phase_list=None,
                               debug=False):
    """
    measure displacement pulse (area + width) for each pick on each arrival
        as needed for moment magnitude calculation

    All measurements are added to the *arrival* extras dict

    :param st: velocity traces
    :type st: obspy.core.Stream or uquake.core.Stream
    :param cat: obspy.core.event.Catalog
    :type list: list of obspy.core.event.Events or uquake.core.event.Events
    """

    fname = 'measure_displacement_pulse'

    if phase_list is None:
        phase_list = ['P']

    traces_info = []
    for event in cat:
        origin = event.preferred_origin() if event.preferred_origin() else \
            event.origins[0]
        arrivals = origin.arrivals

        for arr in arrivals:

            phase = arr.phase

            if phase not in phase_list:
                continue

            pk = Pick(arr.get_pick())

            if pk is None:
                logger.error("%s: arr pha:%s id:%s --> Lost reference to "
                             "pick id:%s --> SKIP" %
                             (fname, arr.phase, arr.resource_id.id,
                              arr.pick_id.id))

                continue
            sta = pk.get_sta()

            trs = st.select(station=sta)

            if trs is None:
                logger.warning("%s: sta:%s has a [%s] arrival but no trace "
                               "in stream --> Skip" %
                               (fname, sta, arr.phase))

                continue

            sensor_type = get_sensor_type_from_trace(trs[0])

            if sensor_type != "VEL":
                logger.info("%s: sta:%s sensor_type != VEL --> Skip"
                            % (fname, sta))

                continue

            for tr in trs:
                try:
                    tr_dis = tr.copy().detrend("demean").detrend("linear")
                    tr_dis.integrate().detrend("linear")
                except Exception as e:
                    print(e)

                    continue
                tr_dis.stats.channel = "%s.dis" % tr.stats.channel

                dd = {}
                dd['peak_dis'] = None
                dd['max_dis'] = None
                dd['tpeak_dis'] = None
                dd['tmax_dis'] = None
                dd['dis_pulse_width'] = None
                dd['dis_pulse_area'] = None

                tr_dict = arr.traces[tr.get_id()]

                polarity = tr_dict['polarity']
                t1 = tr_dict.get('t1', None)
                t2 = tr_dict.get('t2', None)

                if polarity != 0:

                    if t1 is None or t2 is None:
                        logger.error("%s: t1 or t2 is None --> You shouldn't "
                                     "be here!" % (fname))

                        continue

                    i1 = int((t1 - tr.stats.starttime) *
                             tr.stats.sampling_rate)
                    i2 = int((t2 - tr.stats.starttime) *
                             tr.stats.sampling_rate)

                    ipick = int((pk.time - tr.stats.starttime) *
                                tr.stats.sampling_rate)

                    icross = i2
                    tr_dis.data = tr_dis.data - tr_dis.data[i1]
                    # tr_dis.data = tr_dis.data - tr_dis.data[ipick]

                    dis_polarity = np.sign(tr_dis.data[icross])
                    pulse_width, pulse_area = _get_pulse_width_and_area(
                        tr_dis, i1, icross)

                    npulse = int(pulse_width * tr.stats.sampling_rate)

                    max_pulse_duration = .08
                    nmax_len = int(max_pulse_duration * tr.stats.sampling_rate)

                    if pulse_width != 0:

                        ipeak, peak_dis = _get_peak_amp(tr_dis, ipick,
                                                        ipick + npulse)
                        # max_dis = max within max_pulse_duration of pick time
                        imax, max_dis = _get_peak_amp(tr_dis, ipick,
                                                      ipick + nmax_len)

                        tmax_dis = tr.stats.starttime + float(imax *
                                                              tr.stats.delta)
                        tpeak_dis = tr.stats.starttime + float(ipeak *
                                                               tr.stats.delta)
                        tcross_dis = pk.time + pulse_width

                        dd['peak_dis'] = peak_dis
                        dd['max_dis'] = max_dis
                        dd['tpeak_dis'] = tpeak_dis
                        dd['tmax_dis'] = tmax_dis
                        dd['dis_pulse_width'] = pulse_width
                        dd['dis_pulse_area'] = pulse_area

                        if debug:
                            logger.debug("[%s] Dis pol=%d tpick=%s" %
                                         (phase, dis_polarity, pk.time))
                            logger.debug("              tpeak=%s "
                                         "peak_dis=%12.10g" %
                                         (tpeak_dis, peak_dis))
                            logger.debug("             tcross=%s" % tcross_dis)
                            logger.debug("               tmax=%s "
                                         "max_dis=%12.10g" %
                                         (tmax_dis, max_dis))
                            logger.debug("    dis pulse width=%.5f"
                                         % pulse_width)
                            logger.debug("    dis pulse  area=%12.10g"
                                         % pulse_area)

                    else:
                        logger.warning("%s: Got pulse_width=0 for tr:%s pha:%s"
                                       % (fname, tr.get_id(), phase))

                arr.traces[tr.get_id()] = dict(tr_dict, **dd)

                dd['trace_id'] = tr.get_id()
                dd['arrival_id'] = arr.resource_id
                dd['event_id'] = event.resource_id
                dd['origin_id'] = origin.resource_id
                traces_info.append(dd)

            # Process next tr in trs

        # Process next arr in arrivals

    # Process next event in cat

    return cat


def _find_signal_zeros(tr, istart, max_pulse_duration=.1, nzeros_to_find=3,
                       second_try=False, debug=False,
                       min_pulse_width=.00167, min_pulse_snr=5):
    """
    Locate zero crossing of velocity trace to locate first pulse(s)
    All measurements are added to the *arrival* extras dict

    :param tr: Individual velocity trace
    :type tr: obspy.core.Trace or uquake.core.Trace
    :param istart: index in trace.data to start searching from (eg =pick index)
    :type istart: int
    :param max_pulse_duration: maximum search window (in seconds) for end
                               of pulse
    :type max_pulse_duration: float
    :param nzeros_to_find: Number of zero crossings to find
    :type nzeros_to_find: int
    :param second_try: If true then iterate once over early/late pick to locate
                       first pulse
    :type second_try: boolean
    :param min_pulse_width: minimum allowable pulse width (in seconds)
    :type min_pulse_width: float
    :param min_pulse_snr: minimum allowable pulse snr
    :type min_pulse_snr: float
    :param debug: If true then output/plot debug msgs
    :type debug: boolean

    :returns: first_sign, zeros
    :rtype: int, np.array
    """

    fname = '_find_signal_zeros'

    data = tr.data
    sign = np.sign(data)

    i1 = -9

    if second_try:
        logger.debug("%s: This is the second try!" % fname)

    noise_tlen = .05
    noise_npts = int(noise_tlen * tr.stats.sampling_rate)
    noise_end = istart - int(.005 * tr.stats.sampling_rate)
    # noise_level = np.mean(data[noise_end - noise_npts: noise_end])
    noise = data[noise_end - noise_npts: noise_end]
    noise1 = np.abs(np.mean(noise))
    noise2 = np.abs(np.median(noise))
    noise3 = np.abs(np.std(noise))
    noise_level = np.max([noise1, noise2, noise3])
    noise_level = noise1

    # pick_snr = np.abs(data[istart]/noise_level)

    nmax_look = int(max_pulse_duration * tr.stats.sampling_rate)

    # Just used for debug
    pick_time = tr.stats.starttime + float(istart * tr.stats.delta)

    # Stage 0: Take polarity sign (s0) from first data point after
    #          after istart (=ipick) with SNR >= thresh * noise_level

    s0 = 0
    i0 = 0
    snr_thresh = 10.

    iend = istart + nmax_look
    if iend > len(data):
        iend = len(data)

    for i in range(istart, iend):
        if np.abs(data[i]) >= snr_thresh * np.abs(noise_level):
            s0 = sign[i]
            i0 = i

            break

    # Stage 1: Back up from this first high SNR point to find the earliest point
    #          with the *same* polarity.  Take this as i1 = pulse start
    i1 = i0
    s1 = s0
    snr_scale = 1.4

    if s0 and i0:
        for i in range(i0, istart - 3, -1):
            snr = np.abs(data[i] / noise_level)

            if sign[i] == s0 and snr >= snr_scale:
                # print("  sign matches --> set i1=i=%d" % i)
                i1 = i
            else:
                # print("  sign NO match --> break")

                break

    if i1 <= 0:
        logger.debug("%s: tr:%s pick_time:%s Didn't pass first test" %
                     (fname, tr.get_id(), pick_time))
        # tr.plot()

        return 0, None

    first_sign = s1

    # Stage 2: Find the first zero crossing after this
    #          And iterate for total of nzeros_to_find subsequent zeros

    zeros = np.array(np.zeros(nzeros_to_find, ), dtype=int)
    zeros[0] = i1

    t1 = tr.stats.starttime + float(i1) * tr.stats.delta
    # print("i1=%d --> t=%s" % (i1, t1))

    # TODO: Need to catch flag edge cases where we reach end of range with
    #       no zero set!
    for j in range(1, nzeros_to_find):
        # for i in range(i1, i1 + 200):

        for i in range(i1, i1 + nmax_look):
            # print("j=%d i=%d sign=%d" % (j,i,sign[i]))

            if sign[i] != s1:
                # half_per = float( (i - i1) * tr.stats.delta)
                # f = .5 / half_per
                # ipeak,peak = get_peak_amp(tr, i1, i)
                # print("sign:[%2s] t1:%s - t2:%s (T/2:%.6f f:%f) peak:%g" % \
                # (s1, t1, t2, half_per, f, peak))
                i1 = i
                s1 = sign[i]
                zeros[j] = i

                break
    t1 = tr.stats.starttime + float(zeros[0]) * tr.stats.delta
    t2 = tr.stats.starttime + float(zeros[1]) * tr.stats.delta

    # At this point, first (vel) pulse is located between zeros[0] and zeros[1]
    pulse_width = float(zeros[1] - zeros[0]) * tr.stats.delta
    ipeak, peak_vel = _get_peak_amp(tr, zeros[0], zeros[1])
    # noise_level defined this way is just for snr comparison
    noise_level = np.max([noise1, noise2, noise3])
    pulse_snr = np.abs(peak_vel / noise_level)

    logger.debug("find_zeros: sta:%s cha:%s First pulse t1:%s t2:%s ["
                 "polarity:%d] pulse_width:%f peak:%g snr:%f" %
                 (tr.stats.station, tr.stats.channel, t1, t2, first_sign,
                  pulse_width, peak_vel, pulse_snr))

    # Final gate = try to catch case of early pick on small bump preceding main
    #              arrival move istart to end of precursor bump and retry
    if ((pulse_width < min_pulse_width or pulse_snr < min_pulse_snr)
            and not second_try):
        logger.debug("Let's RUN THIS ONE AGAIN ============== tr_id:%s" %
                     tr.get_id())
        # if pulse_width < min_pulse_width and not second_try:
        istart = zeros[1]

        return _find_signal_zeros(tr, istart,
                                  max_pulse_duration=max_pulse_duration,
                                  nzeros_to_find=nzeros_to_find,
                                  second_try=True)

    # if debug:
    # tr.plot()

    return first_sign, zeros


def _get_peak_amp(tr, istart, istop):
    """
    Measure peak (signed) amplitude between istart and istop on trace
    :param tr: velocity trace
    :type tr: obspy.core.trace.Trace or uquake.core.Trace
    :param istart: pick index in trace
    :type istart: int
    :param istop: max index in trace to search
    :type istart: int
    :returns: imax, amp_max: index + value of max
    :rtype: int, float
    """

    abs_max = -1e12

    if istop < istart:
        logger.error("_get_peak_amp: istart=%d < istop=%d !" % (istart, istop))
        exit()

    for i in range(istart, istop):
        if np.abs(tr.data[i]) >= abs_max:
            abs_max = np.abs(tr.data[i])
            imax = i

    return imax, tr.data[imax]


def _get_pulse_width_and_area(tr, ipick, icross, max_pulse_duration=.08):
    """
    Measure the width & area of the arrival pulse on the displacement trace
    Start from the displacement peak index (=icross - location of first zero
            crossing of velocity)

    :param tr: displacement trace
    :type tr: obspy.core.trace.Trace or uquake.core.Trace
    :param ipick: index of pick in trace
    :type ipick: int
    :param icross: index of first zero crossing in corresponding velocity trace
    :type icross: int
    :param max_pulse_duration: max allowed duration (sec) beyond pick to search
                               for zero crossing of disp pulse
    :type max_pulse_duration: float

    return pulse_width, pulse_area
    :returns: pulse_width, pulse_area: Returns the width and area of the
                                       displacement pulse
    :rtype: float, float
    """

    fname = '_get_pulse_width_and_area'

    data = tr.data
    sign = np.sign(data)

    nmax = int(max_pulse_duration * tr.stats.sampling_rate)
    iend = ipick + nmax

    epsilon = 1e-10

    if icross >= iend:
        i = iend - 1

    for i in range(icross, iend):
        diff = np.abs(data[i] - data[ipick])

        if diff < epsilon or sign[i] != sign[icross]:
            break

        if i == iend - 1:
            logger.info("%s: Unable to locate termination of displacement "
                        "pulse for tr:%s!" % (fname, tr.get_id()))

            return 0, 0

    istop = i
    pulse_width = float(istop - ipick) * tr.stats.delta
    pulse_area = np.trapz(data[ipick:istop], dx=tr.stats.delta)

    return pulse_width, pulse_area


def set_pick_snrs(st, picks, pre_wl=.03, post_wl=.03):
    """
    This function sets the pick snr on each individual trace
     (vs the snr_picker which calculates snr on the composite trace)

    The resulting snr is stored in the tr.stats[key] dict
    where key = {'P_arrival', 'S_arrival'}

    :param st: traces
    :type st: either obspy.core.Stream or uquake.core.Stream
    :param picks: P & S picks
    :type list: list of either obspy or uquake picks
    :param pre_wl: pre pick window for noise calc
    :type float:
    :param post_wl: post pick window for signal calc
    :type float:
    """

    pick_dict = copy_picks_to_dict(picks)

    for tr in st:
        sta = tr.stats.station

        if sta in pick_dict:
            for phase in pick_dict[sta]:
                pick_time = pick_dict[sta][phase].time

                if phase == 'S':
                    snr = calculate_snr(Stream(traces=[tr]), pick_time,
                                        pre_wl=pre_wl, post_wl=post_wl)
                else:
                    snr = calculate_snr(Stream(traces=[tr]), pick_time,
                                        pre_wl=pre_wl, post_wl=post_wl)
                key = "%s_arrival" % phase

                if key not in tr.stats:
                    tr.stats[key] = {}
                tr.stats[key]['snr'] = snr
        else:
            logger.warning("set_pick_snrs: sta:%s not in pick_dict" % sta)

    return


def calc_velocity_flux(st_in,
                       cat,
                       inventory,
                       phase_list=None,
                       use_fixed_window=True,
                       pre_P=.01,
                       P_len=.05,
                       pre_S=.01,
                       S_len=.1,
                       Q=1e12,
                       correct_attenuation=False,
                       triaxial_only=True,
                       debug=False):
    """
    For each arrival (on phase_list) calculate the velocity flux using
        the corresponding traces and save to the arrival.vel_flux to
        be used in the calculation of radiated seismic energy

    :param st_in: velocity traces
    :type st_in: obspy.core.Stream or uquake.core.Stream
    :param cat: obspy.core.event.Catalog
    :type cat: list of obspy.core.event.Events or uquake.core.event.Events
    :param phase_list: ['P'], ['S'], or ['P', 'S'] - list of arrival phases
    to process
    :type phase_list: list
    :param triaxial_only: if True --> only calc flux for 3-comp stations
    :type triaxial_only: boolean
    :param Q: Anelastic Q to use for attenuation correction to flux
    :type Q: float
    :param correct_attenuation: if True, scale spec by e^-pi*f*travel-time/Q
    before summing
    :type correct_attenuation: boolean
    """

    fname = "calc_velocity_flux"

    if phase_list is None:
        phase_list = ['P', 'S']

    # MTH: Try to avoid copying cat - it causes the events to lose their
    # link to preferred_origin!
    # cat = cat_in.copy()
    # Defensive copy - not currently needed since we only trim on copies,
    # still ...
    st = st_in.copy().detrend('demean').detrend('linear')

    for event in cat:
        origin = event.preferred_origin() if event.preferred_origin() else \
            event.origins[0]

        for arr in origin.arrivals:

            phase = arr.phase

            if phase not in phase_list:
                continue

            pick = Pick(arr.get_pick())

            if pick is None:
                logger.error("%s: arr pha:%s id:%s --> Lost reference to "
                             "pick id:%s --> SKIP" %
                             (fname, arr.phase, arr.resource_id.id,
                              arr.pick_id.id))

                continue
            sta = pick.get_sta()

            trs = st.select(station=sta)

            if trs is None:
                logger.warning("%s: sta:%s has a [%s] arrival but no trace "
                               "in stream --> Skip" %
                               (fname, sta, phase))

                continue

            if triaxial_only and len(trs) != 3:
                logger.info("%s: sta:%s is not 3-comp --> Skip" % (fname, sta))

                continue

            sensor_type = get_sensor_type_from_trace(trs[0])

            if sensor_type != "VEL":
                logger.info("%s: sta:%s sensor_type != VEL --> Skip" % (
                    fname, sta))

                continue

            if use_fixed_window:
                if phase == 'P':
                    pre = pre_P
                    win_secs = P_len
                else:
                    pre = pre_S
                    win_secs = S_len

                starttime = pick.time - pre
                endtime = starttime + win_secs

            not_enough_trace = False

            for tr in trs:
                if starttime < tr.stats.starttime or endtime > \
                        tr.stats.endtime:
                    logger.warning("%s: sta:%s pha:%s tr:%s is too short to "
                                   "trim --> Don't use" %
                                   (fname, sta, phase, tr.get_id()))
                    not_enough_trace = True

                    break

            if not_enough_trace:
                continue

            tr3 = trs.copy()

            tr3.trim(starttime=starttime, endtime=endtime)
            dt = tr3[0].stats.delta

            # flux_t = np.sum( [tr.data**2 for tr in tr3]) * dt

            tsum = 0

            for tr in tr3:
                tsum += np.sum(tr.data ** 2) * dt

            if not correct_attenuation:
                flux = tsum
                fsum = None

            # The only reason to do this in the freq domain is if we
            #    want to apply attenuation correction
            else:
                travel_time = pick.time - origin.time

                fsum = 0.

                # exp(pi * f * (R/v) * 1/Q) grows so fast with freq that it's out of
                # control above f ~ 1e3 Hz
                # We could design Q = Q(f) - e.g., make Q grow fast with f to
                # counteract this.
                # Alternatively, we limit the freq range of the (
                # attenuation-corrected) energy calc:
                #   To compare the t/f calcs using Parseval's:
                #   1. Set Q to something like 1e12 in the settings
                #   2. Set fmin=0 so that the low freqs are included in the summationi
                sensor_response = inventory.select(arr.get_sta())
                poles = np.abs(sensor_response[0].response.get_paz().poles)
                fmin = np.min(poles) / (2 * np.pi)
                fmax = 500.  # looking at the response in the frequency
                # domain, beyond 500 Hz, the response is dominated by the
                # brown environmental noise.
                # In addition, it's necessary to locate fmin/fmax for each arrival
                # based on the min/max freqs where the velocity spec exceeds the
                # noise spec.
                # Thus, we can't actually include all the radiated energy, just the
                # energy above the noise level
                #   so these estimates will likely be low
                fmin = arr.fmin
                fmax = arr.fmax

                if fmax / fmin < 3:
                    logger.info("%s: sta:%s fmin:%.1f fmax:%.1f too "
                                "narrowband --> Skip"
                                % (fname, sta, fmin, fmax))

                    continue

                for tr in tr3:
                    data = tr.data
                    nfft = 2 * npow2(data.size)
                    df = 1. / (dt * float(nfft))  # df is same as for
                    y, freqs = unpack_rfft(rfft(data, n=nfft), df)
                    y *= dt
                    y[1:-1] *= np.sqrt(2.)

                    tstar = travel_time / Q

                    index = [(freqs >= fmin) & (freqs <= fmax)]
                    freqs = freqs[index]
                    y = y[index]
                    fsum += np.sum(np.abs(y) * np.abs(y) * np.exp(
                        2. * np.pi * freqs * tstar)) * df

                print("arr sta:%s [%s] tsum=%g fsum=%g fmin=%f fmax=%f" %
                      (sta, arr.phase, tsum, fsum, fmin, fmax))
                # exit()

                flux = fsum

            # Note we are saving the "flux" but it has not (yet) been scaled
            # by rho*vel. Instead it is just the sum of the integrals of the
            # component velocities squared for the corresponding arrival and
            # scaling is done in calc_energy(..)

            # arr.vel_flux = flux
            arr.vel_flux = tsum
            arr.vel_flux_Q = fsum

    return cat


def plot_spec(freqs, spec, tstar, title=None):
    corrected_spec = spec * np.exp(2. * np.pi * freqs * tstar)

    plt.loglog(freqs, spec, color='blue')
    plt.loglog(freqs, corrected_spec, color='green')

    if title:
        plt.title(title)
    plt.grid()
    plt.show()

    return
