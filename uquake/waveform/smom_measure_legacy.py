import numpy as np

from scipy import optimize
from scipy.fftpack import fft, fftfreq, rfft, rfftfreq

from uquake.core.data.inventory import (get_sensor_type_from_trace,
                                        get_corner_freq_from_pole)
from uquake.core.util.tools import copy_picks_to_dict

import matplotlib.pyplot as plt
import copy
from uquake.helpers.logging import logger

"""
    smom_measures - A collection of functions used in the calculation of
                     smom (long-period plateau) in the frequency domain
"""


def measure_pick_smom(st, inventory, event, synthetic_picks,
                      fmin=20, fmax=1000,
                      use_fixed_fmin_fmax=False,
                      plot_fit=False,
                      P_or_S='P',
                      debug_level=0,
                      **kwargs):
    """
        Calculate spectra of each channel/arrival and compare to Brune model to fit
        for long_period plateau (smom)
    """

    fname = "measure_pick_smom"

    # Get P(S) spectra at all stations/channels that have a P(S) arrival:
    sta_dict = get_spectra(st, event, inventory, synthetic_picks,
                           calc_displacement=False, S_win_len=.1,
                           P_or_S=P_or_S)

    vel_dict = copy.deepcopy(sta_dict)

    # Stack vel spectra to get fc ~ peak_f
    # Note that fc_S is predicted to be < fc_P
    stacked_spec, freqs = stack_spectra(sta_dict)
    peak_f = peak_freq(stacked_spec, freqs, fmin=25.)

    # fit,smom_dict = calc_fit(vel_dict, fc=peak_f, fmin=fmin, fmax=fmax,
    # plot_fit=plot_fit,
    # debug=False,
    # use_fixed_fmin_fmax=use_fixed_fmin_fmax)
    # exit()

    if debug_level > 0:
        logger.debug("%s: pha:%s velocity stack corner freq fc=%.1f" %
                     (fname, P_or_S, peak_f))
    if debug_level > 1:
        plot_spec(stacked_spec, freqs, title='Stack [%s] Vel spec '
                                             'peak_f=%.1f' % (P_or_S, peak_f))

    # Now recalculate the spectra as Displacment spectra:

    sta_dict = get_spectra(st, event, inventory, synthetic_picks,
                           calc_displacement=True,
                           S_win_len=.1, P_or_S=P_or_S)
    stacked_spec, freqs = stack_spectra(sta_dict)
    fit, fc_stack = calc_fit1(stacked_spec, freqs, fmin=1, fmax=fmax,
                              fit_displacement=True)

    # Calculate fmin/fmax from velocity signal/noise spec
    # and add into this displacement spec dict:
    for sta_code, sta_dd in sta_dict.items():
        for cha_code, cha_dict in sta_dd['chan_spec'].items():
            cha_dict['fmin'] = vel_dict[sta_code]['chan_spec'][cha_code][
                'fmin']
            cha_dict['fmax'] = vel_dict[sta_code]['chan_spec'][cha_code][
                'fmax']

    fit, smom_dict = calc_fit(sta_dict, fc=peak_f, fmin=fmin, fmax=fmax,
                              plot_fit=plot_fit,
                              debug=False,
                              use_fixed_fmin_fmax=use_fixed_fmin_fmax)

    phase = P_or_S
    arr_dict = {}
    arrivals = [arr for arr in event.preferred_origin().arrivals if
                arr.phase == phase]
    for arr in arrivals:
        pk = arr.get_pick()
        sta = pk.get_sta()
        if sta not in arr_dict:
            arr_dict[sta] = {}
        arr_dict[sta][phase] = arr

    for sta, sta_dict in smom_dict.items():
        smoms = []
        fits = []
        ts = []
        fmin = []
        fmax = []
        for cha, cha_dict in sta_dict.items():
            smoms.append(cha_dict['smom'])
            fits.append(cha_dict['fit'])
            ts.append(cha_dict['tstar'])
            fmin.append(cha_dict['fmin'])
            fmax.append(cha_dict['fmax'])

        smom = np.sqrt(np.sum(np.array(smoms) ** 2))
        fit = np.sum(np.array(fits)) / float(len(fits))
        fit = np.mean(fits)
        tstar = np.median(ts)
        min_f = np.max(fmin)
        max_f = np.min(fmax)

        if debug_level > 0:
            logger.debug("%s: sta:%3s pha:%s smom:%12.10g ts:%.4f nchans:%d" %
                         (fname, sta, phase, smom, tstar, len(fits)))

        arr = arr_dict[sta][phase]
        arr.smom = smom
        arr.fit = fit
        arr.tstar = tstar
        arr.fmin = min_f
        arr.fmax = max_f
        print("%s: sta:%3s pha:%s smom:%12.10g ts:%.4f nchans:%d fmin:%f "
              "fmax:%f" % (fname, sta, phase, smom, tstar, len(fits), min_f,
                           max_f))

    return smom_dict, peak_f


def plot_spec(spec, freqs, title='Stacked spec'):
    plt.loglog(freqs, spec, color='blue')
    plt.xlim(1e0, 3e3)

    if title:
        plt.title(title)
    plt.grid()
    plt.show()


def peak_freq(spec_array, freqs, fmin=0., fmax=None):
    """ Return peak_f of spec_array within bounds: fmin <= freqs <= fmax

        :param spec_array: Input spectrum
        :type  spec_array: np.array
        :param freqs: Corresponding frequencies
        :type  freqs: np.array
        :param fmin: Restrict peak search above this freq
        :type  fmin: float
        :param fmax: Restrict peak search below this freq
        :type  fmax: float

        :return: peak frequency within bounds: fmin <= peak_frequency <= fmax
        :rtype: float
    """

    if fmax is None:
        fmax = freqs[-1]

    mask = [all(f) for f in zip(freqs >= fmin, freqs <= fmax)]

    return freqs[mask][np.argmax(spec_array[mask], axis=0)]


def stack_spectra(sta_dict):
    """ stack the (normalized) pre-calculated spectra (fft) of each channel

        :param sta_dict: dictionary of pre-calculated P & S channel spectra (complex values) at each station
        :return: normalized_spectrum_stack (real modulus), freqs
        :rtype: (np.array, np.array)
    """

    stack = 0.
    n = 0.
    for sta_code, sta in sta_dict.items():
        for cha_code, cha in sta['chan_spec'].items():
            signal_fft, freqs, noise_fft = (
            cha['signal_fft'], cha['freqs'], cha['noise_fft'])

            # Normalize each fft individually and add to stack

            stack += np.abs(signal_fft) / np.amax(np.abs(signal_fft))

            n += 1

    return stack / np.amax(stack), freqs


def get_spectra(st, event, inventory, synthetic_picks,
                calc_displacement=False,
                S_win_len=.1, P_or_S='P'):
    """ Calculate the fft at each channel in the stream that has an arrival in event.arrivals

        :param st: uquake.core.stream.Stream
        :type  st: uquake.core.stream.Stream
        :param event: uquake.core.event.Event
        :type  event: uquake.core.event.Event
        :param stations: list of station metadata
        :type  stations: obspy.core.inventory.network.Network
        :param calc_displacement: If true calculate displacement spectra (else velocity)
        :type  calc_displacement: bool

        :return: sta_dict: dict of station P & S spectra for each channel
        :rtype: dict
    """

    pre_window_start_sec = .01
    max_S_P_time = 0.25

    # TODO Right now S win len = .1, may want to make this a function of distance and/or handle
    #     if win len exceeds trace len, etc

    fname = 'get_spectra'

    origin = event.preferred_origin() if event.preferred_origin() else \
    event.origins[0]

    # Only use picks that have an arrival used for this origin:
    picks = [arr.pick_id.get_referred_object() for arr in origin.arrivals]
    pick_dict = copy_picks_to_dict(picks)

    synthetic_dict = copy_picks_to_dict(synthetic_picks)

    if P_or_S == 'P':
        arrivals = [arr for arr in origin.arrivals if arr.phase == 'P']
    else:
        arrivals = [arr for arr in origin.arrivals if arr.phase == 'S']

    arr_dict = {}
    for arr in arrivals:
        pk = arr.pick_id.get_referred_object()
        sta = pk.waveform_id.station_code
        pha = arr.phase
        if sta not in arr_dict:
            arr_dict[sta] = {}
        arr_dict[sta][pha] = arr

    arr_stations = set()
    for arr in arrivals:
        arr_stations.add(
            arr.pick_id.get_referred_object().waveform_id.station_code)

    st_stations = set()
    for tr in st:
        st_stations.add(tr.stats.station)

    stations_without_trace = arr_stations - st_stations

    # Only keep stations that have at least one arrival and at least one trace
    sta_codes = arr_stations.intersection(st_stations)

    # Turn into a list sorted numerically by sta code (but keep sta code = str)
    sta_codes = sorted([sta for sta in sta_codes], key=lambda x: int(x))

    # For now, let's be explicit about our assumptions:
    # For each trace, we'll use P-.01 to S-.01 to calculate the P spectrum, so
    #   let's use the max expected S-P time (at the farthest station) to fix NFFT
    dt = st[0].stats.delta  # For now, let's assume ALL traces are sampled
    # at this rate
    npts = int(max_S_P_time / dt)
    nfft = npow2(npts) * 2  # Fix nfft for all calcs
    df = 1 / (float(nfft) * dt)
    # print("%s: Set nfft=%d --> df=%.3f" % (fname, nfft, df))

    # 1. Create a dict of keys=sta_code for all 'P' arrivals with the
    # necessary pick/sta info inside
    sta_dict = {}
    for sta_code in sta_codes:

        d = {}

        if 'P' in pick_dict[sta_code] and pick_dict[sta_code][
            'P'].evaluation_status != "rejected":
            d['ptime'] = pick_dict[sta_code]['P'].time
        else:
            d['ptime'] = synthetic_dict[sta_code]['P'].time

        if 'S' in pick_dict[sta_code] and pick_dict[sta_code][
            'S'].evaluation_status != "rejected":
            d['stime'] = pick_dict[sta_code]['S'].time
        else:
            d['stime'] = synthetic_dict[sta_code]['S'].time

        found = False

        sta_loc = inventory.select(sta_code).loc
        if sta_loc is None:
            print("Oops: sta:%s not found in inventory!" % sta_code)
            raise

        d['loc'] = sta_loc

        # Try to use distance along ray path if set in the arrival dict,
        #   else use Euclidean distance
        dist = arr_dict[sta_code][P_or_S].get('hypo_dist_in_m', None)
        if dist is None:
            dist = np.linalg.norm(d['loc'] - origin.loc)  # Dist in meters

        d['R'] = dist
        sta_dict[sta_code] = d

    # 2. Calc/save signal/noise fft spectra at all channels that have P(S) arrivals:
    for sta_code, sta in sta_dict.items():

        if P_or_S == 'P':
            signal_start = sta['ptime'] - pre_window_start_sec
            signal_end = sta['stime'] - pre_window_start_sec
        else:
            signal_start = sta['stime'] - pre_window_start_sec
            signal_end = sta['stime'] + S_win_len

        signal_len = signal_end - signal_start

        noise_end = sta['ptime'] - pre_window_start_sec
        noise_start = noise_end - signal_len
        if signal_start > signal_end:
            print("Signal start time larger the end time, continuing")
            print(
                "*** sta:%3s ptime:%s stime:%s P_or_S:%s pre_win:%f S_win:%f" % \
                (sta_code, sta['ptime'], sta['stime'], P_or_S,
                 pre_window_start_sec, S_win_len))
            continue

        trs = st.select(station=sta_code)
        if trs:
            chans = {}

            for tr in trs:
                ch = {}

                cha_code = tr.stats.channel

                tt_s = sta['stime'] - tr.stats.starttime
                tr.detrend('demean').detrend('linear').taper(type='cosine',
                                                             max_percentage=0.05,
                                                             side='both')

                signal = tr.copy()
                signal.trim(starttime=signal_start, endtime=signal_end)
                signal.detrend('demean').detrend('linear')
                signal.taper(type='cosine', max_percentage=0.05, side='both')

                noise = tr.copy()
                # if noise_start < trace.stats.starttime - then what ?
                noise.trim(starttime=noise_start, endtime=noise_end)
                noise.detrend('demean').detrend('linear')
                noise.taper(type='cosine', max_percentage=0.05, side='both')

                sensor_type = get_sensor_type_from_trace(tr)

                if sensor_type == 'ACC':
                    signal.integrate().taper(type='cosine',
                                             max_percentage=0.05, side='both')
                    noise.integrate().taper(type='cosine', max_percentage=0.05,
                                            side='both')
                elif sensor_type == 'VEL':
                    pass
                elif sensor_type == 'DISP':
                    print(
                        "%s: Not yet set up to handle input traces = DISPLACMENT !!" % fname)
                    continue
                else:
                    print(
                        "%s: ERROR: sta:%3s cha:%s sensor_type=[%s] is unknown" % \
                        (fname, sta_code, cha_code, sensor_type))
                    continue

                if calc_displacement:
                    signal.integrate().taper(type='cosine',
                                             max_percentage=0.05, side='both')
                    noise.integrate().taper(type='cosine', max_percentage=0.05,
                                            side='both')

                # check(signal)
                # parsevals(signal.data, dt, nfft)

                (signal_fft, freqs) = unpack_rfft(rfft(signal.data, n=nfft),
                                                  df)
                (noise_fft, freqs) = unpack_rfft(rfft(noise.data, n=nfft), df)

                # MTH: Determine the valid freq range: fmin - fmax
                #      To fit to model.
                # fmin/fmax = where smoothed signal spec exceeds smoothed noise spec by snr_threshold
                # fc1 = low-frequency corner of this trace's sensor

                if not calc_displacement:

                    fc1 = get_corner_freq_from_pole(
                        tr.stats.response.get_paz().poles[0])
                    fmin, fmax = find_fmin_fmax_from_spec(signal_fft,
                                                          noise_fft, 1.3, 200.,
                                                          df)

                    # Take the maximum of fc1 and fmin to use for fitting:
                    # fmin is None if signal > noise all the way to DC
                    if fmin is None or fc1 > fmin:
                        fmin = fc1

                    if fmax is None:
                        fmax = freqs[-1]

                    ch['fmin'] = fmin
                    ch['fmax'] = fmax

                # plot_spec(freqs, signal_fft, noise_fft, title=None)

                # np/scipy ffts are not scaled by dt
                signal_fft *= dt
                noise_fft *= dt
                '''
                This is a 1-sided FFT (ie, we only have the +ve frequency half)
                To conserve power a la Parseval's, we need to double
                the power at each frequency *except* for DC & Nyquist
                which are purely real and don't have a -ve freq.

                Equivalently, we scale the FFT amps at these frequencies
                by sqrt(2):
                '''
                signal_fft[1:-1] *= np.sqrt(2.)
                noise_fft[1:-1] *= np.sqrt(2.)

                # plot_signal(signal, noise)

                ch['nfft'] = nfft
                ch['dt'] = dt
                ch['df'] = df
                ch['freqs'] = freqs
                ch['signal_fft'] = signal_fft
                ch['noise_fft'] = noise_fft
                ch['P_or_S'] = P_or_S
                if calc_displacement:
                    ch['disp_or_vel'] = 'DISPLACEMENT'
                else:
                    ch['disp_or_vel'] = 'VELOCITY'
                chans[tr.stats.channel] = ch
            sta['chan_spec'] = chans

            # end for tr in trs
        # end if trs
    # end for sta in sta_dict

    return sta_dict


def brune_omega3(fc: float, mom: float, ts: float, f: float) -> float:
    return mom * 1. / (1 + np.power(f / fc, 3.)) * np.exp(-np.pi * ts * f)


def brune_dis_spec(fc: float, mom: float, ts: float, f: float) -> float:
    return mom * (fc * fc) / (fc * fc + f * f) * np.exp(-np.pi * ts * f)


def brune_vel_spec(fc: float, mom: float, ts: float, f: float) -> float:
    return 2. * np.pi * f * brune_dis_spec(fc, mom, ts, f)


def brune_acc_spec(fc: float, mom: float, ts: float, f: float) -> float:
    return 2. * np.pi * f * brune_vel_spec(fc, mom, ts, f)


def getresidfit(data_spec, model_func, fc: float, freqs: list, Lnorm='L1',
                weight_fr=False, fmin=1., fmax=3000.) -> float:
    def inner(p):
        smom = p[0]
        ts = p[1]
        fit = 0.
        # Neither Powell nor Nelder-Meade seem to have a way to restrict search params to be > 0,
        # and L-BFGS-B does not appear to work, so here's a hack:
        if smom < 0. or ts < 0.:
            return 1e12

        for i, f in enumerate(freqs):
            if f < fmin or f > fmax:
                continue
            if model_func(fc, smom, ts, f) < 0.:
                print(
                    "** OOPS: f=%f vel_spec=%g smom=%g ts=%g model_func=%g" % (
                    f, vel_spec[i], smom, ts, model_func(fc, smom, ts, f)))
                pass
            diff = np.log10(data_spec[i]) - np.log10(
                model_func(fc, smom, ts, f))
            if Lnorm == 'L2':
                diff *= diff
            if weight_fr:
                diff /= f
            fit += diff
        return fit

    return inner


def getresidfit2(data_spec, model_func, freqs: list, Lnorm='L1',
                 weight_fr=False,
                 fmin=1., fmax=3000.) -> float:
    def inner(p):
        smom = p[0]
        fc = p[1]
        fit = 0.
        # Neither Powell nor Nelder-Meade seem to have a way to restrict search params to be > 0,
        # and L-BFGS-B does not appear to work, so here's a hack:
        if smom < 0.:
            return 1e12

        for i, f in enumerate(freqs):
            if f < fmin or f > fmax:
                continue
            diff = np.log10(data_spec[i]) - np.log10(model_func(fc, smom, f))
            if Lnorm == 'L2':
                diff *= diff
            if weight_fr:
                diff /= f
            fit += diff
        return fit

    return inner


def getresidfit3(data_spec, model_func, freqs: list, Lnorm='L1',
                 weight_fr=False,
                 fmin=1., fmax=3000.) -> float:
    def inner(p):
        smom = p[0]
        ts = p[1]
        fc = p[2]
        fit = 0.
        # Neither Powell nor Nelder-Meade seem to have a way to restrict search params to be > 0,
        # and L-BFGS-B does not appear to work, so here's a hack:
        if smom < 0. or ts < 0.:
            return 1e12

        for i, f in enumerate(freqs):
            if f < fmin or f > fmax:
                continue
            if model_func(fc, smom, ts, f) < 0.:
                print(
                    "** OOPS: f=%f vel_spec=%g smom=%g ts=%g model_func=%g" % (
                    f, vel_spec[i], smom, ts, model_func(fc, smom, ts, f)))
                pass
            diff = np.log10(data_spec[i]) - np.log10(
                model_func(fc, smom, ts, f))
            if Lnorm == 'L2':
                diff *= diff
            if weight_fr:
                diff /= f
            fit += diff
        return fit

    return inner


def calc_fit1(spec, freqs, fmin=1., fmax=1000., Lnorm='L2', weight_fr=False,
              fit_displacement=False):
    # spec = spec_amp

    model_func = brune_vel_spec
    if fit_displacement:
        model_func = brune_dis_spec

    residfit = getresidfit3(spec, model_func, freqs, Lnorm=Lnorm, \
                            weight_fr=weight_fr, fmin=fmin, fmax=fmax)

    # Give it a starting vector:
    #    smom  t* fc
    pp = [1., .001, 100]
    (sol, fit, *rest) = optimize.fmin(residfit, np.array(pp), xtol=10 ** -12,
                                      ftol=10 ** -6, disp=False, full_output=1)
    mom, ts, fc = sol[0], sol[1], sol[2]
    # print(fc)
    if sol[0] < 0.:
        print("Ummm smom < 0 !!!")

    # print("solution=",sol)
    plot_fit = 0
    if plot_fit:
        model_spec = np.array(np.zeros(freqs.size), dtype=np.float_)
        for i, f in enumerate(freqs):
            model_spec[i] = model_func(fc, mom, ts, f)
        plot_spec2(freqs, spec, model_spec,
                   title="Fit to spec stack fc=%.2f" % fc)

    return fit, fc


def calc_fit2(spec, freqs, fmin=10., fmax=1000., Lnorm='L2', weight_fr=False):
    model_func = brune_vel_spec2

    residfit = getresidfit2(spec, model_func, freqs, Lnorm=Lnorm, \
                            weight_fr=weight_fr, fmin=fmin, fmax=fmax)

    # Give it a starting vector:
    pp = [1., 100]
    (sol, fit, *rest) = optimize.fmin(residfit, np.array(pp), xtol=10 ** -12,
                                      ftol=10 ** -6, disp=False, full_output=1)
    if sol[0] < 0.:
        print("Ummm smom < 0 !!!")

    print("solution=", sol)
    plot_fit = 1
    if plot_fit:
        model_spec = np.array(np.zeros(freqs.size), dtype=np.float_)
        for i, f in enumerate(freqs):
            model_spec[i] = model_func(100., 1., f)
            # model_spec[i] = model_func(80., sol[0], f)
            # model_spec[i] = model_func(sol[1], sol[0], f)
        plot_spec2(freqs, spec, model_spec,
                   title="Fit to spec stack fc=%f" % sol[1])
    return fit


def calc_fit(sta_dict, fc, fmin=20., fmax=1000.,
             Lnorm='L2', weight_fr=False,
             plot_fit=False, debug=False,
             use_fixed_fmin_fmax=False):
    fit = 0.
    smom_dict = {}
    for sta_code, sta in sta_dict.items():
        # print("  calc_fit sta:%3s" % sta_code)
        if 'chan_spec' not in sta:
            if debug:
                print("  sta:%3s: Has no chan_spec --> Skip" % sta_code)
            continue
        dd = {}
        for cha_code, cha in sta['chan_spec'].items():
            signal_fft, freqs, noise_fft = (
            cha['signal_fft'], cha['freqs'], cha['noise_fft'])

            model_func = brune_vel_spec
            if cha['disp_or_vel'] == 'DISPLACEMENT':
                model_func = brune_dis_spec

            ch_dict = {}
            # Optimize fit of model_func to signal_fft to solve for smom:

            if not use_fixed_fmin_fmax:
                fmin = cha['fmin']
                fmax = cha['fmax']

            if debug:
                print("    calc_fit sta:%3s cha:%s fmin:%.1f fmax:%.1f" % \
                      (sta_code, cha_code, fmin, fmax))

            residfit = getresidfit(np.abs(signal_fft), model_func, fc, freqs,
                                   Lnorm=Lnorm, \
                                   weight_fr=weight_fr, fmin=fmin, fmax=fmax)

            # Give it a starting vector:
            #     smom    tstar
            # MTH: this should be wert origin time anyway ...
            # tt_s = sta['stime'] - tr.stats.starttime
            tt_s = .02
            # pp = [1e-5, tt_s/500.]
            # pp = [1e-5, tt_s/200.]
            pp = [1e-10, tt_s / 200.]
            # (sol,fopt,*rest)= optimize.fmin(residfit, np.array(pp),xtol=10**-12,ftol=10**-12,disp=False, full_output=1)
            (sol, fopt, *rest) = optimize.fmin(residfit, np.array(pp),
                                               ftol=.01, disp=False,
                                               full_output=1)
            if sol[0] < 0.:
                print("Ummm smom < 0 !!!")

            # gives same result as fopt
            # ch_fit = residfit(sol)
            fit += fopt

            ch_dict['smom'] = sol[0]
            ch_dict['tstar'] = sol[1]
            ch_dict['fit'] = fopt
            ch_dict['P_or_S'] = cha['P_or_S']
            ch_dict['fmin'] = cha['fmin']
            ch_dict['fmax'] = cha['fmax']

            if debug:
                print("calc_fit: sta:%s cha:%s smom:%12.10g fit:%.2f ts=%f" % (
                sta_code, cha_code, sol[0], fopt, sol[1]))
                # print("calc_fit: sta:%s cha:%s smom:%12.10g fit:%.2f" % (sta_code, cha_code, sol[0], fopt))

            # if plot_fit and cha['P_or_S'] == "S":
            plot_fit = 0
            if plot_fit:
                model_spec = np.array(np.zeros(freqs.size), dtype=np.float_)
                w2_no_atten = np.array(np.zeros(freqs.size), dtype=np.float_)
                w3_no_atten = np.array(np.zeros(freqs.size), dtype=np.float_)
                brune_vel = np.array(np.zeros(freqs.size), dtype=np.float_)
                corrected_vel_spec = np.array(np.zeros(freqs.size),
                                              dtype=np.float_)
                Q = 100
                tstar = sta['R'] / (5.3e3 * Q)
                print("tstar=", tstar)
                for i, f in enumerate(freqs):
                    # print("i=%d f=%f" % (i, f))
                    model_spec[i] = model_func(fc, sol[0], sol[1], f)
                    w2_no_atten[i] = model_func(fc, sol[0], 0, f)
                    w3_no_atten[i] = brune_omega3(fc, sol[0], 0, f)
                    # brune_vel[i] = brune_vel_spec(fc, sol[0], sol[1], f)
                    brune_vel[i] = brune_vel_spec(fc, sol[0], 0, f)
                    corrected_vel_spec[i] = signal_fft[i] * np.exp(
                        np.pi * f * tstar)

                # plot_spec5(freqs, vel_spec,  np.abs(noise_fft), model_spec,
                # plot_spec5(freqs, np.abs(signal_fft),  np.abs(noise_fft), model_spec,
                plot_spec5(freqs, np.abs(corrected_vel_spec),
                           np.abs(noise_fft), model_spec,
                           w2_no_atten, brune_vel,
                           title="sta:%s ch:%s spec fit phase:%s [lambda=%12.10g]" % \
                                 (sta_code, cha_code, ch_dict['P_or_S'],
                                  sol[0]),
                           subtitle="fmin:%.1f fmax:%.1f use_fixed_fmin_fmax:%s" % \
                                    (fmin, fmax, use_fixed_fmin_fmax))

            dd[cha_code] = ch_dict
        smom_dict[sta_code] = dd

    return fit, smom_dict


def plot_spec5(freqs, signal_fft, noise_fft, model_spec,
               model2_spec, model3_spec,
               title=None, subtitle=None):
    # plt.loglog(freqs, np.abs(signal_fft), color='blue')
    plt.loglog(freqs, np.abs(signal_fft), color='blue', linestyle="--")
    # plt.loglog(freqs, np.abs(noise_fft),  color='red')
    plt.loglog(freqs, model2_spec, color='black')
    plt.loglog(freqs, model_spec, color='green')
    # plt.loglog(freqs, model_spec,  color='green', linewidth=.5)
    plt.loglog(freqs, model3_spec, color='red')

    plt.legend(['signal', 'w^2', 'w^2 + ts', 'Brune vel'])
    # plt.legend(['signal', 'w^2', 'w^2 + ts', 'w^3'])
    # plt.legend(['signal', 'noise', 'model'])

    plt.xlim(1e-3, 3e3)
    # plt.ylim(1e-12, 1e-6)
    # plt.ylim(1e-15, 1e-9)
    # suptitle is larger and goes on top (?!)
    if title:
        plt.suptitle(title)
    if subtitle:
        plt.title(subtitle)
    plt.grid()
    plt.show()
    return


def plot_spec3(freqs, signal_fft, noise_fft, model_spec, title=None,
               subtitle=None):
    plt.loglog(freqs, np.abs(signal_fft), color='blue')
    plt.loglog(freqs, np.abs(noise_fft), color='red')
    plt.loglog(freqs, model_spec, color='green')
    plt.legend(['signal', 'noise', 'model'])
    plt.xlim(1e0, 3e3)
    # plt.ylim(1e-12, 1e-6)
    plt.ylim(1e-15, 1e-9)
    # suptitle is larger and goes on top (?!)
    if title:
        plt.suptitle(title)
    if subtitle:
        plt.title(subtitle)
    plt.grid()
    plt.show()

    return


from scipy.signal import savgol_filter


def find_fmin_fmax_from_spec(signal_fft, noise_fft, snr_thresh, fc, df):
    npoly = 2
    nsmooth = 31
    signal = savgol_filter(np.abs(signal_fft), nsmooth, npoly)
    noise = savgol_filter(np.abs(noise_fft), nsmooth, npoly)
    snr = signal / noise

    fmin = None
    fmax = None

    ic = int(fc / df)
    i1 = -999
    i2 = -999
    for i in range(ic, 0, -1):
        if snr[i] <= snr_thresh:
            i1 = i
            break
    if i1 > 0:
        fmin = float(i1) * df

    for i in range(ic, snr.size):
        if snr[i] <= snr_thresh:
            i2 = i
            break

    if i2 > 0:
        fmax = float(i2) * df

    return fmin, fmax


from scipy.signal import savgol_filter


def plot_spec(freqs, signal_fft, noise_fft=None, title=None):
    npoly = 2
    nsmooth = 31
    signal = savgol_filter(np.abs(signal_fft), nsmooth, npoly)
    noise = savgol_filter(np.abs(noise_fft), nsmooth, npoly)
    # signal = np.abs(signal_fft)
    # noise = np.abs(noise_fft)

    # plt.loglog(freqs, np.abs(signal_fft), color='blue')
    plt.loglog(freqs, signal, color='blue')

    if noise_fft is not None:
        plt.loglog(freqs, noise, color='red')
        # plt.loglog(freqs, np.abs(noise_fft),  color='red')
        plt.legend(['signal', 'noise'])
    # plt.loglog(freqs, model_spec,  color='green')
    # plt.legend(['signal', 'noise', 'model'])
    plt.xlim(1e0, 3e3)
    # plt.ylim(1e-8, 1.2e-4)
    # plt.ylim(1e-12, 1e-4)
    if title:
        plt.title(title)
    plt.grid()
    plt.show()


def plot_spec2(freqs, spec, model_spec, title=None):
    plt.loglog(freqs, spec, color='blue')
    plt.loglog(freqs, model_spec, color='green')
    plt.legend(['signal', 'model'])
    plt.xlim(1e0, 3e3)
    # plt.ylim(1e-12, 1e-4)
    if title:
        plt.title(title)
    plt.grid()
    plt.show()


def plot_signal(signal, noise=None):
    # def plot_signal(signal, noise):
    '''
        signal = obspy Trace windowed around signal
        noise  = obspy Trace windowed around pre-signal noise
    '''
    t = []
    dt = signal.stats.sampling_rate
    for i in range(signal.stats.npts):
        t.append(float(i) * dt)
    plt.plot(t, signal, color='blue')
    if noise:
        plt.plot(t, noise, color='red')
        plt.legend(['signal', 'noise'])
    plt.title("%s: P window" % signal.get_id())
    plt.grid()
    plt.show()


def unpack_rfft(rfft, df):
    n = rfft.size
    if n % 2 == 0:
        n2 = int(n / 2)
    else:
        print("n is odd!!")
        exit()
    # print("n2=%d" % n2)

    c_arr = np.array(np.zeros(n2 + 1, ), dtype=np.complex_)
    freqs = np.array(np.zeros(n2 + 1, ), dtype=np.float_)

    c_arr[0] = rfft[0]
    c_arr[n2] = rfft[n - 1]
    freqs[0] = 0.
    freqs[n2] = float(n2) * df

    for i in range(1, n2):
        freqs[i] = float(i) * df
        c_arr[i] = np.complex(rfft[2 * i - 1], rfft[2 * i])

    return c_arr, freqs


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")
        # raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")
        # raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError,
               "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        # raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y


def npow2(n: int) -> int:
    """ return power of 2 >= n
    """
    if n <= 0:
        return 0

    nfft = 2
    while (nfft < n):
        nfft = (nfft << 1)
    return nfft


if __name__ == '__main__':
    main()
