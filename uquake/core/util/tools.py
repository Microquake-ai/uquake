from datetime import datetime, timedelta

import numpy as np
from scipy.fftpack import fft, fftfreq, ifft
from scipy.signal import iirfilter, sosfilt, zpk2sos


def datetime_to_epoch_sec(dtime):
    return (dtime - datetime(1970, 1, 1)) / timedelta(seconds=1)


def make_picks(stcomp, pick_times_utc, phase, pick_params):
    snr_wlens = np.array(pick_params.snr_wlens)
    wlen_search = pick_params.wlen_search
    stepsize = pick_params.stepsize
    edge_time = wlen_search / 2 + np.max(snr_wlens)

    picks = []

    for tr, ptime in zip(stcomp, pick_times_utc):

        if tr.time_within(ptime, edge_time) is True:
            picks.append(tr.make_pick(ptime, wlen_search,
                                      stepsize, snr_wlens, phase_hint=phase))

    return picks


def copy_picks_to_dict(picks):
    pick_dict = {}

    for pick in picks:
        station = pick.waveform_id.station_code
        phase = pick.phase_hint

        if station not in pick_dict:
            pick_dict[station] = {}
        # MTH: If you copy the pick you pollute the reference id space
        #      and arrival.pick_id.get_referred_object() no longer works!
        # pick_dict[station][phase]=copy.deepcopy(pick)
        pick_dict[station][phase] = pick

    return pick_dict


def picks_to_dict(picks):
    pd = {}

    for p in picks:
        key = p.waveform_id.get_seed_string()

        if key not in pd:
            pd[key] = []
        pd[key].append(p.time)

    return pd


def repick_using_snr(sig, ipick, wlen_search, stepsize, snr_wlens):
    origin_inds, snrs = sliding_snr(sig, ipick, wlen_search, stepsize,
                                    snr_wlens)
    newpick = origin_inds[np.argmax(snrs)]
    snr = np.max(snrs)

    return newpick, snr


def sliding_snr(sig, ipick, wlen_search, stepsize, snr_wlens, plot=False):
    wl_noise, wl_sig = snr_wlens.astype(int)
    hl = int(wlen_search // 2)
    i0 = max(wl_noise, ipick - hl)
    i1 = min(len(sig) - wl_sig, ipick + hl)

    origin_inds = np.arange(i0, i1, stepsize)
    snrs = np.zeros(len(origin_inds), dtype=np.float32)

    for i, og in enumerate(origin_inds):
        energy_noise = np.mean((sig[og - wl_noise:og]) ** 2)
        energy_sig = np.mean((sig[og:og + wl_sig]) ** 2)
        snrs[i] = energy_sig / energy_noise
    snrs = 10 * np.log10(snrs)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(sig / np.max(sig))
        plt.plot(origin_inds, snrs)
        plt.axvline(ipick, color='green', label='pick_old')
        plt.axvline(i0, color='black', label='scan_win')
        plt.axvline(i1, color='black')

        ot = origin_inds[np.argmax(snrs)]
        plt.axvline(ot, color='green', linestyle='--', label='pick_new')
        plt.axvline(ot - wl_noise, color='red', label='snr_win')
        plt.axvline(ot + wl_sig, color='red')
        plt.legend()

    return origin_inds, snrs


def create_composite(sigs, groups):
    nsig = len(groups)
    npts = sigs.shape[1]

    out = np.zeros((nsig, npts), dtype=sigs.dtype)

    for i, group in enumerate(groups):
        out[i] = np.sign(sigs[group[0]]) * np.sqrt(
            np.sum(sigs[group] ** 2, axis=0))

    return out


def stream_to_array(st, t0, npts_fix, taplen=0.05):
    sr = st[0].stats.sampling_rate
    nsig = len(st)
    taplen_npts = int(npts_fix * taplen)

    data = np.zeros((nsig, npts_fix), dtype=np.float32)

    for i, tr in enumerate(st):
        i0 = int((tr.stats.starttime - t0) * sr + 0.5)
        sig = tr.data - np.mean(tr.data)

        if taplen != 0:
            taper_data(sig, taplen_npts)
        slen = min(len(sig), npts_fix - i0)
        data[i, i0: i0 + slen] = sig[:slen]

    return data


def taper_data(data, wlen):
    tap = hann_half(wlen)
    data[:wlen] *= tap
    data[-wlen:] *= tap[::-1]


def taper2d(data, wlen):
    out = data.copy()
    tap = hann_half(wlen)

    for i in range(data.shape[0]):
        out[i][:wlen] *= tap
        out[i][-wlen:] *= tap[::-1]

    return out


def ttsamp(dist, vel, sr):
    return int(dist / vel * sr + 0.5)


def integrate(sig):
    from scipy.integrate import cumtrapz

    return cumtrapz(sig, initial=0)


def attenuate(sig, sr, dist, Q, vel, gspread=True):
    npts = len(sig)
    fsig = fft(sig)
    freqs = fftfreq(npts, d=1. / sr)
    tstar = dist / (vel * Q)
    factor = np.exp(-np.pi * np.abs(freqs) * tstar)
    fsig *= factor
    sig = np.real(ifft(fsig))

    if gspread:
        sig /= dist

    return sig


def roll_data(data, tts):
    droll = np.zeros_like(data)

    for i, sig in enumerate(data):
        droll[i] = np.roll(sig, -tts[i])

    return droll


def velstack(data, dists2src, sr, vels):
    dnorm = norm2d(data)
    dstack = np.zeros((len(vels), dnorm.shape[1]), dtype=np.float32)

    for ivel, vel in enumerate(vels):
        shifts = (dists2src / vel * sr + 0.5).astype(int)

        for i, shift in enumerate(shifts):
            dstack[ivel] += np.roll(dnorm[i], -shift)

    return dstack


def chan_groups(chanmap):
    return [np.where(sk == chanmap)[0] for sk in np.unique(chanmap)]


def comb_channels(data, cmap):
    groups = [np.where(sk == cmap)[0] for sk in np.unique(cmap)]
    dstack = np.zeros((len(groups), data.shape[1]))

    for i, grp in enumerate(groups):
        dstack[i] = np.mean(np.abs(data[grp]), axis=0)

    return dstack


def bandpass(data, band, sr, corners=4, zerophase=True):
    freqmin, freqmax = band
    fe = 0.5 * sr
    low = freqmin / fe
    high = freqmax / fe

    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)

    if zerophase:
        firstpass = sosfilt(sos, data)

        if len(data.shape) == 1:
            return sosfilt(sos, firstpass[::-1])[::-1]
        else:
            return np.fliplr(sosfilt(sos, np.fliplr(firstpass)))
    else:
        return sosfilt(sos, data)


def filter(data, btype, band, sr, corners=4, zerophase=True):
    # btype: lowpass, highpass, band

    fe = 0.5 * sr
    z, p, k = iirfilter(corners, band / fe, btype=btype,
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)

    if zerophase:
        firstpass = sosfilt(sos, data)

        if len(data.shape) == 1:
            return sosfilt(sos, firstpass[::-1])[::-1]
        else:
            return np.fliplr(sosfilt(sos, np.fliplr(firstpass)))
    else:
        return sosfilt(sos, data)


def decimate(data_in, sr, factor):
    data = data_in.copy()
    fmax = sr / (factor * 2)
    filter(data, 'lowpass', fmax, sr)

    if len(data.shape) == 1:
        return data[::factor]
    else:
        return data[:, ::factor]


def norm2d(d):
    return d / np.max(np.abs(d), axis=1)[:, np.newaxis]


def cross_corr(sig1, sig2, norm=True, pad=False, phase_only=False, phat=False):
    """Cross-correlate two signals."""
    pad_len = len(sig1)

    if pad is True:
        pad_len *= 2
        # pad_len = signal.next_pow_2(pad_len)

    sig1f = fft(sig1, pad_len)
    sig2f = fft(sig2, pad_len)

    if phase_only is True:
        ccf = np.exp(- 1j * np.angle(sig1f)) * np.exp(1j * np.angle(sig2f))
    else:
        ccf = np.conj(sig1f) * sig2f

    if phat:
        ccf = ccf / np.abs(ccf)

    cc = np.real(ifft(ccf))

    if norm:
        cc /= np.sqrt(energy(sig1) * energy(sig2))

    return np.roll(cc, len(cc) // 2)


def energy(sig, axis=None):
    return np.sum(sig ** 2, axis=axis)


def energy_freq(fsig, axis=None):
    return np.sum(np.abs(fsig) ** 2, axis=axis) / fsig.shape[-1]


def freq_window(cf, npts, sr):
    nfreq = int(npts // 2 + 1)
    fsr = npts / sr
    cf = np.array(cf, dtype=float)
    cx = (cf * fsr + 0.5).astype(int)

    win = np.zeros(nfreq, dtype=np.float32)
    win[:cx[0]] = 0
    win[cx[0]:cx[1]] = taper_cosine(cx[1] - cx[0])
    win[cx[1]:cx[2]] = 1
    win[cx[2]:cx[3]] = taper_cosine(cx[3] - cx[2])[::-1]
    win[cx[-1]:] = 0

    return win


def taper_cosine(wlen):
    return np.cos(np.linspace(np.pi / 2., np.pi, wlen)) ** 2


def phase(sig):
    return np.exp(1j * np.angle(sig))


def whiten2D(a, freqs, sr):
    wl = a.shape[1]
    win = freq_window(freqs, wl, sr)
    af = fft(a)

    for sx in range(a.shape[0]):
        whiten_freq(af[sx], win)
    a[:] = np.real(ifft(af))


def whiten(sig, win):
    """Whiten signal, modified from MSNoise."""
    npts = len(sig)
    nfreq = int(npts // 2 + 1)

    assert (len(win) == nfreq)
    # fsr = npts / sr

    fsig = fft(sig)
    # hl = nfreq // 2

    half = fsig[: nfreq]
    half = win * phase(half)
    fsig[: nfreq] = half
    fsig[-nfreq + 1:] = half[1:].conjugate()[::-1]

    return np.real(ifft(fsig))


def whiten_freq(fsig, win):
    # npts = len(fsig)
    # nfreq = int(npts // 2 + 1)
    nfreq = int(len(fsig) // 2 + 1)
    assert (len(win) == nfreq)
    fsig[: nfreq] = win * phase(fsig[: nfreq])
    fsig[-nfreq + 1:] = fsig[1: nfreq].conjugate()[::-1]


def mirror_freqs(data):
    nfreq = int(data.shape[1] // 2 + 1)
    data[:, -nfreq + 1:] = np.fliplr(data[:, 1: nfreq].conjugate())


def amax_cc(sig):
    return np.argmax(sig) - len(sig) // 2


def angle(a, b):
    # return np.arctan((a[1] - b[1]) / (a[0] - b[0]))

    return np.arctan2((a[1] - b[1]), (a[0] - b[0]))


def add_noise(a, freqs, sr, scale, taplen=0.05):
    out = a.copy()

    nsig, npts = a.shape
    fwin = freq_window(freqs, npts, sr)
    nfreq = len(fwin)

    for i in range(nsig):
        fb = np.zeros(npts, dtype=np.complex64)

        phases = np.random.rand(nfreq) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)

        fb[: nfreq] = phases * fwin
        fb[-nfreq + 1:] = fb[1:nfreq].conjugate()[::-1]
        # a[i] = np.real(ifft(fb))
        out[i] += np.real(ifft(fb)) * scale
    taper2d(out, int(taplen * npts))

    return out


def zeropad2d(a, npad):
    nrow, ncol = a.shape
    out = np.zeros((nrow, ncol + npad), dtype=a.dtype)
    out[:, :ncol] = a

    return out


def zero_argmax(a, wlen, taplen=0.05):
    # handles edges incorrectly
    npts = a.shape[1]
    taper = hann_half(int(taplen * wlen))
    win = np.concatenate((taper[::-1], np.zeros(wlen), taper))
    hl = int(len(win) // 2)
    out = a.copy()
    imaxes = np.argmax(np.abs(out), axis=1)

    for i, imax in enumerate(imaxes):
        i0 = imax - hl
        i1 = imax + hl

        if i0 <= 0:
            out[i, 0:i1] *= win[abs(i0):]
        elif i1 > npts:
            out[i, i0:] *= win[:-(i1 - npts)]
        else:
            out[i, i0:i1] *= win

    return out


def hann(npts):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(npts) / (npts - 1))


def hann_half(npts):
    return hann(npts * 2)[:npts]


def fftnoise(f):
    f = np.array(f, dtype='complex')
    npts = (len(f) - 1) // 2
    phases = np.random.rand(npts) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:npts + 1] *= phases
    f[-1:-1 - npts:-1] = np.conj(f[1:npts + 1])

    return ifft(f).real


def band_noise(band, sr, samples):
    freqmin, freqmax = band
    freqs = np.abs(fftfreq(samples, 1. / sr))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs >= freqmin, freqs <= freqmax))[0]
    f[idx] = 1

    return fftnoise(f)


def envelope(data):
    slen = len(data)
    FFT = fft(data, slen)
    FFT[1: slen // 2] *= 2
    FFT[slen // 2:] = 0

    return np.abs(ifft(FFT))


def read_csv(filename, site_code='', **kwargs):
    """
    read a csv file containing site information
    The first line of the csv file should contain the site name
    The expected file structure is as follows and should contain one header line
    <network>, <site name>, <site type>, <no component>, x, y, z
    where x, y and z represents the location of the sites expressed in a local
    coordinate system. Note that the <site name> is limited to four character
    because of NonLinLoc limitations.

    example of file strucuture

    1. <Network>, <site long name>, <site code>, <site type>, <gain>,
    <sensitivity>, <sx>, <sy>, <sz>, <channel 1 code>, <azimuth>, <dip>,
    <channel 2 code>, <azimuth>, <dip>, <channel 3 code>, <azimuth>, <dip>

    :param filename: path to a csv file
    :type filename: string
    :param site_code: site code
    :type site_code: string
    :param has_header: whether or not the input file has an header
    :type has_header: bool
    :rparam: site object
    :rtype: ~uquake.core.station.Site
    """

    from uquake.core.data.station import Site, Network, Station, Channel
    from numpy import loadtxt

    data = loadtxt(filename, delimiter=',', skiprows=1, dtype=object)
    stations = []

    for i, tmp in enumerate(data):

        nc, long_name, sc, st, smt, gain, sensitivity = tmp[:7]
        staloc = tmp[7:10].astype(float)
        orients = tmp[10:22].reshape(3, 4)

        channels = []

        for comp in orients:
            if not comp[0]:
                continue
            xyz = comp[1:4].astype(float)
            channel = Channel(code=comp[0])
            channel.orientation = xyz
            channels.append(channel)

        station = Station(long_name=long_name, code=sc, site_type=st,
                          motion_type=smt, gain=float(gain),
                          sensitivity=float(sensitivity), loc=staloc,
                          channels=channels)

        stations.append(station)

    networks = [Network(code=nc, stations=stations)]
    site = Site(code=site_code, networks=networks)

    return site
