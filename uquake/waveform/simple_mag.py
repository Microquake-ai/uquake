import numpy as np
from numpy.fft import fft
from scipy.optimize import curve_fit
from scipy.ndimage.interpolation import map_coordinates
from uquake.core.logging import logger
from obspy.core.trace import Stats
from uquake.core.stream import Trace, Stream
from uquake.grid.base import Grid
from uquake.core import event


########################################################################################
# Function defining attenuation relationship
########################################################################################


def geometrical_spreading(raypath):
    segment_len = np.sqrt(np.sum(np.diff(raypath, axis=0) ** 2, axis=1))

    # Source receiver distance
    r = np.sum(segment_len)

    if r < 1:
        r = 1

    # Geometrical spreading attenuation
    Ar = 1 / r

    return Ar


def anelastic_scattering_attenuation(raypath_or_distance, velocity, quality,
                                     seismogram, return_seimogram=True):
    """
    Calculate the anelastic_scattering_attenuation
    :param raypath_or_distance: raypath or distance
    :type raypath_or_distance: a list of point along the raypath or a float
    :param velocity: velocity along the raypath
    :type: uquake.core.data.grid.Grid or float
    :param quality: Seismic quality factor
    :type quality: uquake.core.data.grid.Grid or float
    :param seismogram: displacement waveform
    :type seismogram: uquake.core.Trace
    :return: attenuated seismogram
    :rtype: uquake.core.Trace
    """

    freq_coeff, _ = interpolate_Fc_Mw()
    freq_corner = FcMw(Mw, freq_coeff[0], freq_coeff[1])

    f = np.arange(0,
                  1000) / 1000.0 * freq_corner  # calculating the
    # attenuation up to the corner frequency
    Aq = np.exp(- np.pi * f)

    ray = quality.transform_to(raypath)

    Qray = map_coordinates(quality.data, ray.T, mode='nearest')
    Vray = map_coordinates(velocity.data, ray.T, mode='nearest')
    slen = np.sqrt(np.sum(np.diff(raypath, axis=0) ** 2, axis=1))

    Att = np.zeros(f.shape)
    for k, segment in enumerate(ray[:-1, :]):
        Qval = np.mean(Qray[k:k + 2])
        Vval = np.mean(Vray[k:k + 2])
        Att += slen[k] / (Vval * Qval)

    Aq_f = np.exp(-np.pi * f * Att)
    Aq = np.mean(Aq_f)

    return Aq


def radiation_pattern_attenuation():
    """
    :rparam: radiation pattern attenuation for S- and S-waves
    :rtype: tuple of float

    .. NOTE::

    Returned values are taken from Aki and Richard (2002) which are
    0.52 and 0.63 for P- and S-waves, respectively.

    .. EXAMPLE::

    >>> (PRadiation, SRadiation) = radiation_pattern_attenuation()
    """
    return 0.52, 0.63


def geometrical_spreading_attenuation(raypath, velocity=None, quality=None,
                                      Mw=-1):
    """
    :param raypath: Raypath between source and receiver
    :type raypath: nympy array
    :param velocity: velocity grid (default None)
    :type velocity: uquake ImageData
    :param quality: quality factor (default None)
    :type quality: uquake ImageData
    :return: the a tuple containing the geometrical spreading attenuation
    and the attenuation related to anelastic absorbtion and scattering
    """

    ar = geometrical_spreading(raypath)
    if quality:
        aq = anelastic_scattering_attenuation(raypath, velocity, quality, Mw)
        att = ar * aq
    else:
        att = ar

    return att


def calculate_attenuation(raypath, velocity, quality=None, Mw=-1):
    """
    :param source: source location tuple dimension of the grid
    :param receiver: receiver location tuple dimension of the grid
    :param velocity: velocity in EKImageData format
    :param quality: quality factor grid in EKImageData format
    :return the a tuple containing the geometrical spreading attenuation and
    the attenuation related to anelastic absorbtion and scattering
    """

    ar = geometrical_spreading(raypath)
    if quality:
        aq = anelastic_scattering_attenuation(raypath, velocity, quality, Mw)
        att = ar * aq
    else:
        att = ar

    return att


def calculate_attenuation_grid(seed, velocity, quality=None, locations=None,
                               triaxial=True,
                               orientation=(0, 0, 1), pwave=True, buf=0,
                               traveltime=None, eventSeed=False, Mw=-1.,
                               tt=None, return_tt=False,
                               homogeneous=True):
    """
    :param seed: seed (often receiver) location
    :type seed: tuple with same dimension as the grid
    :param velocity: velocity grid
    :type velocity: uquake.core.data.Grid
    :param locations: 2-D grid containing the coordinates at which the
    attenuation is calculated
    for instance [[x1,y1,z1],[x2,y2,z2],...,[xn,yn,zn]].
    :type location: 2D numpy array.
    :param quality: quality factor grid
    :type quality: uquake ImageData
    :param triaxial: true if triaxial sensor is used
    :type triaxial: bool
    :param orientation: sensor orientation taken into account only for
    uniaxial (non triaxial) sensors
    :type orientation: tuple
    :param pwave: true for p-wave
    :type pwave: bool
    :param progress: show progress bar if true
    :type progress: bool
    :param traveltime: precalculated traveltime grid (for MapReduce)
    :type traveltime: uquake.core.data.Grid
    :param eventSeed: True if seed is an event
    :type eventSeed: bool
    :param Mw: Moment magnitude
    :type Mw: float
    :param return_tt: if True, return traveltime grid
    :type return_tt: bool
    :param homogeneous: if True grid is considered homogeneous. Only
    geometrical spreading and radiation patterns
    are accounted for
    :type homogeneous: bool
    :rparam: Attenuation on a grid
    :rtype: uquake ImageData and

    .. note::

    The attenuation is calculated only at points within the velocity grid.
    Attenuation outside of the velocity grid have a value of 0.

    To generate the location grid, one could use the following:
        location = [coord for coord in zip(X, Y, Z)], where X, Y, and Z are
        vectors

    The script works both in 2D and 3D.

    .. example::

    """

    from uquake.simul import eik

    A = []

    if not np.any(locations):
        providedInputLocation = False
        x = np.arange(0, velocity.shape[0]) * velocity.spacing + \
            velocity.origin[0]
        y = np.arange(0, velocity.shape[1]) * velocity.spacing + \
            velocity.origin[1]
        z = np.arange(0, velocity.shape[2]) * velocity.spacing + \
            velocity.origin[2]

        X, Y, Z = np.meshgrid(x, y, z)

        locations = [coord for coord in zip(X.ravel(), Y.ravel(), Z.ravel())]
    else:
        providedInputLocation = True

    NoNode = len(locations)

    c1 = velocity.origin
    c2 = velocity.origin + (np.array(velocity.shape) - \
                            np.array([1, 1, 1])) * velocity.spacing

    for k, coord in enumerate(locations):
        pb()

        # if (np.any(coord - c1 < buf)) or (np.any(c2 - coord < buf)):
        #   A.append(0)
        #   continue

        dist = np.linalg.norm(coord - seed)

        if homogeneous:
            ray = np.array([seed, coord])
            quality = None
        else:
            if not tt:
                tt = eik.EikonalSolver(velocity, seed)

            ray = eik.RayTracer(tt, velocity, np.array(coord))

        if eventSeed:
            ray = ray[-1::-1, :]

        att = calculate_attenuation(ray, velocity, quality, Mw=Mw)

        att_comp = 1
        if not triaxial:

            incidenceVector = tmp / np.linalg.norm(tmp)
            tmpatt = np.dot(incidenceVector, np.array(orientation))
            if pwave:
                att_comp = tmpatt
            else:
                att_comp = np.sqrt(1 - tmpatt ** 2)

        A.append(att * att_comp)

    if not providedInputLocation:
        tmp = np.array(A).reshape(velocity.shape)
    else:
        tmp = np.array(A)

    if return_tt:
        if not tt:
            tt = eik.EikonalSolver(velocity, seed)

        import scipy.ndimage as ndimage

        tt_out = Grid(
            ndimage.map_coordinates(tt.data, tt.transform_to(locations).T),
            spacing=velocity.shape, origin=velocity.origin)
        return Grid(tmp, spacing=velocity.shape,
                        origin=velocity.origin), tt_out
    else:
        return Grid(tmp, spacing=velocity.shape, origin=velocity.origin)


def Mw2M0(Mw):
    """Converts Moment Magnitude, Mw, into Seismic Moment, M0.
    The relation between Moment magnitude and seismic moment is
        Mw = 2/3 log_10(M0) - 6.02 (Hank and Kanamori, 1979)

    :param Mw: Moment magnitude
    :type Mw: float
    :returns: Seismic moment
    :rtype: float
    """

    return 10 ** ((3 / 2.0) * (Mw + 6.07))


def M02Mw(M0):
    """Converts Seismic Moment, M0 into Moment Magnitude, Mw,.
    The relation between Moment magnitude and seismic moment is
        Mw = 2/3 log_10(M0) - 6.02 (Hank and Kanamori, 1979)

    :param Mw: Moment magnitude
    :type Mw: float
    :returns: Seismic moment
    :rtype: float
    """

    return 2 / 3.0 * np.log10(M0) - 6.07


def corner_frequency(Mw, vp=5000.0, vs=3500.0, SSD=1):
    """Calculates the corner frequency, f0, from the moment magnitude,
    Mw, we use the following relationship

    log10(f0) = 1.32 + 0.33 * log10(SSD) + log10(v) - 0.5 * Mw,

    where SSD is the static stress drop in "bar", and v the velocity in "km/s".

    --> Carefully note the unit of the input parameters

    :param Mw: Moment magnitude of the event
    :type Mw: float
    :param vp: p-wave velocity in m/s
    :type vp: float
    :param vs: s-wave velocity in m/s
    :type vs: float
    :param SSD: Static stress drop in "bar" generally though to change
    from 1 - 100 bar.
    :type SSD: float
    :returns: f0_p and f0_s, the corner frequency in Hz for the P- and S-waves,
    respectively (f0_p and f0_s).
    :rtype: tuple of float

    .. note:: For tectonic earthquake, the SSD is reported to be between
    1 - 100.
    For microseismic event, the SSD is generally close to 1 and can be lower.
    """

    f0_p = 10 ** (
            1.32 + 0.33 * np.log10(SSD) + np.log10(vp / 1000.0) - 0.5 * Mw)
    f0_s = 10 ** (
            1.32 + 0.33 * np.log10(SSD) + np.log10(vs / 1000.0) - 0.5 * Mw)

    return (f0_p, f0_s)


def synthetic_seismogram(Mw, duration=0.1, sampling_rate=10000, vp=5000.0,
                         vs=3500.0, rho=2400, SSD=1, pwave=True):
    """
    Create a synthetic displacement pulse at the source seismogram based
    on the brune model (Brune 1970).
    This model is extensively used and generally agrees with observations from
    many different setting and over a large range of magnitude.

    The displacement time function, u(t), is expressed as follows:

        u(t) = A_0 x t x omega_0 x H(t) * exp(-t x omega_0) ,

    where t the time, omega_0, the angular frequency and H(t) is the
    heavyside function. Note that the angular frequency is calculated from
    the static stress drop (SSD). A0 is given by the following
    equation:

        A0 = M0 / (4 * pi * rho * v ** 3)

    References for further reading:
    - Routine data processing in earthquake seismology
    - Relating Peak Particle Velocity and Acceleration to Moment Magnitude
    in Passive (Micro-) Seismic Monitoring
    (www.bcengineers.com/images/BCE_Technical_Note_3.pdf)

    :param Mw: the moment magnitude of the seismic event (default: -1),
    this value determine the wave amplitude and the
    frequency content
    :type Mw: float
    :param noise_level: level of gaussian noise to add to the synthetic
    seismogram (default: 1e-5)
    :type noise_level: float
    :param duration: duration of the seismogram in seconds (default: 0.1),
    the pulse is centered at zero
    :type duration: float
    :param sampling_rate: sampling rate in Hz of the generated time series (
    default: 10000)
    :type sampling_rate: int
    :param vp: P-wave velocity of the material at the source (default: 5000 m/s)
    :type vp: float
    :param vs: S-wave velocity of the material at the source (default: 3500 m/s)
    :type vs: float
    :param rho: density of the material at the source in kg/m**3 (default:
    2400 kg/m**3)
    :param SSD: Static stress drop in "bar" (default: 1 bar)
    :type SSD: float
    :param pwave: Return P-wave displacement seismogram if True and S-wave
    displacement seismogram if false
    :rparam: tuple Obspy Trace containing the seismogram
    :rtype: Obspy Trace

    .. note::
        The velocity and acceleration can easily be obtained by
        differentiating the trace using the Obspy Trace method
        differentiate.

        Example
        >>> tr = synthetic_seismogram(-1)
        >>> tr.differentiate()  # this creates a velocity trace
        >>> tr.differentiate()  # this creates an acceleration trace

        This operation is performed in place on the actual data arrays. The
        raw data is not accessible anymore afterwards. To keep your
        original data, use :meth:`~obspy.core.trace.Trace.copy` to create
        a copy of your trace object.
        This also makes an entry with information on the applied processing
        in ``stats.processing`` of this trace.

    """

    M0 = Mw2M0(Mw)
    (f0p, f0s) = corner_frequency(Mw, vp, vs, SSD)
    # duration = 5 / f0p
    npts = duration * sampling_rate
    t = np.arange(
        npts) / sampling_rate
    if pwave:
        W0 = 2 * np.pi * f0p
        v = vp
    else:
        W0 = 2 * np.pi * f0s
        v = vs

    A0 = M0 / (4 * np.pi * rho * v ** 3)

    data = A0 * t * W0 ** 2 * np.exp(-t * W0)
    data = np.roll(data, len(data) / 2)

    stats = Stats()
    stats.sampling_rate = sampling_rate
    stats.npts = 2 * npts - 1

    from uquake.core.util.cepstrum import minimum_phase
    minphase_data = np.roll(minimum_phase(data, len(data)), len(data) / 2)

    return Trace(data=data, header=stats)


def detection_level_sta_lta_grid(attenuationGrid, VpGrid, VsGrid,
                                 noise_level=1e-3, acceleration=True,
                                 STALTA_threshold=3, SSD=0.1,
                                 minMag=-3., maxMag=2., magResolution=0.1,
                                 pwave=True):
    """
    Returns a grid containing the minimum magnitude detectable at the
    location stloc
    with the same dimensions as the attenuation grid.

    :param attenuationGrid: Attenuation grid
    :type attenuationGrid: uquake.core.data.Grid
    :param VpGrid: P-wave velocity grid
    :type VpGrid: uquake.core.data.Grid
    :param VsGrid: S-wave velocity grid
    :type VsGrid: uquake.core.data.Grid
    :param noise_level: Level of noise to add to the seismogram
    :type noise_level: float
    :param acceleration: if True use acceleration if false use velocity
    :type acceleration: bool
    :param STALTA_threshold: STA/LTA threshold to declare a trigger.
    (default 3).
    :type STALTA_threshold: float
    :param SSD: Static stress drop in "bar" (default: 1 bar)
    :type SSD: float
    :param minMag: minimum moment magnitude to be tested (default=-3)
    :type minMag: float
    :param maxMag: maximum moment magnitude to be tested (default=2)
    :type maxMag: float
    :param magResolution: resolution of magnitude increments between minMag
    and maxMag
    :type magResolution: float
    :param pwave: True if it is a P-wave
    :type pwave: bool
    :rtype: uquake.core.data.Grid
    """

    Sensitivity = attenuationGrid.copy()
    Sensitivity.data[:] = maxMag
    magnitudes = np.arange(minMag, maxMag + magResolution, magResolution)
    for Mw in magnitudes[-1::-1]:
        Pulse = synthetic_seismogram(Mw, duration=1.5, sampling_rate=10000,
                                     vp=5000.0, vs=3500.0,
                                     rho=2400, SSD=SSD,
                                     pwave=pwave)  # Displacement

        Pulse.differentiate()  # Velocity
        if acceleration:
            Pulse.differentiate()  # Acceleration

        amp = np.std(Pulse.data)

        attAmp = attenuationGrid.data * amp
        attEnergy = attAmp ** 2

        ENoise = noise_level ** 2

        STALTA = attEnergy / ENoise

        Sensitivity.data[STALTA >= STALTA_threshold] = Mw

    return Sensitivity


def triggered_sensor_sta_lta_grid(stloc, Rho, Vp, Vs, Qp, Qs, Mw=-1,
                                  STALTA_threshold=3.0, noise_level=1e-5,
                                  acceleration=True, SSD=1, evloc=None,
                                  return_bool=True):
    """Return a boolean grid with same dimension as an input attenuation grid
    describing whether a sensor triggers or not.

    :param stloc: Location of the station to use to calculate the attenuation
    :type stloc: tuple of int or float
    :param Rho: Density grid (kg/m**3).
    :type Rho: ImageData
    :param Vp: P-wave velocity grid (m/s).
    :type Vp: ImageData
    :param Vs: S-wave velocity grid (m/s).
    :type Vs: ImageData
    :param Qp: Quality factor grid for P-wave.
    :type Qp: ImageData
    :param Qs: Quality factor grid for the S-wave.
    :type Qs: ImageData
    :param Mw: Moment magnitude of the event
    :type Mw: float
    :param STALTA_threshold: STA/LTA threshold to declare a trigger.
    (default 3).
    :type STALTA_threshold: float
    :param noise_level: Level of noise to add to the seismogram
    :type noise_level: float
    :param acceleration: if True use acceleration if false use velocity
    :type acceleration: bool
    :param evloc: Event location grid (optional). if None, simulation will
    be performed
    on each nodes of the property grids (Rho, Vp, Vs ...).
    :type evloc: ImageData
    :param return_bool: Return a boolean if True and int if False
    :type return_bool: bool
    :rparam: grid with the same dimension as the input grid
    :rtype: Numpy array of bool or int
    describing sensor triggering

    .. NOTE::

    The dimensions of Rho, Vp and Vs grids need to be the same.

    .. EXAMPLE::

    >>> from numpy.random import randn
    >>> from scipy.ndimage.filters import gaussian_filter
    >>> from uquake.data import ImageData
    >>> GridSpc = 10
    >>> Rho = ImageData(gaussian_filter(randn(100,100) * 100 + 2400.0, 5),
    spacing=GridSpc)
    >>> Vp = ImageData(gaussian_filter(randn(100,100) * 100 + 5000.0),
    spacing=GridSpc)
    >>> Vs = ImageData(gaussian_filter(randn(100,100) * 100 + 3500.0),
    spacing=GridSpc)
    >>> stloc = (100, 100)
    >>> Trigger = TriggerSensorSTALTAGrid(stloc, Rho, Vp, Vs)

    """

    # in the future that might be interesting to enable processing on EMR
    # for portion of codes
    # without having to explicitely use MR_JOB.
    # e.g.
    # 1) open EMR connection and create instances (minimal amount of bootstrap
    #    the executable and library needed should be dowloaded to the server
    #    using a git clone
    #    command for instance)
    # 2) send command to EMR culuster
    # 3) retreive information
    # 4) shutdown EMR cluster

    # need to be fixed...
    # if not evloc:
    #   pass
    if evloc.ndim < 2:
        evloc = evloc[np.newaxis, :]

    # Calculating attenuation grids
    (PRadiation, SRadiation) = radiation_pattern_attenuation()

    logger.info('Calculating P-wave attenuation')
    Att_p = PRadiation * calculate_attenuation_grid(stloc, Vp, evloc,
                                                    quality=Qp)
    logger.info('Calculating S-wave attenuation')
    Att_s = SRadiation * calculate_attenuation_grid(stloc, Vs, evloc,
                                                    quality=Qs)

    # need to better writer the function to pass arguments to this function
    # nargs*
    PulseP = synthetic_seismogram(Mw, duration=0.1, sampling_rate=10000,
                                  vp=5000.0, vs=3500.0,
                                  rho=2400, SSD=SSD,
                                  pwave=True)  # Displacement
    PulseS = synthetic_seismogram(Mw, duration=0.1, sampling_rate=10000,
                                  vp=5000.0, vs=3500.0,
                                  rho=2400, SSD=SSD, pwave=False)

    PulseP.differentiate()  # Velocity
    PulseS.differentiate()
    if acceleration:
        PulseP.differentiate()  # Acceleration
        PulseS.differentiate()  # Acceleration

    ESignalP = np.var(PulseP.data)
    ESignalS = np.var(PulseS.data)

    # Something wrong with last few lines here
    ESignalP_Grid = ESignalP * Att_p
    ESignalS_Grid = ESignalS * Att_s

    ESignal_Grid = ESignalP_Grid.copy()

    ESignal_Grid[ESignalP_Grid < ESignalS_Grid] = ESignalS_Grid[
        ESignalP_Grid < ESignalS_Grid]

    ENoise = noise_level ** 2

    STALTA = ESignal_Grid / ENoise
    Trigger = STALTA >= STALTA_threshold

    diff = STALTA - STALTA_threshold
    # print diff
    logger.info('STALTA DIFF: %s' % (diff))
    # logger.info('STALTA: %.5f  THRESH: %.5f   DIFF: %.5f' % (STALTA,
    # STALTA_threshold,STALTA - STALTA_threshold))
    logger.info('TRIGGERED?: %s' % (Trigger))

    if not return_bool:
        Trigger = Trigger.astype(int)

    return Trigger


# Beyond this point the code needs some (significant) clean up.
# ---------------------------------------------------

# defining the spectral function
def spectral_function(f, log_omega0, fc, q):
    """Defines the far-field Brune displacement spectrum
    (Brune, 1970).

    This function is defined in base-10 log for analytical purposes,
    as singularities can cause the function to fail.

    :param f: frequency at which to calculate the power spectrum
    :type f: float
    :param log_omega0: the base-10 log of the low-frequency level
    :type log_omega0: float
    :param fc: the corner frequency
    :type fc: float
    :param q: quality factor

    :returns: the base-10 log of the power spectrum associated
    with the given frequency f
    :rtype: float
    """
    # if np.any((1.0+(f/float(f0))**2) <= 0):

    # scattering attenuation
    # it is assumed that the correction for the travel time has been done
    # outside of this function.
    scat_att = np.exp(-np.pi * f / q)

    return log_omega0 + np.log10(scat_att) - np.log10((1.0 + (f / fc) ** 2))


def MwFc(fn, m, b):
    """Defines the moment magnitude for a given corner frequency

    The common use is with the help of scipy.optimize.curve_fit,
    to find the parameters m and b that fit a straight line in
    the frequency domain (see interpolate_Fc_Mw())

    :param fn: the corner frequency
    :type fn: float
    :param m: the slope of the linear fit
    :type m: float
    :param b: the intercept of the linear fit
    :type b: float

    :returns: the moment magnitude
    :rtype: float
    """
    return m * np.log10(fn) + b


def FcMw(Mw, m, b):
    """Defines the corner frequency for a given moment magnitude

    The common use is with the help of scipy.optimize.curve_fit,
    to find the parameters m and b that fit a straight line in
    the frequency domain (see interpolate_Fc_Mw())

    :param Mw: the moment magnitude
    :type Mw: float
    :param m: the slope of the linear fit
    :type m: float
    :param b: the intercept of the linear fit
    :type b: float

    :returns: the corner frequency
    :rtype: float
    """
    return 10 ** ((Mw - b) / m)


# we may want to modify this function to account for the source radium
def interpolate_Fc_Mw():
    """Linear regression of the moment magnitude - corner frequency
    empirical relationship

    This function uses scipy.optimize.curve_fit,
    to find the parameters m and b that fit a straight line in
    for the MwFc() function.

    :returns: the linear regression parameters m and b and their covariances
    :rtype: float
    """
    fc = [2000, 1000, 500, 100, 50, 25, 10]
    Mw = [-3, -2, -1, 0, 1, 2, 3]
    popt, pcov = curve_fit(MwFc, fc, Mw)

    return (popt, pcov)


# def Displacementfn(Mw, fc, Rho, V, Fc=0.52):
#   """Calculates the Displacement at the source
#   for a given homogeneous density and velocity model.
#   based on http://www.bcengineers.com/images/BCE_Technical_Note_3.pdf

#   The function calculates the displacement based on the following
#   relationship

#   D = M_0 x t x Omega_0 x exp()

#   :param Mw: the moment magnitude
#   :type Mw: float
#   :param fc: the corner frequency
#   :type fc: float
#   :param Rho: Density at the source
#   :type Rho: float
#   :param V: Seismic velocity at the source
#   :type V: float
#   :param Fc: radiation factor (commonly set to 0.52 or 0.63 for P and S
#   wave respectively [Aki and Richard, 2002])
#   :type Fc: float

#   """


#   # u(t) = M_0 x t x Omega_0 x H(t) x exp(-t x Omega_0)
#   pass

#   M0 = 10 ** ((Mw + 6.01) * (3 / 2))  # the seismic moment
#   w0 = 2 * np.pi * fc  # the corner angular frequency
#   PPV = M0 * Fc * (w0 ** 2) / (4 * np.pi * Rho * (V ** 3))
#   return PPV, w0

def PPVfn(Mw, fc, Rho, V):
    """Calculates the peak-particle-velocity (PPV) at the source
    for a given homogeneous density and velocity model.

    :param Mw: the moment magnitude
    :type Mw: float
    :param fc: the corner frequency in Hz
    :type fc: float
    :param Rho: Density at the source in kg/m**3
    :type Rho: float
    :param V: Seismic velocity at the source in m/s
    :type V: float

    :returns: the PPV
    :rtype: float
    """
    M0 = Mw2M0(Mw)  # the seismic moment
    w0 = 2 * np.pi * fc
    PPV = w0 ** 2 * M0 / (4 * np.pi * Rho * (V ** 3))
    return PPV


def PPAfn(Mw, fc, Rho, V):
    """Calculates the peak-particle-acceleration (PPA) at the source
    for a given homogeneous density and velocity model.

    :param Mw: the moment magnitude
    :type Mw: float
    :param fc: the corner frequency in Hz
    :type fc: float
    :param Rho: Density at the source in kg/m**2
    :type Rho: float
    :param V: Seismic velocity at the source in m/s
    :type V: float


    :returns: the PPA
    :rtype: float
    """
    PPV = PPVfn(Mw, fc, Rho, V)
    w0 = 2 * np.pi * fc
    PPA = np.abs(PPV * (1 - 2 * w0))
    return PPA


def Attenuation(fc, V, R, Q=100):
    """Defines the attenuation of a wave due to
    geometrical spreading and energy (quality factor)
    dissipation.

    :param fc: the corner frequency
    :type fc: float
    :param V: the homogeneous model velocity
    :type V: float
    :param Q: the homogeneous quality factor
    :type Q: float
    :param R: the distance
    :type R: float

    :returns: the attenuation factor
    :rtype: float
    """
    att = (1 / R) * np.exp(-np.pi * fc * R / (V * Q))
    return att


# see http://www.bcengineers.com/images/BCE_Technical_Note_3.pdf
# PPVThreshold=4e-5
def return_triggered_sensor(epos, spos, sorient=None, stype=None, Magnitude=-1,
                            V=5000, Rho=2400, Fc=0.6,
                            use_sensor_orientation=False,
                            use_sensor_type=False, PPAThreshold=0.02):
    # The correction for sensor orientation is the projection of the
    # incoming wavefront with respect to the sensor orientation
    # the result is a fraction e.g., 0.2

    # The correction for sensor type is related to the frequency response
    # the result will also be a fraction e.g., 0.5

    popt, pcov = interpolate_Fc_Mw()
    fc = FcMw(Magnitude, popt[0], popt[1])
    Mw = MwFc(fc, popt[0], popt[1])

    R = np.linalg.norm(spos - epos, axis=1)
    PPA = PPAfn(Mw, fc, Rho, V) * Attenuation(fc, V, R) * \
          radiation_pattern_attenuation()[0]

    indices = np.nonzero(PPA >= PPAThreshold)[0]
    return indices


class Trigger():

    def __init__(self, Mw, vp=5000.0, vs=3500.0, rho=2400, STALTA_threshold=3.,
                 SSD=0.02,
                 noise_level=0.02, acceleration=True):

        st = synthetic_seismogram(Mw, vp=vp, vs=vs, rho=rho, SSD=SSD,
                                  pwave=True)
        if acceleration:
            st.differentiate().differentiate()
        else:
            st.differentiate()

        self.amp = np.std(st.data)
        self.STALTA_threshold = STALTA_threshold
        self.noise_level = noise_level

    def Trigger(self, raypath):
        att = geometrical_spreading(raypath) * radiation_pattern_attenuation()[
            0]

        signal_energy = (att * self.amp) ** 2

        return signal_energy / (self.noise_level ** 2) > self.STALTA_threshold


class Sensitivity():

    def __init__(self, Mw_min=-3., Mw_max=3., Mw_spacing=0.1, vp=5000.0,
                 vs=3500.0,
                 rho=2400, STALTA_threshold=3., SSD=1,
                 noise_level=0.02, acceleration=True, pwave=True):

        nMw = (Mw_max - Mw_min) / Mw_spacing + 1
        self.Mw = np.linspace(Mw_min, Mw_max, nMw)
        self.pwave = pwave

        signal_amplitude = []

        for Mw_ in self.Mw:
            st = synthetic_seismogram(Mw_, vp=vp, vs=vs, rho=rho, SSD=SSD,
                                      pwave=pwave, duration=0.5)

            if acceleration:
                amp = np.std(st.differentiate().differentiate().data)
            else:
                amp = np.std(st.differentiate())

            amp = np.std(st.data)
            signal_amplitude.append(amp)

        self.signal_amplitude = np.array(signal_amplitude)
        self.STALTA_threshold = STALTA_threshold
        self.noise_level = noise_level
        self.Mw_max = Mw_max

    def Sensitivity(self, raypath):
        if self.pwave:
            att = geometrical_spreading(raypath) * \
                  radiation_pattern_attenuation()[0]
        else:
            att = geometrical_spreading(raypath) * \
                  radiation_pattern_attenuation()[1]

        signal_energy = (self.signal_amplitude * att) ** 2
        if np.any(np.abs(signal_energy / (
                self.noise_level ** 2)) > self.STALTA_threshold):
            return self.Mw[np.abs(signal_energy / (
                    self.noise_level ** 2)) > self.STALTA_threshold][0]
        else:
            return self.Mw_max


def trigger_sensor(raypath, Mw, vp=5000.0, vs=3500.0,
                   rho=2400, STALTA_threshold=3., SSD=0.02,
                   noise_level=0.02, acceleration=True):
    """
    Calculate the sensitivity at a specific location given a series sensor
    locations (location of senosor within an array)
    :param raypath: coordinates of the segment endpoints along the raypath
    :type raypath: numpy array
    :param Mw: Moment magnitude of the event
    :type Mw: float
    :param sensor_locs: sensor locations
    :type sensor_locs: numpy array
    :param Mw: Minimum moment magnitude (Mw) to consider (default -3)
    :type Mw_min: float
    :param Mw_max: Maximum moment magnitude (Mw) to consider (default 1)
    :type Mw_max: float
    :param Mw_spacing: resolution of the moment magnitude search (default 0.1)
    :type Mw_spacing: float
    :param vp: P-wave velocity at the source location in m/s (default 5000.)
    :type vp: float
    :param vs: S-wave velocity at the source location in m/s (default 3500.)
    :type vs: float
    :param rho: Density at the source location in kg/m**3 (default 2400.)
    :type rho: float
    :param STALTA_threshold: STA/LTA threshold for a trigger to be declared
    (default 3)
    :type STALTA_threshold: float
    :param SSD: Static stress drop (default 0.01)
    :type SSD: float
    :param noise_level: Noise amplitude level at sensor in the unit of
    sensor response (e.g. velocity or acceleration) (default 0.02 (m/s**2))
    :type noise_level: either float or numpy array with the same dimensions
    as sensor_loc
    :param acceleration: If true noise_level is assumed to be acceleration
    and calculation are performed consequently
    :type acceleration: bool
    :rparam: sensor trigger station
    :rtype: bool
    """

    att = geometrical_spreading(raypath) * radiation_pattern_attenuation()[0]
    st = synthetic_seismogram(Mw, vp=vp, vs=vs, rho=rho, SSD=SSD, pwave=True)
    if acceleration:
        st.differentiate().differentiate()
    else:
        st.differentiate()

    amp = np.std(att * st.data)

    return (amp ** 2) / (noise_level ** 2) > STALTA_threshold


def measure_sensitivity(raypath, Mw_min=-3., Mw_max=2., Mw_spacing=0.1,
                        vp=5000.0, vs=3500.0,
                        rho=2400, STALTA_threshold=3., SSD=0.02,
                        noise_level=0.02, acceleration=True):
    """
    Calculate the sensitivity at a specific location given a series sensor
    locations (location of senosor within an array)
    :param raypath: coordinates of the segment endpoints along the raypath
    :type raypath: numpy array
    :param sensor_locs: sensor locations
    :type sensor_locs: numpy array
    :param Mw_min: Minimum moment magnitude (Mw) to consider (default -3)
    :type Mw_min: float
    :param Mw_max: Maximum moment magnitude (Mw) to consider (default 1)
    :type Mw_max: float
    :param Mw_spacing: resolution of the moment magnitude search (default 0.1)
    :type Mw_spacing: float
    :param vp: P-wave velocity at the source location in m/s (default 5000.)
    :type vp: float
    :param vs: S-wave velocity at the source location in m/s (default 3500.)
    :type vs: float
    :param rho: Density at the source location in kg/m**3 (default 2400.)
    :type rho: float
    :param STALTA_threshold: STA/LTA threshold for a trigger to be declared
    (default 3)
    :type STALTA_threshold: float
    :param SSD: Static stress drop (default 0.01)
    :type SSD: float
    :param noise_level: Noise amplitude level at sensor in the unit of
    sensor response (e.g. velocity or acceleration) (default 0.02 (m/s**2))
    :type noise_level: either float or numpy array with the same dimensions
    as sensor_loc
    :param acceleration: If true noise_level is assumed to be acceleration
    and calculation are performed consequently
    :type acceleration: bool
    :rparam: return the minimum matnigude that sensitivity in Miminum
    magnitude sensitivity
    :rtype: float

    .. note:

    """
    nMw = (Mw_max - Mw_min) / Mw_spacing + 1
    Mw = np.linspace(Mw_min, Mw_max, nMw)

    att = geometrical_spreading(raypath) * radiation_pattern_attenuation()[0]

    signal_energy = []

    for Mw_ in zip(Mw):
        st = synthetic_seismogram(Mw_, vp=vp, vs=vs, rho=rho, SSD=SSD,
                                  pwave=True, )
        if acceleration:
            amp = att * np.std(st.differentiate().differentiate().data)
        else:
            amp = att * np.std(st.differentiate())

        amp = np.std(att * st.data)
        signal_energy.append(amp ** 2)

    signal_energy = np.array(signal_energy)

    if np.any(np.abs(signal_energy / (noise_level ** 2)) > STALTA_threshold):
        return \
            Mw[np.abs(signal_energy / (noise_level ** 2)) > STALTA_threshold][
                0]
    else:
        return None


def moment_magnitude(stream, cat, inventory, vp, vs, only_triaxial=True,
                     density=2700, min_dist=20, win_length=0.04,
                     len_spectrum=2 ** 12, clipped_fraction=0.1,
                     max_frequency=600, preferred_origin_only=True):
    """
    WARNING
    Calculate the moment magnitude for an event.
    :param stream: seismogram
    :type stream: uquake.Stream # does not exist yet
    :param cat: catalog object
    :type cat: uquake.core.event.Catalog
    :param inventory: network information (contains stations information)
    :type inventory: uquake.station.Site
    :param vp: P-wave velocity
    :type vp: float or uquake.core.data.Grid
    :param vs: S-wave velocity
    :type vs: float or uquake.core.data.Grid
    :param only_triaxial: whether only triaxial sensor are used in the
    magnitude calculation (optional) (not yet implemented)
    :type only_triaxial: bool
    :param density: density in kg / m**3 (assuming homogeneous for now)
    :type density: float
    :param win_length: length of the window in second in which magnitude is
    calculated
    :type win_length: float
    :param min_dist: minimum distance between sensor an event to allow
    magnitude calculation
    :param len_spectrum: length of the spectrum
    :param clipped_fraction: allowed clipped fraction (fraction of the
    signal equal to the min or the max.
    :param max_frequency: maximum frequency used in the calculation on
    magnitude. After a certain frequency, the noise starts to dominate the
    signal and the biases the calculation of the magnitude and corner
    frequency.
    :param preferred_origin_only: calculates the magnitude for the
    preferred_origin only
    :rtype: uquake.core.event.Catalog
    """

    # rigidity in Pa (shear-wave modulus)

    if only_triaxial:
        logger.info(
            'only triaxial sensor will be used in magnitude calculation')

    fcs = []

    quality = {'station_code': [], 'phase': [], 'origin_id': [], 'quality': []}

    if preferred_origin_only:
        origins = [cat[0].preferred_origin()]

    else:
        origins = cat[0].origins

    for origin in origins:
        ev_loc = np.array([origin.x, origin.y, origin.z])

        if not ((type(vp) == np.float) or (type(vp) == np.int)):
            vp_src = vp.interpolate(ev_loc, grid_coordinate=False)
            vs_src = vs.interpolate(ev_loc, grid_coordinate=False)
        else:
            vp_src = vp
            vs_src = vs

        moment_magnitudes = []
        corner_frequencies = []
        stations = []

        spectrum_norm_matrix = []
        frequencies = []
        indices = []
        for k, arr in enumerate(origin.arrivals):
            pick = arr.get_pick()
            network_code = pick.waveform_id.network_code
            station_code = pick.waveform_id.station_code
            location_code = pick.waveform_id.location_code
            travel_time = arr.get_pick().time - origin.time
            # ensuring backward compatibility
            if not pick:
                pick = cat[0].picks[k]
            at = pick.time
            phase = pick.phase_hint

            sensor_response = inventory.select(network=network_code,
                                               station=station_code,
                                               location=location_code)

            if sensor_response is None:
                logger.warning(f'no response was found in the inventory for '
                               f'sensor '
                               f'{network_code}.{station_code}.'
                               f'{location_code}')
                continue

            if sensor_response[0][0][0].response is None:
                logger.warning(f'no response was found in the inventory for '
                               f'sensor '
                               f'{network_code}.{station_code}.'
                               f'{location_code}')
                continue

            st_loc = sensor_response[0][0][0].loc
            if not sensor_response:
                logger.warning(f'sensor response not found for sensor '
                               f'{network_code}.{station_code}'
                               f'.{location_code}')
                continue

            poles = np.abs(sensor_response[0][0][0].response.get_paz().poles)
            st_trs = stream.select(network=network_code,
                                   station=station_code,
                                   location=location_code)

            if len(st_trs) == 0:
                continue

            if only_triaxial and (len(st_trs) < 3):
                continue

            # creating displacement pulse
            st_trs.detrend('demean').detrend('linear')
            st_trs.taper(max_percentage=0.05, type='cosine')

            data = st_trs.composite()[0].data

            len_max = len(data[data == np.max(data)]) + \
                      len(data[data == np.min(data)])

            if len_max / len(data) > clipped_fraction:
                logger.info('Clipped waveform detected: station %s '
                            'will not be used for magnitude calculation' %
                            sensor_response.code)
                continue

            pulse = st_trs.copy()
            pulse.attach_response(inventory)

            # filter the pulse using the corner frequency of the sensor
            low_bp_freq = np.min(poles) / (2 * np.pi)
            high_bp_freq = np.max(poles) / (2 * np.pi)
            if high_bp_freq > pulse[0].stats.sampling_rate / 2:
                high_bp_freq = pulse[0].stats.sampling_rate / 2.5

            high_bp_freq = max_frequency
            pulse = pulse.taper(max_percentage=0.05, type='cosine')
            pulse.filter('bandpass', freqmin=low_bp_freq, freqmax=high_bp_freq)
            dp = pulse.remove_response(output='DISP')

            # ideally the sensor signal should be deconvolved and a larger
            # portion of the spectrum should be used. It is possible to get
            # to frequency lower than the corner frequency of the sensor
            # down the the noise floor. This would be a bit more
            # complicated. The max frequency could also be found looking at
            # the noise floor.

            # if hasattr(sensor_response[0][0][0], motion):
            #     motion = sensor_response[0][0][0]
            #     if sensor_response[0][0][0].motion == 'ACCELERATION':
            #         dp = pulse.copy().integrate().integrate()
            #     elif sensor_response[0][0][0].motion == 'VELOCITY':
            #         dp = pulse.copy().integrate()
            #
            # else:
            #     dp =

            # dp = pulse.copy()

            # creating a signal containing only one for comparison
            tr_one = Trace(data=np.ones(len(pulse[0].data)))
            tr_one.stats = pulse[0].stats
            st_one = Stream(traces=[tr_one])

            dp = dp.trim(starttime=at - 0.01, endtime=at + 2 * win_length)
            dp = dp.taper(type='cosine', max_percentage=0.5, max_length=0.08,
                          side='left')

            # applying the same operation to the one signal
            # st_one_trimmed = st_one.trim(starttime=at - 0.01,
            #                              endtime=at + 2 * win_length)
            # st_one_taper = st_one_trimmed.taper(type='cosine',
            #                                     max_percentage=0.5,
            #                                     max_length=0.08,
            #                                     side='left')
            #
            dp_spectrum = np.zeros(len_spectrum)
            water_level = 1e-15
            for tr in dp:
                dp_spectrum += np.abs(np.fft.fft(tr.data, n=len_spectrum))
            # one_spectrum = np.fft.fft(st_one_taper[0].data, n=len_spectrum)

            # dp_spectrum_scaled = dp_spectrum # / (one_spectrum + water_level)

            if arr.distance is not None:
                hypo_dist = arr.distance
            else:
                hypo_dist = np.linalg.norm(st_loc - ev_loc)

            if hypo_dist < min_dist:
                continue

            radiation = radiation_pattern_attenuation()
            if phase.lower() == 'P':
                radiation = radiation[0]
                v_src = vp_src
            else:
                radiation = radiation[1]
                v_src = vs_src

            sr = dp[0].stats.sampling_rate

            f = np.fft.fftfreq(len_spectrum, 1 / sr)

            anelastic = np.exp(travel_time)

            spectrum_norm = dp_spectrum / radiation * hypo_dist * 4 * \
                            np.pi * density * v_src ** 3 / sr * anelastic

            fi = np.nonzero((f >= low_bp_freq) & (f <= high_bp_freq))[0]
            # fr = np.nonzero((f < low_bp_freq) | (f > high_bp_freq))[0]
            # spectrum_norm[fr] = np.nan
            spectrum_norm_matrix.append(np.abs(spectrum_norm[fi]))
            frequencies.append(f[fi])
            indices.append(np.ones(len(fi)) * k)

        if not spectrum_norm_matrix:
            continue

        st_count = len(spectrum_norm_matrix)
        spectrum_norm = np.nanmedian(spectrum_norm_matrix, axis=0)
        f = np.median(frequencies, axis=0)
        fi = np.nonzero((np.isnan(spectrum_norm) == False) & (f > 0))[0]

        p_opt, p_cov = curve_fit(spectral_function, f[fi],
                                 np.log10(spectrum_norm[fi]),
                                 (10, 100, 100), bounds=((1, 0, 10),
                                                         (100, 1000, 5000)))

        mw = 2 / 3.0 * p_opt[0] - 6.02
        mu = 29.5e9
        dmw = 2 / 3.0 * p_cov[0, 0] - 6.02
        fc = p_opt[1]

        mag = event.Magnitude(mag=mw,
                              station_count=st_count, magnitude_type='Mw',
                              evaluation_mode=origin.evaluation_mode,
                              evaluation_status=origin.evaluation_status,
                              origin_id=origin.resource_id)

        mag.corner_frequency_hz = fc

        from obspy.core.event import QuantityError
        mag.mag_errors = QuantityError(uncertainty=dmw)
        # mag.fc_errors = QuantityError(uncertainty=dfc)
        cat[0].magnitudes.append(mag)
        if origin.resource_id == cat[0].preferred_origin().resource_id:
            cat[0].preferred_magnitude_id = mag.resource_id.id

    return cat
