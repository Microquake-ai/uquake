# import warnings
# warnings.simplefilter("ignore", UserWarning)
# warnings.simplefilter("ignore")

import numpy as np


def calc_static_stress_drop(Mw, fc, phase='S', v=3.5, use_brune=False):
    """
    Calculate static stress drop from moment/corner_freq relation
    Note the brune model (instantaneous slip) gives stress drops ~ 8 x lower
    than the Madariaga values for fcP, fcS

    :param Mw: moment magnitude
    :type Mw: float
    :param fc: corner frequency [Hz]
    :type fc: float
    :param phase: P or S phase
    :type phase: string
    :param v: P or S velocity [km/s] at source
    :type v: float
    :param use_brune: If true --> use Brune's original scaling
    :type use_brune: boolean
    :returns: static stress drop [MPa]
    :rtype: float

    """

    if use_brune:  # Use Brune scaling
        c = .375
    else:  # Use Madariaga scaling
        if phase == 'S':
            c = .21
        else:
            c = .32

    v *= 1e5  # cm/s

    a = c * v / fc  # radius of circular fault from corner freq

    logM0 = 3 / 2 * Mw + 9.1  # in N-m
    M0 = 10 ** logM0 * 1e7  # dyn-cm

    stress_drop = 7. / 16. * M0 * (1 / a) ** 3  # in dyn/cm^2
    stress_drop /= 10.  # convert to Pa=N/m^2

    return stress_drop / 1e6  # MPa


cos = np.cos
sin = np.sin
degs2rad = np.pi / 180.


def double_couple_rad_pat(takeoff_angle, takeoff_azimuth, strike, dip, rake,
                          phase='P'):
    """
    Return the radiation pattern value at the takeoff point (angle, azimuth) 
        for a specified double couple source
        see Aki & Richards (4.89) - (4.91)
    All input angles in degrees
    allowable phase = ['P', 'SV', 'SH']
    """

    fname = 'double_couple_rad_pat'
    i_h = takeoff_angle * degs2rad
    azd = (takeoff_azimuth - strike) * degs2rad
    # Below is the convention from Lay & Wallace - it looks wrong!
    # azd = (strike - takeoff_azimuth) * degs2rad
    strike = strike * degs2rad
    dip = dip * degs2rad
    rake = rake * degs2rad

    radpat = None
    if phase == 'P':
        radpat = cos(rake) * sin(dip) * sin(i_h) ** 2 * sin(2. * azd) \
                 - cos(rake) * cos(dip) * sin(2. * i_h) * cos(azd) \
                 + sin(rake) * sin(2. * dip) * (cos(i_h) ** 2 - \
                                                sin(i_h) ** 2 * sin(azd) ** 2) \
                 + sin(rake) * cos(2. * dip) * sin(2. * i_h) * sin(azd)

    elif phase == 'SV':
        radpat = sin(rake) * cos(2. * dip) * cos(2. * i_h) * sin(azd) \
                 - cos(rake) * cos(dip) * cos(2. * i_h) * cos(azd) \
                 + 0.5 * cos(rake) * sin(dip) * sin(2. * i_h) * sin(2. * azd) \
                 - 0.5 * sin(rake) * sin(2. * dip) * sin(2. * i_h) * (
                             1 + sin(azd) ** 2)

    elif phase == 'SH':
        radpat = cos(rake) * cos(dip) * cos(i_h) * sin(azd) \
                 + cos(rake) * sin(dip) * sin(i_h) * cos(2. * azd) \
                 + sin(rake) * cos(2. * dip) * cos(i_h) * cos(azd) \
                 - 0.5 * sin(rake) * sin(2. * dip) * sin(i_h) * sin(2. * azd)

    elif phase == 'S':
        radpat_SV = sin(rake) * cos(2. * dip) * cos(2. * i_h) * sin(azd) \
                    - cos(rake) * cos(dip) * cos(2. * i_h) * cos(azd) \
                    + 0.5 * cos(rake) * sin(dip) * sin(2. * i_h) * sin(
            2. * azd) \
                    - 0.5 * sin(rake) * sin(2. * dip) * sin(2. * i_h) * (
                                1 + sin(azd) ** 2)

        radpat_SH = cos(rake) * cos(dip) * cos(i_h) * sin(azd) \
                    + cos(rake) * sin(dip) * sin(i_h) * cos(2. * azd) \
                    + sin(rake) * cos(2. * dip) * cos(i_h) * cos(azd) \
                    - 0.5 * sin(rake) * sin(2. * dip) * sin(i_h) * sin(
            2. * azd)

        radpat = np.sqrt(radpat_SV ** 2 + radpat_SH ** 2)

    else:
        print("%s: Unrecognized phase[%s] --> return None" % (fname, phase))
        return None

    return radpat


def free_surface_displacement_amplification(inc_angle, vp, vs,
                                            incident_wave='P'):
    """
    Returns free surface displacement amplification for incident P/S wave
        see Aki & Richards prob (5.6)
    All input angles in degrees

    Not sure how useful this will be.
    e.g., It returns the P/SV amplifications for the x1,x3 incidence plane,
    but we rarely rotate into that coord system.
    """

    fname = 'free_surface_displacement_amplification'

    i = inc_angle * degs2rad
    p = sin(i) / vp
    cosi = cos(i)
    cosj = np.sqrt(1. - (vs * p) ** 2)
    p2 = p * p
    b2 = vs * vs
    a = (1 / b2 - 2. * p2)
    Rpole = a * a + 4. * p2 * cosi / vp * cosj / vs

    if incident_wave == 'P':
        x1_amp = 4. * vp / b2 * p * cosi / vp * cosj / vs / Rpole
        x2_amp = 0.
        # The - is because A&R convention has z-axis positive Down
        x3_amp = -2. * vp / b2 * cosi / vp * a / Rpole

    elif incident_wave == 'SV':
        x1_amp = 2. / vs * cosj / vs * a / Rpole
        x2_amp = 0.
        x3_amp = 4. / b * p * cosi / vp * cosj / vs / Rpole

    elif incident_wave == 'SH':
        x1_amp = 0.
        x2_amp = 2.
        x3_amp = 0.
    else:
        print("%s: Unrecognized incident wave [%s] --> return None" %
              (fname, incident_wave))
        return None

    return np.array([x1_amp, x2_amp, x3_amp])
