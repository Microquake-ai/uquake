import matplotlib.pyplot as plt
import numpy as np

from ..core.logging import logger
from obspy.imaging.beachball import beachball


def plot_beachball_obspy(cat, **kwargs):
    # default
    if 'facecolor' not in kwargs.keys():
        kwargs['facecolor'] = [0.5] * 3

    if 'width' not in kwargs.keys():
        kwargs['width'] = 600

    if cat[0].preferred_focal_mechanism is None:
        logger.warning('nothing to do, the catalog does not have a '
                       'preferred focal mechanism')

    np1 = cat[0].preferred_focal_mechanism().nodal_planes.nodal_plane_1

    strike = np1['strike']
    dip = np1['dip']
    rake = np1['rake']

    fm = [strike, dip, rake]

    beachball(fm, **kwargs)
    return plt.gca()


plot_beachball_obspy.__doc__ = beachball.__doc__


def plot_beachball(cat, lower_hemisphere=True, legend=False, output_file=None):
    if cat[0].preferred_focal_mechanism is None:
        logger.warning('nothing to do, the catalog does not have a '
                       'preferred focal mechanism')

    takeoffs = [ar.takeoff_angle for ar in cat[0].preferred_origin().arrivals]
    takeoffs = np.array(takeoffs)
    azimuths = [ar.azimuth for ar in cat[0].preferred_origin().arrivals]
    azimuths = np.array(azimuths)

    takeoffs = np.array(takeoffs)
    azimuths = np.array(azimuths)

    np1 = cat[0].preferred_focal_mechanism().nodal_planes.nodal_plane_1
    np2 = cat[0].preferred_focal_mechanism().nodal_planes.nodal_plane_2

    s1 = np1['strike']
    d1 = np1['dip']
    s2 = np2['strike']
    d2 = np2['dip']

    if lower_hemisphere:
        azimuths[takeoffs > 90] -= 180

    else:
        azimuths[takeoffs < 90] -= 180
        s1 += 180
        s2 += 180

    takeoffs[takeoffs > 90] = 180 - takeoffs[takeoffs > 90]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='stereonet')

    # plotting the nodal planes
    p1 = ax.plane(s1, d1, 'k')
    p2 = ax.plane(s2, d2, 'k')

    polarities = [ar.polarity for ar in cat[0].preferred_origin().arrivals]
    polarities = np.array(polarities, dtype=np.float)

    up = polarities > 0
    dn = polarities < 0

    # compression
    lup = ax.line(90 - takeoffs[up], azimuths[up], 'kx')
    # first arrivals
    ldn = ax.line(90 - takeoffs[dn], azimuths[dn], 'k.',
                  fillstyle='none')

    lines = [lup[0], ldn[0]]

    if legend:
        plt.legend(lines, ['up', 'down'])

    if output_file:
        plt.savefig(output_file, edgecolor='none')
