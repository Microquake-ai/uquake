
import matplotlib.pyplot as plt
import numpy as np
from obspy.core.event.base import Comment
from obspy.core.event.source import FocalMechanism, NodalPlane, NodalPlanes
from obspy.imaging.beachball import aux_plane

from hashwrap.hashwrapper import calc_focal_mechanisms
from .logging import logger


def calc(cat, settings):
    """
    Prepare input arrays needed to calculate focal mechanisms
    and pass these into hashwrap.hashwrapper

    Return list of obspy focalmechanisms & list of matplotlib figs

    :param cat: obspy.core.event.Catalog
    :type list: list of obspy.core.event.Events or microquake.core.event.Events
    :param settings:hash settings
    :type settings dictionary

    :returns: obsy_focal_mechanisms, matplotlib_figures
    :rtype: list, list
    """

    fname = 'calc_focal_mechanism'

    plot_focal_mechs = settings.plot_focal_mechs

    sname = []
    p_pol = []
    p_qual = []
    qdist = []
    qazi = []
    qthe = []
    sazi = []
    sthe = []

    events = []

    for event in cat:

        event_dict = {}

        origin = event.preferred_origin()

        event_dict['event_info'] = origin.time.datetime.strftime('%Y-%m-%d '
                                                                 '%H:%M:%S')
        event_dict['event'] = {}
        event_dict['event']['qdep'] = origin.loc[2]
        event_dict['event']['sez'] = 10.
        event_dict['event']['icusp'] = 1234567

        arrivals = [arr for arr in event.preferred_origin().arrivals if
                    arr.phase == 'P']

        for arr in arrivals:

            if not arr.get_pick():
                logger.warning(
                    f"Missing pick for arrival {arr.resource_id} on"
                    f" event {event.resource_id}")
                continue

            if arr.get_pick().snr is None:
                logger.warning("%s P arr pulse_snr == NONE !!!" %
                               arr.pick_id.get_referred_object(
                               ).waveform_id.station_code)
                continue

            if arr.polarity is None:
                continue

            sname.append(arr.pick_id.get_referred_object(
            ).waveform_id.station_code)
            p_pol.append(arr.polarity)
            qdist.append(arr.distance)
            qazi.append(arr.azimuth)
    # MTH: both HASH and test_stereo expect takeoff theta measured wrt
            # vertical Up!
            qthe.append(180. - arr.takeoff_angle)
            sazi.append(2.)
            sthe.append(10.)

            if arr.get_pick().snr <= 6:
                qual = 0
            else:
                qual = 1
            p_qual.append(qual)

        event_dict['sname'] = sname
        event_dict['p_pol'] = p_pol
        event_dict['p_qual'] = p_qual
        event_dict['qdist'] = qdist
        event_dict['qazi'] = qazi
        event_dict['qthe'] = qthe
        event_dict['sazi'] = sazi
        event_dict['sthe'] = sthe

    events.append(event_dict)

    outputs = calc_focal_mechanisms(events, settings,
                                    phase_format='FPFIT')

    focal_mechanisms = []

    plot_figures = []

    for i, out in enumerate(outputs):
        logger.info("%s.%s: Process Focal Mech i=%d" % (__name__, fname, i))
        p1 = NodalPlane(strike=out['strike'], dip=out['dip'], rake=out['rake'])
        s, d, r = aux_plane(out['strike'], out['dip'], out['rake'])
        p2 = NodalPlane(strike=s, dip=d, rake=r)

        fc = FocalMechanism(nodal_planes=NodalPlanes(nodal_plane_1=p1,
                                                     nodal_plane_2=p2),
                            azimuthal_gap=out['azim_gap'],
                            station_polarity_count=out[
                                'station_polarity_count'],
                            station_distribution_ratio=out['stdr'],
                            misfit=out['misfit'],
                            evaluation_mode='automatic',
                            evaluation_status='preliminary',
                            comments=[Comment(text="HASH v1.2 Quality=[%s]"
                                                   % out['quality'])]
                            )

        focal_mechanisms.append(fc)

        event = events[i]

        title = "%s (s,d,r)_1=(%.1f,%.1f,%.1f) _2=(%.1f,%.1f,%.1f)" % \
                (event['event_info'], p1.strike, p1.dip, p1.rake, p2.strike,
                 p2.dip, p2.rake)

        if plot_focal_mechs:
            gcf = test_stereo(np.array(event['qazi']),
                              np.array(event['qthe']),
                              np.array(event['p_pol']),
                              sdr=[p1.strike, p1.dip, p1.rake],
                              event_info=event['event_info'])
            # sdr=[p1.strike,p1.dip,p1.rake], title=title)
            plot_figures.append(gcf)

    return focal_mechanisms, plot_figures


def test_stereo(azimuths, takeoffs, polarities, sdr=[], event_info=None):
    '''
        Plots points with given azimuths, takeoff angles, and
        polarities on a stereonet. Will also plot both planes
        of a double-couple given a strike/dip/rake
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='stereonet')
    up = polarities > 0
    dn = polarities < 0

    plot_upper_hemisphere = True

# This assumes takeoffs are measured wrt vertical up
#   so that i=0 (vertical) up has plunge=90:
#          ax.line(plunge, trend) - where plunge=90 plots at center and
    #          plunge=0 at edge
    h_rk = ax.line(90.-takeoffs[up], azimuths[up], 'bo')  # compressional
    # first arrivals
    h_rk = ax.line(90.-takeoffs[dn], azimuths[dn], 'go', fillstyle='none')

    if sdr:
        s1, d1, r1 = sdr[0], sdr[1], sdr[2]
        s2, d2, r2 = aux_plane(*sdr)

        if plot_upper_hemisphere:
            s1 += 180.
            s2 += 180.
        h_rk = ax.plane(s1, d1, 'g')
        ax.pole(s1, d1, 'gs', markersize=7)
        h_rk = ax.plane(s2, d2, 'b')
        ax.pole(s2, d2, 'bs', markersize=7)

    ax.grid(True)

    plt.title("upper hemisphere")

    title = event_info + " (s,d,r)=(%.1f, %.1f, %.1f)" % (s1, d1, r1)

    if title:
        plt.suptitle(title)

    # plt.show()

    return plt.gcf()
