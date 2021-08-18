""" Collection of functions to calculate moment magnitude

"""

import warnings

import numpy as np
from uquake.helpers.logging import logger
from obspy.core.event import Comment, ResourceIdentifier, WaveformStreamID
from uquake.core.event import Magnitude
from obspy.core.event.magnitude import (StationMagnitude,
                                        StationMagnitudeContribution)

from uquake.core.event import Pick
from uquake.waveform.amp_measures import measure_pick_amps
from uquake.waveform.mag_utils import double_couple_rad_pat, \
    free_surface_displacement_amplification
from uquake.waveform.smom_measure_legacy import measure_pick_smom

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore")


def moment_magnitude_new(st, event,
                         vp=5300, vs=3500, density=2700,
                         only_triaxial=True,
                         min_dist=20,
                         fmin=20, fmax=1000,
                         use_smom=False):
    # TODO: this moment_magnitude_new function looks broken
    picks = event.picks

    if use_smom:
        measure_pick_smom(st, picks, debug=False)
        comment = "Average of freq-domain P & S moment magnitudes"
    else:
        measure_pick_amps(st, picks, debug=False)
        comment = "Average of time-domain P & S moment magnitudes"

    # Use time(or freq) lambda to calculate moment magnitudes for each arrival

    Mw_P, station_mags_P = calc_magnitudes_from_lambda(cat, vp=vp, vs=vs,
                                                       density=density,
                                                       P_or_S='P',
                                                       use_smom=use_smom)

    Mw_S, station_mags_S = calc_magnitudes_from_lambda(cat, vp=vp, vs=vs,
                                                       density=density,
                                                       P_or_S='S',
                                                       use_smom=use_smom)

    # Average Mw_P,Mw_S to get event Mw and wrap with list of
    #   station mags/contributions

    Mw = 0.5 * (Mw_P + Mw_S)

    station_mags = station_mags_P + station_mags_S
    set_new_event_mag(event, station_mags, Mw, comment)


def set_new_event_mag(event, station_mags, Mw, comment, make_preferred=False):
    count = len(station_mags)

    sta_mag_contributions = []

    for sta_mag in station_mags:
        sta_mag_contributions.append(StationMagnitudeContribution(
            station_magnitude_id=sta_mag.resource_id)
        )

    origin_id = event.preferred_origin().resource_id

    event_mag = Magnitude(origin_id=origin_id,
                          mag=Mw,
                          magnitude_type='Mw',
                          station_count=count,
                          evaluation_mode='automatic',
                          station_magnitude_contributions=sta_mag_contributions,
                          comments=[Comment(text=comment)],
                          )

    event.magnitudes.append(event_mag)
    event.station_magnitudes = station_mags

    if make_preferred:
        event.preferred_magnitude_id = ResourceIdentifier(
            id=event_mag.resource_id.id,
            referred_object=event_mag)

    return


def calc_magnitudes_from_lambda(cat,
                                vp=5300, vs=3500, density=2700,
                                P_or_S='P',
                                use_smom=False,
                                use_sdr_rad=False,
                                use_free_surface_correction=False,
                                min_dist=20.,
                                **kwargs):
    """
    Calculate the moment magnitude at each station from lambda,
      where lambda is either:
        'dis_pulse_area' (use_smom=False) - calculated by integrating arrival
            displacement pulse in time
        'smom' (use_smom=True) - calculated by fiting Brune spectrum to
            displacement spectrum in frequency
    """

    fname = 'calc_magnitudes_from_lambda'

    # Don't loop over event here, do it in the calling routine
    #   so that vp/vs can be set for correct source depth
    event = cat[0]
    origin = event.preferred_origin() if event.preferred_origin() else \
        event.origins[0]
    ev_loc = origin.loc
    origin_id = origin.resource_id

    rad_P, rad_S = 0.52, 0.63

    if P_or_S == 'P':
        v = vp
        rad = rad_P
        mag_type = 'Mw_P'
    else:
        v = vs
        rad = rad_S
        mag_type = 'Mw_S'

    if use_smom:
        magnitude_comment = 'station magnitude measured in frequeny-domain (' \
                            'smom)'
        lambda_key = 'smom'
    else:
        magnitude_comment = 'station magnitude measured in time-domain (' \
                            'dis_pulse_area)'
        lambda_key = 'dis_pulse_area'

    if use_free_surface_correction and np.abs(ev_loc[2]) > 0.:
        logger.warning("%s: Free surface correction requested for event ["
                       "h=%.1f] > 0" % (fname, ev_loc[2]))

    if use_sdr_rad and 'sdr' not in kwargs:
        logger.warning("%s: use_sdr_rad requested but NO [sdr] given!" % fname)

    station_mags = []
    Mw_list = []

    Mw_P = []

    arrivals = [arr for arr in event.preferred_origin().arrivals if
                arr.phase == P_or_S]

    for arr in arrivals:

        try:
            pk = arr.pick_id.get_referred_object()
            sta = pk.waveform_id.station_code
            cha = pk.waveform_id.channel_code
            net = pk.waveform_id.network_code
        except AttributeError:
            logger.warning('Missing data on arrival', exc_info=True)

            continue

        fs_factor = 1.

        if use_free_surface_correction:
            if arr.get('inc_angle', None):
                inc_angle = arr.inc_angle
                fs_factor = free_surface_displacement_amplification(
                    inc_angle, vp, vs, incident_wave=P_or_S)

                # MTH: Not ready to implement this.  The reflection coefficients
                #      are expressed in x1,x2,x3 coords
                # print("inc_angle:%.1f x1:%.1f x3:%.1f" % (inc_angle, fs_factor[0], fs_factor[2]))
                # MTH: The free surface corrections are returned as <x1,x2,x3>=<
                fs_factor = 1.
            else:
                logger.warning("%s: sta:%s cha:%s pha:%s: inc_angle NOT set "
                               "in arrival dict --> use default" %
                               (fname, sta, cha, arr.phase))

        if use_sdr_rad and 'sdr' in kwargs:
            strike, dip, rake = kwargs['sdr']

            if arr.get('takeoff_angle', None) and arr.get('azimuth', None):
                takeoff_angle = arr.takeoff_angle
                takeoff_azimuth = arr.azimuth
                rad = double_couple_rad_pat(takeoff_angle, takeoff_azimuth,
                                            strike, dip, rake, phase=P_or_S)
                rad = np.abs(rad)
                logger.debug("%s: phase=%s rad=%f" % (fname, P_or_S, rad))
                magnitude_comment += ' radiation pattern calculated for (s,' \
                                     'd,r)= (%.1f,%.1f,%.1f) theta:%.1f ' \
                                     'az:%.1f pha:%s |rad|=%f' % \
                                     (strike, dip, rake, takeoff_angle,
                                      takeoff_azimuth,
                                      P_or_S, rad)
                # logger.info(magnitude_comment)
            else:
                logger.warnng("%s: sta:%s cha:%s pha:%s: "
                              "takeoff_angle/azimuth NOT set in arrival dict --> use default radiation pattenr" %
                              (fname, sta, cha, arr.phase))

        _lambda = getattr(arr, lambda_key)

        if _lambda is not None:

            M0_scale = 4. * np.pi * density * v ** 3 / (rad * fs_factor)

            # R  = np.linalg.norm(sta_dict['station'].loc -ev_loc) #Dist in meters

            # MTH: obspy arrival.distance = *epicentral* distance in degrees
            #   >> Add attribute hypo_dist_in_m to uquake arrival class
            #         to make it clear
            if arr.distance:
                R = arr.distance
            else:
                R = arr.hypo_dist_in_m

            if R >= min_dist:

                M0 = M0_scale * R * np.abs(_lambda)
                Mw = 2. / 3. * np.log10(M0) - 6.033
                # print("MTH: _lambda=%g R=%.1f M0=%g" % (np.abs(_lambda), R, M0))

                Mw_list.append(Mw)

                station_mag = StationMagnitude(origin_id=origin_id, mag=Mw,
                                               station_magnitude_type=mag_type,
                                               comments=[Comment(
                                                   text=magnitude_comment)],
                                               waveform_id=WaveformStreamID(
                                                   network_code=net,
                                                   station_code=sta,
                                                   channel_code=cha))
                station_mags.append(station_mag)

            else:
                logger.info("arrival sta:%s pha:%s dist=%.2f < min_dist("
                            "=%.2f) --> Skip" % (fname, sta, arr.phase, R,
                                                 min_dist))

        # else:
        # logger.warning("arrival sta:%s cha:%s arr pha:%s lambda_key:%s is NOT SET --> Skip" \
        # % (sta, cha, arr.phase, lambda_key))

    logger.info("nmags=%d avg:%.1f med:%.1f std:%.1f" %
                (len(Mw_list), np.mean(Mw_list), np.median(Mw_list),
                 np.std(Mw_list)))

    return np.median(Mw_list), station_mags


def calculate_energy_from_flux(cat,
                               inventory,
                               vp,
                               vs,
                               rho=2700.,
                               use_sdr_rad=False,
                               use_water_level=False,
                               rad_min=0.2):
    fname = 'calculate_energy_from_flux'

    for event in cat:
        origin = event.preferred_origin() if event.preferred_origin() else \
        event.origins[0]

        use_sdr = False

        if use_sdr_rad and event.preferred_focal_mechanism() is not None:
            mech = event.preferred_focal_mechanism()
            np1 = event.preferred_focal_mechanism().nodal_planes.nodal_plane_1
            sdr = (np1.strike, np1.dip, np1.rake)
            use_sdr = True

        # for phase in ['P', 'S']:
        # for arr in [x for x in arrivals if x.phase == phase]:

        P_energy = []
        S_energy = []

        for arr in origin.arrivals:
            pk = Pick(arr.get_pick())
            # try:
            sta = pk.get_sta()
            sta_response = inventory.select(sta)
            # except AttributeError:
            #     logger.warning(
            #         f'Cannot get station for arrival "{arr.resource_id}"'
            #         f' for event "{event.resource_id}".')

            # continue
            phase = arr.phase

            if phase.upper() == 'P':
                velocity = vp.interpolate(sta_response.loc)
                rad_pat = 4 / 15

            elif phase.upper() == 'S':
                velocity = vs.interpolate(sta_response.loc)
                rad_pat = 2 / 5

            # could check for arr.hypo_dist_in_m here but it's almost identical
            R = arr.distance

            # MTH: Setting preferred flux = vel_flux_Q = attenuation tstar
            # corrected flux
            flux = 0

            if arr.vel_flux_Q is not None:
                flux = arr.vel_flux_Q
                logger.info("%s: vel_flux_Q exists in [%s], using this for "
                            "energy, arr for sta:%s" %
                            (fname, phase, sta))
            elif arr.vel_flux is not None:
                flux = arr.vel_flux
                logger.info("%s: vel_flux exists in [%s], using this for "
                            "energy, arr for sta:%s" %
                            (fname, phase, sta))
            else:
                logger.info("%s: No vel_flux set for arr sta:%s pha:%s --> "
                            "skip energy calc" %
                            (fname, sta, phase))

                continue

            energy = (4. * np.pi * R ** 2) * rho * velocity * flux

            scale = 1.

            if use_sdr:
                if arr.get('takeoff_angle', None) and arr.get('azimuth', None):
                    takeoff_angle = arr.takeoff_angle
                    takeoff_azimuth = arr.azimuth
                    strike = sdr[0]
                    dip = sdr[1]
                    rake = sdr[2]
                    rad = double_couple_rad_pat(takeoff_angle, takeoff_azimuth,
                                                strike, dip, rake, phase=phase)

                    if use_water_level and np.abs(rad) < rad_min:
                        rad = rad_min

                    scale = rad_pat / rad ** 2

            energy *= scale

            arr.energy = energy

            if phase == 'P':
                P_energy.append(energy)
            else:
                S_energy.append(energy)

        if not P_energy:
            P_energy = [0]
            logger.warning('No P energy measurements. The P-wave energy will '
                           'be set to 0, the total energy will not include '
                           'the P-wave energy. This will bias the total '
                           'energy value')
        if not S_energy:
            S_energy = [0]
            logger.warning(
                'No S energy measurements. The S-wave energy will be set to '
                '0, the total energy will not include the P-wave energy. '
                'This will bias the total energy value')

        energy_p = np.median(P_energy)
        energy_s = np.median(S_energy)
        E = energy_p + energy_s

        nvals = len(S_energy) + len(P_energy)
        comment = 'Energy [N-m] calculated from sum of median P + median S ' \
                  'energy'
        comment_ep = '"Ep":{}, "std_Ep":{}'.format(np.median(P_energy),
                                                   np.std(P_energy))
        comment_ep = '{' + comment_ep + '}'
        comment_es = '"Es":{}, "std_Es":{}'.format(np.median(S_energy),
                                                   np.std(S_energy))
        comment_es = '{' + comment_es + '}'

        if E > 0:
            # Note: this is not a "mag" (there is no log10).
            #       Just using magnitude class to hold it in quakeml

            energy_mag = Magnitude(origin_id=origin.resource_id,
                                   mag=E,
                                   magnitude_type='E',
                                   station_count=nvals,
                                   evaluation_mode='automatic',
                                   comments=[Comment(text=comment),
                                             Comment(text=comment_ep),
                                             Comment(text=comment_es)],
                                   )
            energy_mag.energy_p_joule = energy_p
            energy_mag.energy_s_joule = energy_s
            energy_mag.energy_joule = E

            event.magnitudes.append(energy_mag)

        else:
            logger.warning("%s: Calculated val of Energy E=[%s] nS=%d nP=%d "
                           "is not fit to keep!"
                           % (fname, E, nvals, len(P_energy)))
            logger.warning("%s: Energy mag not written to Quakeml" % fname)

    return cat


if __name__ == '__main__':
    main()
