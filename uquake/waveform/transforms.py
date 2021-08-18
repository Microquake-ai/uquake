import numpy as np
from uquake.helpers.logging import logger

"""
    Place to keep trace transforms - mainly rotations to ENZ, PSVSH, etc.
"""


def rotate_to_P_SV_SH(st, cat, debug=False):
    fname = 'rotate_to_P_SV_SH'

    st_new = st.copy()

    event = cat[0]
    for arr in event.preferred_origin().arrivals:
        if arr.phase == 'S':
            continue

        pk = arr.pick_id.get_referred_object()
        sta = pk.waveform_id.station_code
        baz = arr.backazimuth
        az = arr.azimuth
        takeoff = arr.takeoff_angle
        inc_angle = arr.inc_angle

        if inc_angle is None:
            baz, inc_angle = event.preferred_origin(
            ).get_incidence_baz_angles(sta, arr.phase)
            inc_angle *= 180 / np.pi

        if inc_angle is None:
            logger.warning("%s: sta:%s [%s] has inc_angle=None --> skip "
                           "rotation!" % (fname, sta, arr.phase))
            continue

        trs = st_new.select(station=sta)
        if len(trs) == 3:

            cos_i = np.cos(inc_angle * np.pi / 180.)
            sin_i = np.sin(inc_angle * np.pi / 180.)
            cos_baz = np.cos(baz * np.pi / 180.)
            sin_baz = np.sin(baz * np.pi / 180.)

            col1 = np.array([cos_i, sin_i, 0.])
            col2 = np.array([-sin_i * sin_baz, cos_i * sin_baz, -cos_baz])
            col3 = np.array([-sin_i * cos_baz, cos_i * cos_baz, sin_baz])

            A = np.column_stack((col1, col2, col3))

            if debug:
                print("sta:%s az:%.1f baz:%.1f takeoff:%.1f inc:%.1f" % (
                sta, az, baz, takeoff, inc_angle))

            E = trs[0].data
            N = trs[1].data
            Z = trs[2].data
            D = np.row_stack((Z, E, N))

            foo = A @ D

            # if sta in ['59', '87']:
            # trs.plot()

            trs[0].data = foo[0, :]
            trs[1].data = foo[1, :]
            trs[2].data = foo[2, :]
            trs[0].stats.channel = 'P'
            trs[1].stats.channel = 'SV'
            trs[2].stats.channel = 'SH'

            '''
            P = trs[0].copy().trim(starttime = pk.time -.02, endtime=pk.time +.02)
            SV = trs[1].copy().trim(starttime = pk.time -.02, endtime=pk.time +.02)
            SH = trs[2].copy().trim(starttime = pk.time -.02, endtime=pk.time +.02)

            S = np.sqrt(SV.data**2 + SH.data**2)
            print(type(S))
            print(S)

            PtoS = np.var(P.data)/np.var(S)
            print(type(PtoS))
            print(PtoS)

            print("P_max:%g SV_max:%g SH_max:%g P/S:%f" % (np.max(np.abs(P.data)), np.max(np.abs(SV.data)), \
                                                    np.max(np.abs(SH.data), PtoS)))

            #if sta in ['59', '87']:
                #trs.plot()
            #exit()
            '''


        else:
            print("sta:%s --> only has n=%d traces --> can't rotate" % (
            sta, len(trs)))

    return st_new


def rotate_to_ENZ(st, inventory):
    st_new = st.copy()

    for sta in st_new.unique_stations():

        trs = st_new.select(station=sta)

        if not inventory.select(sta):
            logger.warning(f'missing station "{sta}" in inventory')
            continue

        # catching edge case when a uniaxial sensors contains three traces
        # with two traces containing only NaN.

        if len(trs) == 3:
            if np.any([np.all(np.isnan(trs[0].data)),
                       np.all(np.isnan(trs[1].data)),
                       np.all(np.isnan(trs[2].data))]):
                continue

        if len(trs) == 3 and len(inventory.select(sta).channels) == 3:

            try:
                col1 = inventory.get_channel(sta=sta, cha='X').cosines
                col2 = inventory.get_channel(sta=sta, cha='Y').cosines
                col3 = inventory.get_channel(sta=sta, cha='Z').cosines
            except AttributeError as err:
                logger.error(err)
                continue

            A = np.column_stack((col1, col2, col3))
            At = A.transpose()

            x = trs[0].data
            y = trs[1].data
            z = trs[2].data
            D = np.row_stack((x, y, z))

            foo = At @ D

            trs[0].data = foo[0, :]
            trs[1].data = foo[1, :]
            trs[2].data = foo[2, :]
            trs[0].stats.channel = 'E'
            trs[1].stats.channel = 'N'
            trs[2].stats.channel = 'Z'

    return st_new
