# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: event.py
#  Purpose: plugin for reading and writing GridData object into various format
#   Author: microquake development team
#    Email: devs@microquake.org
#
# Copyright (C) 2016 microquake development team
# --------------------------------------------------------------------
"""
plugin for reading and writing event (catalog) object from and into various
format

:copyright:
    microquake development team (devs@microquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""


def write_simple_sqlite(catalog, filename, **kwargs):
    """
    :param catalog: catalogue object
    :type catalog: microquake.core.catalog
    :param filename: output filename
    :type filename: str
    :return: None
    """

    table_creation_cmd = \
    """
    CREATE TABLE IF NOT EXISTS events(
    id text PRIMARY KEY,
    datetime text NOT NULL,
    x real NOT NULL,
    y real NOT NULL,
    z real NOT NULL,
    magnitude real NOT NULL,
    magnitude_type text NOT NULL,
    Ep real,
    Es real,
    number_triggers integer NOT NULL,
    number_p_picks integer NOT NULL,
    number_s_picks integer NOT NULL)
    """

    import sqlite3
    from numpy import unique

    conn = sqlite3.connect('example.sqlite')

    with conn:
        # create the event table if it does not exists
        conn.execute(table_creation_cmd)

        for event in catalog:
            # check if event exists in the database
            if event.preferred_origin():
                origin = event.preferred_origin()
            else:
                origin = event.origins[0]

            if event.preferred_magnitude():
                mag = event.preferred_magnitude().mag
                mag_type = event.preferred_magnitude().magnitude_type
            elif event.magnitudes:
                mag = event.magnitudes[0].mag
                mag_type = event.magnitudes[0].magnitude_type
            else:
                from microquake.core.event import Magnitude
                mag = -33
                mag_type = "NC"

            id = origin.time.strftime("%Y%m%d%H%M%S%f")
            datetime = origin.time.strftime("%Y/%m/%d %H:%M:%S.%f")
            x = origin.x
            y = origin.y
            z = origin.z
            Ep = 0
            Es = 0
            stations = []
            s_picks = 0
            p_picks = 0
            for arrival in origin.arrivals:
                pick = arrival.pick_id.get_referred_object()
                stations.append(pick.waveform_id.station_code)
                if pick.phase_hint == "S":
                    s_picks += 1
                else:
                    p_picks += 1
            number_triggers = len(unique(stations))

            if conn.execute("SELECT id FROM events WHERE datetime=?",
                            (datetime,)).fetchall():
                continue

            conn.execute("INSERT INTO events(id, datetime, x, y, z, magnitude, " + \
                         " magnitude_type, Ep, Es, number_triggers," + \
                         " number_p_picks, number_s_picks) VALUES " + \
                         " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (id, datetime, x, y, z, mag, mag_type, Ep, Es,
                          number_triggers, s_picks, p_picks))

