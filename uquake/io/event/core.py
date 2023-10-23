# Copyright (C) 2023, Jean-Philippe Mercier
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: event.py
#  Purpose: plugin for reading and writing GridData object into various format
#   Author: uquake development team
#    Email: devs@uquake.org
#
# Copyright (C) 2016 uquake development team
# --------------------------------------------------------------------
"""
plugin for reading and writing event (catalog) object from and into various
format

:copyright:
    uquake development team (devs@uquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""


def write_simple_sqlite(catalog, filename, **kwargs):
    """
    :param catalog: catalogue object
    :type catalog: uquake.core.catalog
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
                from uquake.core.event import Magnitude
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

            conn.execute(
                "INSERT INTO events(id, datetime, x, y, z, magnitude, " + \
                " magnitude_type, Ep, Es, number_triggers," + \
                " number_p_picks, number_s_picks) VALUES " + \
                " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (id, datetime, x, y, z, mag, mag_type, Ep, Es,
                 number_triggers, s_picks, p_picks))
