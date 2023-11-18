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

# insert supported read/write format plugin lists dynamically in docstrings
from obspy.core.util.base import _add_format_plugin_table
from uquake.core import read, read_events, read_inventory


_add_format_plugin_table(read, "waveform", "read", numspaces=4)
_add_format_plugin_table(read_events, "event", "read", numspaces=4)
_add_format_plugin_table(read_inventory, "inventory", "read", numspaces=4)
_add_format_plugin_table(read_mde, "data_exchange", "read", numspaces=4)
_add_format_plugin_table(Stream.write, "waveform", "write", numspaces=8)
_add_format_plugin_table(Catalog.write, "event", "write", numspaces=8)
_add_format_plugin_table(Inventory.write, "inventory", "write", numspaces=8)
_add_format_plugin_table(MicroseismicDataExchange.write, "data_exchange", "write",
                         numspaces=8)
