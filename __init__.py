# insert supported read/write format plugin lists dynamically in docstrings
from obspy.core.util.base import _add_format_plugin_table


_add_format_plugin_table(read, "waveform", "read", numspaces=4)
_add_format_plugin_table(read_events, "event", "read", numspaces=4)
_add_format_plugin_table(read_inventory, "inventory", "read", numspaces=4)
_add_format_plugin_table(Stream.write, "waveform", "write", numspaces=8)
_add_format_plugin_table(Catalog.write, "event", "write", numspaces=8)
_add_format_plugin_table(Inventory.write, "inventory", "write", numspaces=8)
