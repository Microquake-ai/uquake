from .event import generate_catalog
from .inventory import generate_inventory
from .stream import generate_waveform
from uquake.core.data_exchange import MicroseismicDataExchange


def generate_mde(n_events=1, n_stations=20):
    """
    Generate a MicroseismicDataExchange object
    :return:
    """

    inventory = generate_inventory(n_stations=n_stations)
    streams = generate_waveform(inventory)
    catalog = generate_catalog(n_events=n_events)

    return MicroseismicDataExchange(stream=streams, catalog=catalog,
                                    inventory=inventory)
