from uquake.core.data_exchange import MicroseismicDataExchange
from uquake.core.stream import Stream, Trace
from uquake.core.event import Catalog
from uquake.core.inventory import Inventory
from uquake.core import read_inventory, read_events
import io
import zarr


def write_zarr(filepath, mde: MicroseismicDataExchange):
    """
    Write a MicroseismicDataExchange object to a zarr file

    :param filepath: The path of the zarr file to write to
    :param mde: The MicroseismicDataExchange object to write to the zarr file
    """

    z = zarr.open(filepath, mode='w')

    # Serialize and store Catalog
    if mde.catalog is not None:
        z['catalog'] = mde.catalog.to_bytes
    else:
        z['catalog'] = None

    # Serialize and store Inventory
    if mde.inventory is not None:
        z['inventory'] = mde.inventory.to_bytes()
    else:
        z['inventory'] = None

    # Store Stream
    if mde.stream is not None:
        stream_to_zarr_group(mde.stream, zarr_group_path=filepath + '/stream')


def stream_to_zarr_group(stream, zarr_group_path):
    """
    Converts an ObsPy/uQuake stream to a Zarr group.
    Each trace is stored as a separate Zarr array with its associated metadata.
    :param stream: ObsPy/uQuake Stream object
    :param zarr_group_path: Path to create/save the Zarr group
    """

    # Create or open a Zarr group
    root_group = zarr.open_group(zarr_group_path, mode='a')

    for tr in stream:
        # Get metadata values
        network = tr.stats.network
        station = tr.stats.station
        location = tr.stats.location
        channel = tr.stats.channel

        # Create nested Zarr groups and dataset following the hierarchy
        network_group = root_group.require_group(network)
        station_group = network_group.require_group(station)
        location_group = station_group.require_group(location)
        arr = location_group.create_dataset(channel, data=tr.data,
                                            shape=(len(tr.data),),
                                            dtype='float32', overwrite=True)

        # Store selected stats as Zarr attributes
        for key in ['network', 'station', 'location',
                    'channel', 'sampling_rate', 'starttime', 'calib']:
            # Convert non-string objects to strings for easier storage and retrieval
            arr.attrs[key] = str(tr.stats[key])


def read_zarr(filepath):
    """
    Read a MicroseismicDataExchange object from a zarr file

    :param filepath: The path of the zarr file to read from
    :return: A reconstructed MicroseismicDataExchange object
    """

    z = zarr.open(filepath, mode='r')

    catalog = get_catalog(z)
    # Deserialize Inventory

    inventory = get_inventory(z)

    # Reconstruct Stream
    if 'stream' in z:
        stream = zarr_to_stream(filepath + '/stream')
    else:
        stream = None

    return MicroseismicDataExchange(catalog=catalog, inventory=inventory, stream=stream)


def get_catalog(z: zarr.hierarchy.group.Group):
    """
    Deserialize a Catalog from a Zarr group
    :param z: Zarr group
    :return:
    """
    if 'catalog' in z and z['catalog'] is not None:
        catalog_array = z['catalog']
        if catalog_array.ndim == 0:  # Check if it is a scalar (zero-dimensional)
            catalog_bytes = io.BytesIO(catalog_array[()])  # Access as a scalar
        else:
            catalog_bytes = io.BytesIO(catalog_array[:])  # Access normally for higher dimensions
        catalog = Catalog.read(catalog_bytes, format='QUAKEML')
    else:
        catalog = None

    return catalog


def get_inventory(z):
    if 'inventory' in z and z['inventory'] is not None:
        inventory_array = z['inventory']
        if inventory_array.ndim == 0:  # Check if it is a scalar (zero-dimensional)
            inventory_bytes = io.BytesIO(inventory_array[()])  # Access as a scalar
        else:
            inventory_bytes = io.BytesIO(inventory_array[:])  # Access normally for higher dimensions
        inventory = Inventory.read(inventory_bytes, format='STATIONXML')
    else:
        inventory = None

    return inventory


def zarr_to_stream(zarr_group_path):
    """
    Reconstructs an ObsPy/uQuake Stream from a Zarr group.
    :param zarr_group_path: Path to the Zarr group
    :return: An ObsPy/uQuake Stream object
    """

    root_group = zarr.open_group(zarr_group_path, mode='r')
    stream = Stream()
    traces = []

    for network in root_group:
        for station in root_group[network]:
            for location in root_group[network][station]:
                # Iterate over each channel within the location
                for channel in root_group[network][station][location]:
                    # Access the Zarr array for the channel
                    arr = root_group[network][station][location][channel]

                    # Process the array (e.g., reconstruct the Trace object)
                    tr = Trace(data=arr[:])

                    # Retrieve and set stats from Zarr attributes
                    for key in ['network', 'station', 'location', 'channel',
                                'sampling_rate', 'starttime', 'calib']:
                        # Assigning attributes to the Trace object from Zarr attributes
                        tr.stats[key] = arr.attrs[key]

                    traces.append(tr)

    return Stream(traces=traces)


def process_channel(arr, stream):
    """
    Process a single channel and add it to the stream.
    :param arr: Zarr array representing the channel data
    :param stream: Stream object to append the Trace to
    """
    tr = Trace(data=arr[:])
    for key in ['network', 'station', 'location', 'channel', 'sampling_rate', 'starttime', 'calib']:
        tr.stats[key] = arr.attrs[key]
    stream.append(tr)


