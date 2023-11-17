
from uquake.core.data_exchange import MicroseismicDataExchange
from uquake.core.stream import Stream, Trace
from uquake.core.event import Catalog
from uquake.core.inventory import Inventory
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
        catalog_bytes = io.BytesIO()
        mde.catalog.write(catalog_bytes, format='QUAKEML')
        catalog_bytes.seek(0)
        z['catalog'] = catalog_bytes.getvalue()
    else:
        z['catalog'] = None

    # Serialize and store Inventory
    if mde.inventory is not None:
        inventory_bytes = io.BytesIO()
        mde.inventory.write(inventory_bytes, format='STATIONXML')
        inventory_bytes.seek(0)
        z['inventory'] = inventory_bytes.getvalue()
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
                    'channel', 'sampling_rate', 'starttime', 'calib', 'resource_id']:
            # Convert non-string objects to strings for easier storage and retrieval
            arr.attrs[key] = str(tr.stats[key])


def read_zarr(filepath):
    """
    Read a MicroseismicDataExchange object from a zarr file

    :param filepath: The path of the zarr file to read from
    :return: A reconstructed MicroseismicDataExchange object
    """

    z = zarr.open(filepath, mode='r')

    # Deserialize Catalog
    if 'catalog' in z and z['catalog'] is not None:
        catalog_bytes = io.BytesIO(z['catalog'][:])
        catalog = Catalog.read(catalog_bytes, format='QUAKEML')
    else:
        catalog = None

    # Deserialize Inventory
    if 'inventory' in z and z['inventory'] is not None:
        inventory_bytes = io.BytesIO(z['inventory'][:])
        inventory = Inventory.read(inventory_bytes, format='STATIONXML')
    else:
        inventory = None

    # Reconstruct Stream
    if 'stream' in z:
        stream = zarr_to_stream(filepath + '/stream')
    else:
        stream = None

    return MicroseismicDataExchange(catalog=catalog, inventory=inventory, stream=stream)


def zarr_to_stream(zarr_group_path):
    """
    Reconstructs an ObsPy/uQuake Stream from a Zarr group.
    :param zarr_group_path: Path to the Zarr group
    :return: An ObsPy/uQuake Stream object
    """

    root_group = zarr.open_group(zarr_group_path, mode='r')
    stream = Stream()

    for network in root_group:
        for station in network:
            for location in station:
                for channel, arr in location.items():
                    # Reconstruct Trace object
                    tr = Trace(data=arr[:])
                    # Retrieve and set stats from Zarr attributes
                    for key in ['network', 'station', 'location',
                                'channel', 'sampling_rate', 'starttime', 'calib',
                                'resource_id']:
                        tr.stats[key] = arr.attrs[key]
                    stream.append(tr)

    return stream





