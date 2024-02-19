from uquake.core.stream import Stream, Trace
from uquake.core.event import Catalog
from uquake.core.inventory import Inventory
from uquake.core import read_inventory, read_events
import io
import zarr
from pathlib import Path
import numpy as np
from typing import List, Union


def write_zarr(filepath, mde):
    """
    Write a MicroseismicDataExchange object to a zarr file

    :param filepath: The path of the zarr file to write to
    :param mde: The MicroseismicDataExchange object to write to the zarr file
    """

    if isinstance(filepath, Path):
        filepath = str(filepath)

    z = zarr.open(filepath, mode='w')

    # Serialize and store Catalog
    if mde.catalog is not None:
        z['catalog'] = mde.catalog.to_bytes
    else:
        pass

    # Serialize and store Inventory
    if mde.inventory is not None:
        z['inventory'] = mde.inventory.to_bytes()
    else:
        pass

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
        for key in tr.stats.__dict__.keys():
            arr.attrs[key] = str(tr.stats[key])


def read_zarr(filepath, networks: List[str] = None, stations: List[str] = None,
              locations: List[str] = None, channels: List[str] = None):
    """
    Read a MicroseismicDataExchange object from a zarr file

    :param filepath: The path of the zarr file to read from
    :param networks: List of networks
    :param stations: List of stations
    :param locations: List of locations
    :param channels: List of channels
    :return: A reconstructed MicroseismicDataExchange object
    """

    if isinstance(filepath, Path):
        filepath = str(filepath)

    z = zarr.open(filepath, mode='r')

    catalog = get_catalog(filepath)
    # Deserialize Inventory

    inventory = get_inventory(filepath)

    # Reconstruct Stream
    if 'stream' in z:
        stream = zarr_to_stream(filepath + '/stream', networks=networks,
                                stations=stations, locations=locations,
                                channels=channels)
    else:
        stream = None

    return {'catalog': catalog,
            'inventory': inventory,
            'stream': stream}


def get_catalog(file_path):
    """
    Deserialize a Catalog from a Zarr group
    :param file_path: file path to the Zarr group
    :return: catalog
    """
    z = zarr.open(file_path, mode='r')
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


def get_inventory(file_path):
    """
    Deserialize an Inventory from a Zarr group
    :param file_path: file path to the Zarr group
    :return: inventory
    """
    z = zarr.open(file_path, mode='r')
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


def zarr_to_stream(zarr_group_path, networks: Union[List[str], str] = None,
                   stations: Union[List[str], str] = None,
                   locations: Union[List[str], str] = None,
                   channels: Union[List[str], str] = None):
    """
    Reconstructs an ObsPy/uQuake Stream from a Zarr group.
    :param zarr_group_path: Path to the Zarr group
    :return: An ObsPy/uQuake Stream object
    """

    root_group = zarr.open_group(zarr_group_path, mode='r')
    stream = Stream()
    traces = []

    if isinstance(networks, str):
        networks = [networks]
    if isinstance(stations, str):
        stations = [stations]
    if isinstance(locations, str):
        locations = [locations]
    if isinstance(channels, str):
        channels = [channels]

    list_networks = [network[0] for network in root_group.groups()]
    networks = networks if networks else list_networks
    for net in networks:
        list_stations = [station[0] for station in root_group[net].groups()]
        stations = stations if stations else list_stations
        for sta in stations:
            list_locations = [location[0] for location in
                              list(root_group[net][sta].groups())]
            locations = locations if locations else list_locations
            for loc in locations:
                if channels:
                    chas = channels
                else:
                    # List all datasets (channels) directly under the current location
                    chas = [name for name in root_group[net][
                        sta][loc].array_keys()]

                for cha in chas:
                    arr = root_group[net][sta][loc][cha]

                    # Construct the Trace object
                    tr_data = np.array(arr)
                    tr = Trace(data=tr_data)

                    # Retrieve and set stats from Zarr attributes
                    for key in arr.attrs.keys():
                        try:
                            tr.stats[key] = arr.attrs[key]
                        except AttributeError as e:
                            pass

                    stream.append(tr)

    return stream


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


