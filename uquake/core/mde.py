from uquake.core.stream import Stream, Trace
from uquake.core.event import Catalog
from uquake.core.inventory import Inventory
from uquake.core import read, read_events, read_inventory
from pathlib import Path
import random
import tarfile
from io import BytesIO, StringIO
import tempfile
import string
import pyasdf


class MicroseismicDataExchange(object):
    """
    A class to handle the exchange of microseismic data (stream, catalog, and inventory) using the Zarr format.

    :ivar stream: The ObsPy/uQuake Stream object representing the seismic waveform data.
    :type stream: Stream, optional
    :ivar catalog: The ObsPy/uQuake Catalog object representing the event data.
    :type catalog: Catalog, optional
    :ivar inventory: The ObsPy/uQuake Inventory object representing the station and channel metadata.
    :type inventory: Inventory, optional
    """

    def __init__(self, stream: Stream = None, catalog: Catalog = None,
                 inventory: Inventory = None):
        """
        Initializes the MicroseismicDataExchange object.

        :param stream: The ObsPy/uQuake Stream object representing the seismic waveform data.
        :type stream: Stream, optional
        :param catalog: The ObsPy/uQuake Catalog object representing the event data.
        :type catalog: Catalog, optional
        :param inventory: The ObsPy/uQuake Inventory object representing the station and channel metadata.
        :type inventory: Inventory, optional
        """
        self.stream = stream
        self.catalog = catalog
        self.inventory = inventory

    def write(self, file_path: str, waveform_tag):
        """
        Writes the stream, catalog, and inventory data to a Zarr file.

        :param file_path: The path to the Zarr file where the data will be written.
        :type file_path: str
        """
        asdf_handler = ASDFHandler(file_path)
        asdf_handler.add_catalog(self.catalog)
        asdf_handler.add_inventory(self.inventory)
        asdf_handler.add_waveforms(self.stream, waveform_tag)

    @classmethod
    def read(cls, file_path: str) -> 'MicroseismicDataExchange':
        """
        Reads the stream, catalog, and inventory data from a Zarr file.

        :param file_path: The path to the Zarr file from which the data will be read.
        :type file_path: str
        :return: An instance of the MicroseismicDataExchange class with the read data.
        :rtype: MicroseismicDataExchange
        """
        asdf_handler = ASDFHandler(file_path)
        stream = asdf_handler.get_stream()
        catalog = asdf_handler.get_catalog()
        inventory = asdf_handler.get_inventory()

        return cls(stream=stream, catalog=catalog, inventory=inventory)


class ASDFHandler:
    def __init__(self, asdf_file_path):
        """
        Initialize the ASDFHandler with a given ASDF file path.
        :param asdf_file_path: Path to the ASDF file.
        """
        self.asdf_file_path = asdf_file_path
        self.ds = pyasdf.ASDFDataSet(self.asdf_file_path, mode='a')

    def add_catalog(self, catalog):
        """
        Add a seismic catalog to the ASDF dataset.
        :param catalog: ObsPy Catalog object
        """
        self.ds.add_quakeml(catalog)

    def get_catalog(self):
        """
        Retrieve the seismic catalog from the ASDF dataset.
        :return: ObsPy Catalog object
        """
        return self.ds.events

    def add_inventory(self, inventory):
        """
        Add a seismic inventory to the ASDF dataset.
        :param inventory: ObsPy Inventory object
        """
        self.ds.add_stationxml(inventory)

    def get_inventory(self):
        """
        Retrieve the seismic inventory from the ASDF dataset.
        :return: ObsPy Inventory object
        """
        return self.ds.waveforms[self.ds.waveforms.list()[0]].StationXML

    def add_waveforms(self, stream, tag):
        """
        Add a seismic stream to the ASDF dataset.
        :param stream: ObsPy Stream object
        :param tag: Tag describing the waveforms
        """
        for tr in stream:
            self.ds.add_waveforms(tr, tag)

    def get_waveforms(self, network, station, location, channel, starttime, endtime,
                      tag):
        """
        Retrieve specific waveforms from the ASDF dataset.
        :return: ObsPy Stream object
        """
        return self.ds.get_waveforms(network, station, location, channel, starttime,
                                     endtime, tag=tag)

# Note: Other functions (e.g., select_traces) can be added as needed. However, with ASDF and ObsPy's capabilities,
# the need for manual traversal, as seen in the Zarr example, is greatly reduced.


def read_mde(file_path):
    return MicroseismicDataExchange.read(file_path)


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
        network = tr.stats.network_code
        station = tr.stats.station_code
        location = tr.stats.location_code
        channel = tr.stats.channel_code

        # Create nested Zarr groups and dataset following the hierarchy
        network_group = root_group.require_group(network)
        station_group = network_group.require_group(station)
        location_group = station_group.require_group(location)
        arr = location_group.create_dataset(channel, data=tr.data,
                                            shape=(len(tr.data),),
                                            dtype='float32', overwrite=True)

        # Store selected stats as Zarr attributes
        for key in ['network_code', 'station_code', 'location_code',
                    'channel_code', 'sampling_rate', 'starttime', 'calib', 'resource_id']:
            # Convert non-string objects to strings for easier storage and retrieval
            arr.attrs[key] = str(tr.stats[key])



def validate_mde(filepath: str) -> dict:
    """
    Validates the contents of an .mde file.

    :param filepath: Path to the .mde file to be validated.
    :return: A dictionary containing the validation results.
    """
    validation_report = {
        'catalog': False,
        'stream': False,
        'inventory': False
    }

    # Open and extract files from the tarball
    with tarfile.open(filepath, 'r:gz') as tar:
        # Check for necessary files
        if 'catalog.xml' not in tar.getnames():
            raise ValueError("Missing catalog.xml in the .mde file!")
        if 'stream.mseed' not in tar.getnames():
            raise ValueError("Missing stream.mseed in the .mde file!")
        if 'inventory.xml' not in tar.getnames():
            raise ValueError("Missing inventory.xml in the .mde file!")

        # Validate catalog
        with tar.extractfile('catalog.xml') as f:
            catalog_bytes = f.read()
            catalog = read_events(BytesIO(catalog_bytes), format='quakeml')
            # ... perform more in-depth validation if necessary
            validation_report['catalog'] = True

        # Validate stream
        with tar.extractfile('stream.mseed') as f:
            stream_bytes = f.read()
            stream = read(BytesIO(stream_bytes), format='MSEED')
            # ... perform more in-depth validation if necessary
            validation_report['stream'] = True

        # Validate inventory
        with tar.extractfile('inventory.xml') as f:
            inventory_bytes = f.read()
            inventory = read_inventory(BytesIO(inventory_bytes), format='stationxml')
            # ... perform more in-depth validation if necessary
            validation_report['inventory'] = True

    return validation_report


def generate_unique_names(n):
    """
    Generate n number of unique 5-character names comprising only lower and upper case
    letters.

    :param n: The number of unique names to generate.
    :return: A list of n unique 5-character names.
    """
    names = set()  # Set to store unique names

    while len(names) < n:
        name = ''.join(random.choices(string.ascii_letters, k=5))
        names.add(name)

    return list(names)
