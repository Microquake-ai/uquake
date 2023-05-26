from uquake.core.stream import Stream
from uquake.core.event import Catalog
from uquake.core.inventory import Inventory


class SeismicDataEnsemble(object):
    def __init__(self, stream: Stream, catalog: Catalog, inventory: Inventory):
        self.stream = stream
        self.catalog = catalog
        self.inventory = inventory

    def write(self, filepath):
        filepath = Path(filepath)

        traces = []

        if len(self.inventory[0].code) > 2:
            self.inventory[0].alternate_code = self.inventory[0].code[:2]
        else:
            self.inventory[0].alternate_code = self.inventory[0].code

        alternate_station_codes = generate_unique_names(len(self.inventory[0].stations))
        for station, asc in zip(self.inventory[0].stations, alternate_station_codes):
            station.alternate_code = asc

            st = self.stream.copy().select(station=station.code)
            for tr in st:
                tr.stats.station = station.alternate_code
                tr.stats.network = self.inventory[0].alternate_code
                traces.append(tr)

        st = Stream(traces=traces)

        with tarfile.open(filepath.with_suffix('.uqe'), 'w:gz') as tar:

            catalog_bytes = BytesIO()
            self.catalog.write(catalog_bytes, format='quakeml')
            catalog_bytes.seek(0)
            tarinfo = tarfile.TarInfo('catalog.xml')
            tarinfo.size = len(catalog_bytes.getvalue())
            tar.addfile(tarinfo, catalog_bytes)

            stream_bytes = BytesIO()
            st.write(stream_bytes, format='MSEED', encodings='STEIM2')
            stream_bytes.seek(0)
            tarinfo = tarfile.TarInfo('stream.mseed')
            tarinfo.size = len(stream_bytes.getvalue())
            tar.addfile(tarinfo, stream_bytes)

            inventory_bytes = BytesIO()
            self.inventory.write(inventory_bytes, format='stationxml')
            inventory_bytes.seek(0)
            tarinfo = tarfile.TarInfo('inventory.xml')
            tarinfo.size = len(inventory_bytes.getvalue())
            tar.addfile(tarinfo, inventory_bytes)

        return None

    @classmethod
    def read(cls, file_path):
        file_path = Path(file_path).with_suffix('.uqe')
        catalog = None
        stream = None
        inventory = None

        with tarfile.open(file_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    file_bytes = tar.extractfile(member).read()
                    file_name = member.name

                    if file_name == 'catalog.xml':
                        catalog = read_events(BytesIO(file_bytes), format='quakeml')
                    elif file_name == 'stream.mseed':
                        stream = read(BytesIO(file_bytes), format='MSEED')
                    elif file_name == 'inventory.xml':
                        with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
                            tmpfile.write(file_bytes)
                            inventory = read_inventory(tmpfile.name, format='stationxml')

        traces = []
        if inventory and stream:
            for station in inventory[0].stations:
                st2 = stream.select(station=station.alternate_code)
                for tr in st2:
                    tr.stats.station = station.code
                    tr.stats.network = inventory[0].code
                    traces.append(tr)

        out_stream = Stream(traces=traces)

        return cls(out_stream, catalog, inventory)

    def patch_using_catalog(self, catalogue_line, network):
        event_type = event.uquake_event_from_letter_id(
            catalogue_line['EvtType'], network)
        self.catalog[0].event_type = event_type
        if catalogue_line['WF_Modified']:
            evaluation_mode = 'manual'
            if catalogue_line['Rejected']:
                evaluation_status = 'rejected'
            else:
                evaluation_status = 'final'
        else:
            evaluation_mode = 'automatic'
            if catalogue_line['MultiProc']:
                if catalogue_line['Rejected']:
                    evaluation_status = 'rejected'
                else:
                    evaluation_status = 'confirmed'
            elif catalogue_line['RemoteProc']:
                if catalogue_line['Rejected']:
                    evaluation_status = 'rejected'
                else:
                    evaluation_status = 'preliminary'

        for pick in self.catalog[0].picks:
            pick.evaluation_mode = evaluation_mode
            pick.evaluation_status = evaluation_status

        self.catalog[0].preferred_origin().evaluation_mode = evaluation_mode
        self.catalog[0].preferred_origin().evaluation_status = evaluation_status

        self.catalog[0].preferred_magnitude().evaluation_mode = evaluation_mode
        self.catalog[0].preferred_magnitude().evaluation_status = evaluation_status

        trg_id = catalogue_line['TrgID']

        self.catalog[0].resource_id.id = f'esg:/{network}/{trg_id}'


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
