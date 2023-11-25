from uquake.core.data_exchange import ZarrHandler
from uquake.core import read_events

file = '/mnt/HDD_5TB_01/Cozamin/principal_events/zarr/Cozamin231115123813005.zarr'

mde = ZarrHandler.read(file)
cat = read_events('test.xml', format='QUAKEML')

