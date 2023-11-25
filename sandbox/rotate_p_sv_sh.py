from uquake.core.data_exchange import ZarrHandler
from uquake.core import read_events
from uquake.grid import base
from importlib import reload
reload(base)
import matplotlib.pyplot as plt
import numpy as np

file = '/mnt/HDD_5TB_01/Cozamin/principal_events/zarr/Cozamin231115123813005.zarr'

mde = ZarrHandler.read(file)
cat = read_events('test.xml', format='QUAKEML')

for ray in cat[0].preferred_origin().rays:

    nodes = np.copy(ray.nodes)

    new_ray = base.correct_ray(nodes)
    plt.clf()
    plt.plot(new_ray[:, 0], new_ray[:, 1], label='corrected')
    plt.plot([ray.nodes[0, 0], ray.nodes[-1, 0]], [ray.nodes[0, 1], ray.nodes[-1, 1]],
             label='optimal')
    plt.plot(ray.nodes[:, 0], ray.nodes[:, 1], label='original')
    plt.legend()
    plt.show()

    input('press enter')
