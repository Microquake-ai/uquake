from uquake.core.data_exchange import ZarrHandler
from uquake.core import read_events
from uquake.grid import base
from importlib import reload
reload(base)
import matplotlib.pyplot as plt
import numpy as np
from uquake.grid import read_grid
from pathlib import Path

file = '/mnt/HDD_5TB_01/Cozamin/principal_events/zarr/Cozamin231115123813005.zarr'
gridpath = Path('/home/jpmercier/Repositories/museis.ai/2023-0002-Cozamin-Large-Event/Project/COZAMIN/CZM/times/')

mde = ZarrHandler.read(file)
cat = read_events('test.xml', format='QUAKEML')

# gd = read_grid('time.pickle', format='pickle')

# ray = gd.ray_tracing(cat[0].preferred_origin().loc)

for grid_file in gridpath.glob('*.pickle'):
    gd = read_grid(grid_file, format='pickle')
    ray = gd.ray_tracer(cat[0].preferred_origin().loc)

    nodes = np.copy(ray.nodes)

    # new_ray = base.correct_ray(nodes)
    plt.clf()
    # for i in range(1, len(nodes)):
    #     plt.plot(nodes[i, 0], nodes[i, 1], 'b.')
    #     plt.show()
    #     input('grapout')
    plt.plot(ray.nodes[:, 0], ray.nodes[:, 1], 'r.', label='corrected')
    plt.plot([ray.nodes[0, 0], ray.nodes[-1, 0]], [ray.nodes[0, 1], ray.nodes[-1, 1]],
             'b.', label='optimal')
    # plt.plot(ray.nodes[:, 0], ray.nodes[:, 1], 'r', label='original')
    plt.legend()
    plt.show()

    print(ray)
    input('press enter')
