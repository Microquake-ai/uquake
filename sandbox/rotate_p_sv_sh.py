from uquake.core.data_exchange import ZarrHandler
from uquake.core import read_events
from uquake.grid import base
from importlib import reload
reload(base)
import matplotlib.pyplot as plt
import numpy as np
from uquake.grid import read_grid
from pathlib import Path
from uquake.core.event import RayEnsemble

file = '/mnt/HDD_5TB_01/Cozamin/principal_events/zarr/Cozamin231115123813005.zarr'
gridpath = Path('/home/jpmercier/Repositories/museis.ai/'
                '2023-0002-Cozamin-Large-Event/Project/COZAMIN/CZM/times/')

cat = read_events('test.xml', format='QUAKEML')
mde = ZarrHandler.read(file)
mde.catalog = cat
mde.p_sv_sh_stream_from_hodogram()
st = mde.stream.detrend('demean').detrend('linear').copy()
st2 = st.copy()
inventory = mde.inventory

rays = RayEnsemble([])
for instrument in inventory.instruments:
    if len(instrument.channels) == 3:
        rays.append(cat[0].preferred_origin().rays.select(station=instrument.station,
                                                          location=instrument.location,
                                                          phase='P')[0])

rays.plot_distribution()
input('press enter to continue')
# rays = cat[0].preferred_origin().rays

# st_rotated = st.rotate_from_hodogram(cat, inventory)
st_rotated = st.copy().rotate_p_sv_sh(cat[0].preferred_origin().rays, inventory)

plt.figure(figsize=(20, 20))
# for instrument in inventory.instruments:
#     plt.clf()
#     incidence_p = cat[0].preferred_origin().rays.select(
#         station=instrument.station_code)[0].incidence_p
#     for channel in inventory.select(station=instrument.station.code)[0][0]:
#         print(np.dot(channel.orientation_vector, incidence_p))
#         for tr in st_rotated.select(station=instrument.station_code, channel=channel.code):
#             plt.plot(tr.data, label=f'{instrument.code}.{channel.code}')
#     plt.legend()
#     plt.show()


for instrument in inventory.instruments:
    if len(instrument.channels) == 3:
        # plt.clf()
        st_rotated_tmp = st_rotated.select(instrument=instrument.code)
        # st_rotated_tmp.plot(event=cat[0])
        st_tmp = st2.select(instrument=instrument.code)
        st_tmp.plot(event=cat[0])
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(st_tmp[i].data)
            plt.plot(st_rotated_tmp[i].data)
            plt.show()

st_copy = st.copy()
for arrival in cat[0].preferred_origin().arrivals:
    if arrival.phase == 'S':
        continue
    st2 = st_copy.select(network=arrival.pick.waveform_id.network_code,
                         station=arrival.pick.waveform_id.station_code,
                         location=arrival.pick.waveform_id.location_code).copy()

    if len(st2) != 3:
        continue
    st2 = st2.trim(starttime=arrival.pick.time + 0.002,
                   endtime=arrival.pick.time + 0.02)

    plt.clf()
    plt.plot(st2[0].data, st2[1].data)
    plt.show()


# gd = read_grid('time.pickle', format='pickle')

# ray = gd.ray_tracing(cat[0].preferred_origin().loc)
plt.close('all')
for grid_file in gridpath.glob('*.pickle'):
    gd = read_grid(grid_file, format='pickle')
    ray = gd.ray_tracer(cat[0].preferred_origin().loc, )



    plt.figure(1)
    plt.clf()
    i = gd.transform_to_grid(gd.seed.loc).astype(int)
    plt.imshow(gd.data[:, :, int(i[-1])])
    plt.plot(i[1], i[0], 'ro')
    # plt.show()

    ray = gd.ray_tracer(cat[0].preferred_origin().loc, )

    nodes = np.copy(ray.nodes)

    # new_ray = base.correct_ray(nodes)
    # for i in range(1, len(nodes)):
    #     plt.plot(nodes[i, 0], nodes[i, 1], 'b.')
    #     plt.show()
    #     input('grapout')

    nodes_grid = []
    for node in ray.nodes:
        nodes_grid.append(gd.transform_to_grid(node))

    nodes_grid = np.array(nodes_grid)

    plt.plot(nodes_grid[:, 1], nodes_grid[:, 0], 'b.')
    # plt.plot(ray.nodes[:, 1], ray.nodes[:, 0], 'r.', label='corrected')
    # plt.plot([ray.nodes[0, 0], ray.nodes[-1, 0]], [ray.nodes[0, 1], ray.nodes[-1, 1]],
    #          'b.', label='optimal')
    # plt.plot(ray.nodes[:, 0], ray.nodes[:, 1], 'r', label='original')
    # plt.legend()
    plt.show()

    print(ray)
    input('press enter')
