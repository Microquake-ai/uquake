from uquake.core.data_exchange import ZarrHandler
from uquake.core import read_events
from uquake.grid import base
from importlib import reload
reload(base)
import matplotlib.pyplot as plt
import numpy as np
from uquake.grid import read_grid
from pathlib import Path
from uquake.core.inventory import InstrumentSensitivity
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

st_rotated = st.copy().rotate_p_sv_sh(cat[0].preferred_origin().rays, inventory)

rays = RayEnsemble([])
for instrument in inventory.instruments:
    if len(instrument.channels) == 3:
        for ray in cat[0].preferred_origin().rays.select(
                station=instrument.station_code, location=instrument.location_code,
                phase='P'):
            rays.append(ray)

rays.plot_distribution()

# cat[0].preferred_origin().rays.plot_distribution()
input('press enter to continue')

for network in inventory:
    for station in network:
        for channel in station:
            if '1' in channel.code:
                channel.code = channel.code.replace('1', '_P')
            elif '2' in channel.code:
                channel.code = channel.code.replace('2', '_SV')
            elif '3' in channel.code:
                channel.code = channel.code.replace('3', '_SH')

            channel.response.instrument_sensitivity = InstrumentSensitivity(
                value=1, frequency=1, input_units='M/S', output_units='M/S')

plt.figure(figsize=(20, 20))
for instrument in inventory.instruments:
    plt.clf()
    for phase in ['P', 'S']:
        for arrival in cat[0].preferred_origin().arrivals:
            if arrival.pick.waveform_id.station_code == instrument.station_code:
                if arrival.pick.waveform_id.location_code == instrument.location_code:
                    if arrival.phase == phase:
                        break
        pick_time = arrival.pick.time
        st = st_rotated.select(instrument=instrument.code).copy()
        st.remove_response(inventory=inventory, pre_filt=[0.5, 1, 50, 100],
                           output='DISP')
        brune_pulse = st[0].create_brune_pulse(10, pick_time)

        brune_pulse /= np.std(brune_pulse)

        for i, tr in enumerate(st):
            tr.data /= np.std(tr.data)
            plt.subplot(3, 1, i + 1)
            plt.plot(tr.data)
            plt.plot(brune_pulse)
            plt.axvline(tr.time_to_index(pick_time), ls='--', color='k')
            plt.show()
            print(np.sign((tr.data * brune_pulse).sum()))
        input('press enter to continue')
