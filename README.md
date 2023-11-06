# README #

# uquake

The &mu;Quake pronounced muQuake (microQuake) is a Python library for processing and analyzing microseismic data. It provides a suite of tools for handling microseismic event detection, location, and characterization.

## Installation

To install `uquake`, you can use `pip`:

```bash
pip install uquake
```

Alternatively, if you are using poetry, you can add uquake to your project's dependencies by running:

```bash
Copy code
poetry add uquake
Usage
python
Copy code
import uquake
```

# Example usage

## Objects

```python
from uquake.core import read_events, read, read_inventory

# read a catalog
cat = read_events('catalog.xml')
inv = read_inventory('inventory.xml')
st = read('waveform.mseed')

# access in cartesian coordinates

x_event = cat[0].origins[0].x
y_event = cat[0].origins[0].y
z_event = cat[0].origins[0].z

x_instrument = inv[0][0].instrument.x
y_instrument = inv[0][0].instrument.y
x_instrument = inv[0][0].instrument.z

# packaging of the file in ASDF format

from uquake.core.data_exchange import MicroseismicDataExchange, read_mde
mde = MicroseismicDataExchange(st, cat, inv)

mde.write('microseismic_data.asdf')
mde2 = MicroseismicDataExchange.read('microseismic_data.asdf')

cat = mde2.catalog
st  = mde2.stream
inv = mde2.inventory

```

## Grids

```python
from uquake.grid.extended import VelocityGrid3D, VelocityGridEnsemble, SeedEnsemble
from uquake.core import read_inventory
import numpy as np

# create a grid
p_velocity_data = np.random.rand(100, 100, 100) * 1000 + 5000
s_velocity_data = np.random.rand(100, 100, 100) * 600 + 3000
smoothing_kernel = 10  # standard deviation of the gaussian smoothing kernel in grid space units
p_velocity = VelocityGrid3D(p_velocity_data, spacing=10, origin=(0, 0, 0)).smooth(10)
s_velocity = VelocityGrid3D(s_velocity_data, spacing=10, origin=(0, 0, 0)).smooth(10)

# create an ensemble of grids
ensemble = VelocityGridEnsemble(p_velocity, s_velocity)

# save the ensemble
ensemble.write('velocity_grid.h5')

# creating the travel-time grid
seed_ensemble = SeedEnsemble.generate_random_seeds_in_grid(p_velocity)

tt_grid_p = p_velocity.compute_travel_time_grid(seed_ensemble)
tt_grid_s = s_velocity.compute_travel_time_grid(seed_ensemble)

tt_grid = tt_grid_p + tt_grid_s
```

Documentation
The full documentation for uquake is available [here](https://microquake-ai.github.com/uquake/docs).


Features
- [x] Read and write microseismic data in ASDF format
- [x] Augment the Obspy library to handle cartesian coordinates
- [x] Support for 3D grid generation and manipulation

License
uquake is released under the AGPL License.

Support
If you have any questions or need support, please file an issue in the GitHub issue tracker.


