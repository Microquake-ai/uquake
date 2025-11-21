# Copyright (C) 2023, Jean-Philippe Mercier
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: <filename>
#  Purpose: <purpose>
#   Author: <author>
#    Email: <email>
#
# Copyright (C) <copyright>
# --------------------------------------------------------------------
"""


:copyright:
    <copyright>
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import matplotlib
import numpy as np
from .base import Grid
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from loguru import logger
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional, Sequence, Tuple
from .base import ray_tracer
import shutil
from uquake.grid import read_grid
from scipy import sparse
from scipy.interpolate import interp1d
from enum import Enum
from typing import List, Literal
from uquake.core.event import WaveformStreamID
from uquake.core.coordinates import Coordinates, CoordinateSystem
from uquake.core.inventory import Inventory
from uquake.synthetic.inventory import generate_unique_instrument_code
from uquake.core.event import ResourceIdentifier
from .base import __default_grid_label__
from typing import Set, Tuple, Union
from ttcrpy import rgrid
from scipy.signal import fftconvolve
from disba import PhaseDispersion, PhaseSensitivity, GroupDispersion, GroupSensitivity
from evtk import hl
from scipy.signal import fftconvolve
import time
import warnings
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

try:  # optional fast marching solver
    import skfmm  # type: ignore

    _SKFMM_AVAILABLE = True
except ImportError:  # pragma: no cover - availability depends on environment
    skfmm = None  # type: ignore
    _SKFMM_AVAILABLE = False

try:  # optional Estuary fast-marching bindings
    from estuaire.core.data import EKImageData  # type: ignore
    from estuaire.core.frechet import compute_frechet as _eikonal_compute_frechet  # type: ignore
except (ImportError, SyntaxError):  # pragma: no cover - optional dependency
    EKImageData = None  # type: ignore
    _eikonal_compute_frechet = None



__cpu_count__ = cpu_count()

valid_phases = ('P', 'S')

# In many cases, where Z is ignored, North-Up-Down and North-East-Up can be treated as the same
NORTH_EAST_SYSTEMS = {CoordinateSystem.NED, CoordinateSystem.NEU}


def _require_skfmm(context: str) -> None:
    """Raise an informative error when scikit-fmm is required but unavailable."""

    if not _SKFMM_AVAILABLE:
        raise ImportError(
            f"scikit-fmm is required to {context}. Install it with 'pip install scikit-fmm' "
            "or rerun using method='ttcrpy'."
        )


def _deduplicate_points(coords: np.ndarray, decimals: int = 8):
    """Deduplicate coordinate rows while preserving their first occurrence order."""

    coords = np.asarray(coords, dtype=float)
    n_points = coords.shape[0]
    inverse = np.empty(n_points, dtype=int)
    unique_points = []
    representative_indices = []
    mapping = {}

    for idx, row in enumerate(coords):
        key = tuple(np.round(row, decimals=decimals))
        if key not in mapping:
            mapping[key] = len(unique_points)
            unique_points.append(row)
            representative_indices.append(idx)
        inverse[idx] = mapping[key]

    unique_points = np.array(unique_points, dtype=float)
    return unique_points, inverse, representative_indices


class Phases(Enum):
    P = 'P'
    S = 'S'
    RAYLEIGH = 'RAYLEIGH'
    LOVE = 'LOVE'


class GridTypes(Enum):
    VELOCITY = 'VELOCITY'
    VELOCITY_METERS = 'VELOCITY_METERS'
    VELOCITY_KILOMETERS = 'VELOCITY_KILOMETERS'
    SLOWNESS = 'SLOWNESS'
    VEL2 = 'VEL2'
    SLOW2 = 'SLOW2'
    SLOW2_METERS = 'SLOW2_METERS'
    SLOW_LEN = 'SLOW_LEN'
    STACK = 'STACK'
    TIME = 'TIME'
    TIME2D = 'TIME2D'
    PROB_DENSITY = 'PROB_DENSITY'
    MISFIT = 'MISFIT'
    ANGLE = 'ANGLE'
    ANGLE2D = 'ANGLE2D'
    AZIMUTH = 'AZIMUTH'
    TAKEOFF = 'TAKEOFF'
    DENSITY = 'DENSITY_KG_METERS3'

    def __str__(self):
        return self.value


class VelocityType(Enum):
    GROUP = 'GROUP'
    PHASE = 'PHASE'

    def __str__(self):
        return self.value


valid_grid_types = (
    'VELOCITY',
    'VELOCITY_METERS',
    'VELOCITY_KILOMETERS'
    'SLOWNESS',
    'VEL2',
    'SLOW2',
    'SLOW2_METERS',
    'SLOW_LEN',
    'STACK',
    'TIME',
    'TIME2D',
    'PROB_DENSITY',
    'MISFIT',
    'ANGLE',
    'ANGLE2D',
    'DENSITY'
)


class FloatTypes(Enum):
    FLOAT = 'float32'
    DOUBLE = 'float64'

    def __str__(self):
        return self.value


valid_float_types = {
    # NLL_type: numpy_type
    'FLOAT': 'float32',
    'DOUBLE': 'float64'
}


class GridUnits(Enum):
    METER = 'METER'
    FEET = 'FEET'
    KILOMETER = 'KILOMETER'
    DEGREES = 'DEGREES'
    RADIAN = 'RADIAN'

    def __str__(self):
        return self.value


valid_grid_units = (
    'METER',
    'KILOMETER',
)


__velocity_grid_location__ = Path('model')
__time_grid_location__ = Path('time')

__default_grid_units__ = GridUnits.METER
__default_float_type__ = FloatTypes.FLOAT


class DisbaAlgorithm(Enum):
    dunkin = 'dunkin'
    phase_delta = 'phase-delta'


class DisbaParam:
    def __init__(self, algorithm: DisbaAlgorithm = DisbaAlgorithm.dunkin,
                 dc: float = 0.005, dp: float = 0.025):
        """
        :param algorithm
        """
        self.__algorithm__ = algorithm
        self.__dc__ = dc
        self.__dp__ = dp

    @property
    def algorithm(self):
        return self.__algorithm__.value

    @property
    def dc(self):
        return self.__dc__

    @property
    def dp(self):
        return self.__dp__


class Seed:
    def __init__(self, station_code, location_code, coordinates: Coordinates,
                 elevation: float = 0, short_label: Optional[str] = None,
                 units=__default_grid_units__):
        """
        Contains a location
        :param station_code: station code
        :param instrument_code: location code
        :param coordinates:
        """

        self.station = station_code
        self.location = location_code
        self.coordinates = coordinates
        self.elevation = elevation
        self.short_label = short_label
        self.units = units

    def __repr__(self):
        return f'label (station.location): {self.station}.{self.location}\n' \
               f'                       x: {self.coordinates.x:0.2f}\n' \
               f'                       y: {self.coordinates.y:0.2f}\n' \
               f'                       z: {self.coordinates.z:0.2f}\n' \
               f'               elevation: {self.elevation:0.2f}\n' \
               f'       coordinate_system: {self.coordinates.coordinate_system}\n'

    def __str__(self):
        return self.__repr__()

    @property
    def x(self):
        return self.coordinates.x

    @property
    def y(self):
        return self.coordinates.y

    @property
    def z(self):
        return self.coordinates.z

    @property
    def loc(self):
        return self.coordinates.loc

    @property
    def label(self):
        return f'{self.station}.{self.location}'

    @property
    def instrument_code(self):
        return self.label

    @property
    def T(self):
        return np.array(self.coordinates.loc).T

    @classmethod
    def random_in_grid(cls, grid: Grid, n_seeds: int = 1) -> 'List[Seed]':
        """
        Generate a random point within the grid boundary
        :param grid: a grid object
        :type grid: uquake.grid.base.Grid or an object inheriting from Grid
        :param n_seeds: number of locations to generate
        :type n_seeds: int
        :return: a list of Location objects

        :example:
        >> from uquake.grid.base import Grid
        >> from uquake.grid.nlloc import Location
        >> grid_dimensions = [10, 10, 10]
        >> grid_spacing = [1, 1, 1]
        >> grid_origin = [0, 0, 0]
        >> grid = Grid(grid_dimensions, grid_spacing, grid_origin, value=1)
        >> locations = Location.random_in_grid(grid, n_location=10)
        """

        codes = generate_unique_instrument_code(n_codes=n_seeds)
        seeds = []

        for i, (point, code) in enumerate(zip(grid.generate_random_points_in_grid(
                n_points=n_seeds), codes)):
            station_code = code.split('_')[0]
            location_code = code.split('_')[1]
            coordinates = Coordinates(point[0], point[1], point[2])
            seeds.append(cls(station_code, location_code, coordinates))

        return seeds


class SeedEnsemble:

    def __init__(self, seeds: List[Seed] = None, units: GridUnits = GridUnits.METER):
        """
        specifies a series of source location from an inventory object
        :param seeds: a list of locations containing at least the location,
        and location label
        :param units: units of measurement used to express x, y, and z
        :type units: str must be a valid unit of measurement as defined by the
        GridUnits class
        :type seeds: list of dictionary

        :Example:

        >>> seed = Seed('station_code', 'location_code', Coordinates(0, 0, 0))
        >>> seeds = SeedEnsemble([seed])

        """
        self.units = units
        self.seeds = seeds if seeds is not None else []

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, item):
        return self.seeds[item]

    def __repr__(self):
        return f'{self.units}\n' + '\n'.join(self.seeds)

    def __iter__(self):
        return iter(self.seeds)

    @classmethod
    def from_inventory(cls, inventory: Inventory, units: GridUnits = GridUnits.METER) \
            -> 'SeedEnsemble':
        """
        create from an inventory object
        :param inventory: an inventory object
        :type inventory: uquake.core.inventory.Inventory
        :param units: units of measurement used to express x, y, and z
        :type units: str must be a valid unit of measurement as defined by the
        GridUnits class
        :rparam: a Seeds object
        """

        srces = []
        for instrument in inventory.instruments:
            srce = Seed(instrument.station_code, instrument.location_code,
                        instrument.coordinates)
            srces.append(srce)

        return cls(srces)

    @classmethod
    def from_json(cls, json):
        pass

    def add(self, seed: Seed):
        """
        Add a single location to the source list
        :param location:
        :type location: Seed
        """

        self.seeds.append(seed)

    def append(self, seed: Seed):
        """
        Append a list of locations to the source list
        :param location:
        :type location: Seed
        """

        self.seeds.append(seed)

    @classmethod
    def generate_random_seeds_in_grid(cls, grid, n_seeds=1):
        """
        generate n_seeds random seeds inside the grid provided. This function
        is mainly used for testing purposes
        :param grid: a grid
        :type grid: uquake.grid.base.Grid or an object inheriting from Grid
        :param n_seeds: number of seeds to generate
        :return: a list of seeds

        >>> from uquake.grid.base import Grid
        >>> from uquake.grid.extended import SeedEnsemble
        >>> grid_dimensions = [10, 10, 10]
        >>> grid_spacing = [1, 1, 1]
        >>> grid_origin = [0, 0, 0]
        >>> grid = Grid(grid_dimensions, grid_spacing, grid_origin, value=1)
        >>> seeds = SeedEnsemble.generate_random_seeds_in_grid(grid, n_seeds=10)
        """

        seeds = []
        for seed in Seed.random_in_grid(grid, n_seeds=n_seeds):
            # import ipdb
            # ipdb.set_trace()
            seeds.append(seed)

        return cls(seeds)

    def __repr__(self):
        line = ""

        divider = 1000 if self.units == GridUnits.KILOMETER else 1

        return f'{self.units}\n' + '\n'.join([f'{seed}'
                                              for seed in self.seeds])

    @property
    def nlloc(self):
        line = ""

        divider = 1000 if self.units == GridUnits.KILOMETER else 1

        for seed in self.seeds:
            # test if location name is shorter than 6 characters

            line += f'GTSRCE {seed.label} XYZ ' \
                    f'{seed.x / divider:>15.6f} ' \
                    f'{seed.y / divider:>15.6f} ' \
                    f'{seed.z / divider:>15.6f} ' \
                    f'{seed.elevation:0.2f}\n'

        return line

    @property
    def locs(self):
        locations = []
        for seed in self.seeds:
            locations.append([seed.x, seed.y, seed.z])
        return np.array(locations)

    @property
    def labels(self):
        seed_labels = []
        for seed in self.seeds:
            seed_labels.append(seed.label)

        return np.array(seed_labels)


class TypedGrid(Grid):
    """
    base 3D rectilinear grid object
    """

    def __init__(self, data_or_dims, origin, spacing, phase: Phases = None,
                 value=0, grid_type=GridTypes.VELOCITY_METERS,
                 grid_units=__default_grid_units__,
                 float_type="FLOAT",
                 grid_id: ResourceIdentifier = ResourceIdentifier(),
                 label: str = __default_grid_label__,
                 coordinate_system: CoordinateSystem = CoordinateSystem.NED):
        """
        :param data_or_dims: data or data dimensions. If dimensions are
        provided a homogeneous grid is created with value=value
        :param origin: origin of the grid
        :type origin: list
        :param spacing: the spacing between grid nodes
        :type spacing: list
        :param phase: Phase
        :type phase: Phases
        :param value: a value to fill the grid with if data_or_dims is a list
        :type value: float
        :param grid_type: grid type
        :type grid_type: GridTypes
        :param grid_units: grid units
        :type grid_units: GridUnits
        :param float_type:  the float type either 'FLOAT' or 'DOUBLE'
        :type float_type: FloatTypes
        :param grid_id: the grid id
        :type grid_id: str
        :param label: label of the grid
        :type label: str
        :param coordinate_system: coordinate system either NED or ENU
        :type coordinate_system: CoordinateSystem
        """

        super().__init__(data_or_dims, spacing=spacing, origin=origin,
                         value=value, resource_id=grid_id, label=label,
                         coordinate_system=coordinate_system)

        self.phase = phase
        self.grid_type = grid_type
        self.grid_units = grid_units
        self.float_type = float_type

    @property
    def grid_id(self):
        """Unique identifier attached to this grid instance.

        :returns: The resource identifier propagated from the base grid.
        :rtype: ResourceIdentifier
        """
        return self.resource_id

    def mv(self, base_name, origin, destination):
        """
        move a NLLoc grid with a certain base_name from an origin to a
        destination
        :param NLLocGridObject:
        :type NLLocGridObject: uquake.grid.extended.TypedGrid
        :param base_name:
        :type base_name: str
        :param origin:
        :type origin: str
        :param destination:
        :type destination: str
        :return:
        """

        self.write(base_name, destination)
        for ext in self.extensions:
            shutil.move(f'{origin}/{base_name}.{ext}',
                        f'{destination}/{base_name}.{ext}')

    @classmethod
    def from_ods(cls, origin, dimensions, spacing, phase, val=0):
        grid = super().from_ods(origin, dimensions, spacing, val=val)
        return cls(grid.data, origin, spacing, phase)

    @classmethod
    def from_ocs(cls, origin, corner, spacing, phase, val=0):
        grid = super().from_ocs(origin, corner, spacing, val=val)
        return cls(grid.data, grid.origin, grid.spacing, phase)

    @classmethod
    def from_ocd(cls, origin, corner, dimensions, phase, val=0):
        grid = super().from_ocd(origin, corner, dimensions, val=val)
        return cls(grid.data, grid.origin, grid.spacing, phase)

    @property
    def model_id(self):
        return self.resource_id

    def is_like(self, other):
        if self.dims != other.dims:
            return False
        if list(self.origin) != list(other.origin):
            return False
        if list(self.spacing) != list(other.spacing):
            return False
        if self.grid_units != other.grid_units:
            return False
        if self.label != other.label:
            return False
        return True

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return all([self.is_like(other), self.data == other.data])
        else:
            return False

    def plot_slice(self, axis: int, slice_position: float, grid_space: bool = False,
                   field_name=None, mask: Optional[dict] = None,**kwargs):
        fig, ax = plt.subplots()
        match axis:
            case 2:
                if not grid_space:
                    tmp = self.transform_to((self.origin[0], self.origin[1],
                                             slice_position))
                    k = int(tmp[2])
                else:
                    k = int(slice_position)
                if not self.in_grid([0, 0, k], grid_space=True):
                    raise IndexError(f'The slice plane {slice_position} falls'
                                     f' outside the grid.')

                if 'cmap' not in kwargs:
                    kwargs.setdefault('cmap', 'seismic')

                im = ax.imshow(self.data[:, :, k].T, origin='lower',
                               extent=(self.origin[0],
                                       self.corner[0],
                                       self.origin[1],
                                       self.corner[1]),
                               **kwargs)

                if mask is not None:
                    positive_mask = super().masked_region_xy(**mask,
                                                             ax=ax)
                    grid_data = np.where(positive_mask, self.data[:, :, k], np.nan)
                else:
                    grid_data = self.data[:, :, k]

                vmin = np.nanpercentile(grid_data, 1)
                vmax = np.nanpercentile(grid_data, 99)
                im.set_clim(vmin, vmax)

                if self.grid_units == GridUnits.METER:
                    ax.set_xlabel("X (m)")
                    ax.set_ylabel("Y (m)")
                if self.grid_units == GridUnits.KILOMETER:
                    ax.set_xlabel("X (km)")
                    ax.set_ylabel("Y (km)")
            case 1:
                if not grid_space:
                    tmp = self.transform_to((self.origin[0], slice_position,
                                             self.origin[2]))
                    j = int(tmp[1])
                else:
                    j = int(slice_position)
                if not self.in_grid([0, j, 0], grid_space=True):
                    raise IndexError(f'The slice plane {slice_position} falls '
                                     f'outside the grid.')
                im = ax.imshow(self.data[:, j, :].T, extent=(self.origin[0],
                                                             self.corner[0],
                                                             self.corner[2],
                                                             self.origin[2],),
                               cmap="seismic",
                               **kwargs)
                if self.grid_units == GridUnits.METER:
                    ax.set_xlabel("X (m)")
                    ax.set_ylabel("Z (m)")
                if self.grid_units == GridUnits.KILOMETER:
                    ax.set_xlabel("X (km)")
                    ax.set_ylabel("Z (km)")
                ax.xaxis.set_label_position('top')
                ax.xaxis.tick_top()
                if mask is not None:
                    logger.warning("Vertical slice masking functionality is "
                                   "currently unavailable")
            case 0:
                if not grid_space:
                    tmp = self.transform_to((slice_position, self.origin[1],
                                             self.origin[2]))
                    i = int(tmp[0])
                else:
                    i = int(slice_position)
                if not self.in_grid([i, 0, 0], grid_space=True):
                    raise IndexError(f'The slice plane {slice_position} falls '
                                     f'outside the grid.')

                if mask is not None:
                    logger.warning("Vertical slice masking functionality is "
                                   "currently unavailable")

                im = ax.imshow(self.data[i, :, :].T, extent=(self.origin[1],
                                                             self.corner[1],
                                                             self.corner[2],
                                                             self.origin[2]),
                               cmap="seismic",
                               **kwargs)
                if self.grid_units == GridUnits.METER:
                    ax.set_xlabel("Y (m)")
                    ax.set_ylabel("Z (m)")
                if self.grid_units == GridUnits.KILOMETER:
                    ax.set_xlabel("Y (km)")
                    ax.set_ylabel("Z (km)")
                ax.xaxis.set_label_position('top')
                ax.xaxis.tick_top()

        cb = fig.colorbar(im, ax=ax, orientation='vertical')
        cb.update_normal(im)
        if field_name is None:
            cb.set_label(self.grid_type.value, rotation=270, labelpad=15)
        else:
            cb.set_label(field_name, rotation=270, labelpad=15)
        return fig, ax

class TypedGridIrregular(Grid):
    pass

class Direction(Enum):
    UP = 'UP'
    DOWN = 'DOWN'


class ModelLayer:
    """
    1D model varying in Z
    """

    def __init__(self, z_top, value_top):
        """
        :param z_top: Top of the layer z coordinates
        :param value_top: Value at the top of the layer
        """
        self.z_top = z_top
        self.value_top = value_top

    def __repr__(self):
        return f'top - {self.z_top:5.0f} | value - {self.value_top:5.0f}\n'


class LayeredVelocityModel(object):

    def __init__(self, network_code: str, velocity_model_layers: List = None,
                 phase: Phases = Phases.P,
                 grid_units: GridUnits = __default_grid_units__,
                 float_type=__default_float_type__,
                 gradient=False, grid_id: ResourceIdentifier = ResourceIdentifier(),
                 coordinate_system: CoordinateSystem = CoordinateSystem.NED,
                 label=__default_grid_label__):
        """
        Initialize
        :param network_code: network code
        :type network_code: str
        :param velocity_model_layers: a list of VelocityModelLayer
        :type velocity_model_layers: list
        :param phase: Phase either Phases.P or Phases.S
        :type phase: Phases default = Phases.P
        :param grid_units: units of measurement used to express x, y, and z
        :type grid_units: GridUnits
        :param float_type: float type either 'FLOAT' or 'DOUBLE'
        :type float_type: FloatTypes default = 'FLOAT'
        :param gradient: whether the model is a gradient model or not
        (no control layer by layer)
        :type gradient: bool default = False
        :param grid_id: the grid id
        :type grid_id: uquake.core.event.ResourceIdentifier
        :param coordinate_system: coordinate system either NED or ENU
        :type coordinate_system: CoordinateSystem
        :param label: grid label
        :type label: str
        """

        self.network_code = network_code

        if velocity_model_layers is None:
            self.velocity_model_layers = []

        if isinstance(phase, Phases):
            self.phase = phase
        elif isinstance(phase, str):
            self.phase = Phases(phase)
        else:
            raise TypeError('phase must be a Phases object')

        if isinstance(grid_units, GridUnits):
            self.grid_units = grid_units
        elif isinstance(grid_units, str):
            self.grid_units = GridUnits(grid_units)
        else:
            raise TypeError('grid_units must be a GridUnits object')

        if isinstance(float_type, FloatTypes):
            self.float_type = float_type
        elif isinstance(float_type, str):
            self.float_type = FloatTypes(float_type)
        else:
            raise TypeError('float_type must be a FloatTypes object')

        self.grid_type = GridTypes.VELOCITY_METERS

        self.grid_id = grid_id

        self.gradient = gradient
        self.coordinate_system = coordinate_system
        self.label = label

    def __repr__(self):
        output = ''
        for i, layer in enumerate(self.velocity_model_layers):
            output += f'layer {i + 1:4d} | {layer}'

        return output

    def add_layer(self, layer: ModelLayer):
        """
        Add a layer to the model. The layers must be added in sequence from the
        top to the bottom
        :param layer: a ModelLayer object
        """
        if not (type(layer) is ModelLayer):
            raise TypeError('layer must be a VelocityModelLayer object')

        if self.velocity_model_layers is None:
            self.velocity_model_layers = [layer]
        else:
            self.velocity_model_layers.append(layer)

    def to_1d_model(self, z_min, z_max, spacing):
        # sort the layers to ensure the layers are properly ordered
        z = []
        v = []
        for layer in self.velocity_model_layers:
            z.append(layer.z_top)
            v.append(layer.value_top)

        if np.max(z) < z_max:
            i_z_max = np.argmax(z)
            v_z_max = v[i_z_max]

            z.append(z_max)
            v.append(v_z_max)

        if np.min(z) > z_min:
            i_z_min = np.argmin(z)
            v_z_min = v[i_z_min]

            z.append(z_min)
            v.append(v_z_min)

        i_sort = np.argsort(z)

        z = np.array(z)
        v = np.array(v)

        z = z[i_sort]
        v = v[i_sort]

        z_interp = np.arange(z_min, z_max, spacing)
        kind = 'previous'
        if self.gradient:
            kind = 'linear'

        f_interp = interp1d(z, v, kind=kind)

        v_interp = f_interp(z_interp)

        return z_interp, v_interp

    def to_3d_grid(self, dims, origin, spacing):
        model_grid_3d = VelocityGrid3D.from_layered_model(self, dims, origin,
                                                          spacing, label=self.label)
        return model_grid_3d

    def plot(self, z_min, z_max, spacing, invert_z_axis=True, *args, **kwargs):
        """
        Plot the 1D velocity model
        :param z_min: lower limit of the model
        :param z_max: upper limit of the model
        :param spacing: plotting resolution in z
        :param invert_z_axis: whether the z axis is inverted or not
        (default = True)
        :return: matplotlib axis
        """

        z_interp, v_interp = self.to_1d_model(z_min, z_max, spacing)

        x_label = None
        if self.phase == 'P':
            x_label = 'P-wave velocity'
        elif self.phase == 'S':
            x_label = 's-wave velocity'

        if self.grid_units.value == 'METER':
            units = 'm'
        else:
            units = 'km'

        y_label = f'z [{units}]'
        ax = plt.axes()
        ax.plot(v_interp, z_interp, *args, **kwargs)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if invert_z_axis:
            ax.invert_yaxis()

        ax.set_aspect(2)

        plt.tight_layout()

        return ax


class DensityGrid3D(TypedGrid):

    def __init__(self, network_code, data_or_dims, origin, spacing, value=0,
                 float_type=__default_float_type__,
                 grid_id: ResourceIdentifier = ResourceIdentifier(),
                 grid_type=GridTypes.DENSITY,
                 grid_units=__default_grid_units__,
                 coordinate_system=CoordinateSystem.NED,
                 label: str = __default_grid_label__, **kwargs):

        """

        :param network_code:
        :param data_or_dims:
        :param origin:
        :param spacing:
        :param phase:
        :param value:
        :param float_type:
        :param grid_id:
        :param grid_type:
        :param grid_units:
        param coordinate_system:
        :param label:
        :param kwargs:
        """

        self.network_code = network_code

        if (type(spacing) is int) | (type(spacing) is float):
            spacing = [spacing, spacing, spacing]

        super().__init__(data_or_dims, origin, spacing, None,
                         value=value, grid_type=grid_type,
                         grid_units=grid_units, coordinate_system=coordinate_system,
                         float_type=float_type,
                         grid_id=grid_id, label=label)

    def plot_slice(self, axis: int, slice_position: float, grid_space: bool = False,
                   ** kwargs):
        field_name = 'Density (kg/$m^{3}$)'
        fig, ax = super().plot_slice(axis, slice_position, grid_space, field_name)
        return fig, ax

    def write(self, filename, format='VTK', **kwargs):
        field_name = None
        if format == 'VTK':
            field_name = f'density'

        super().write(filename, format=format, field_name=field_name, **kwargs)


class VelocityGrid3D(TypedGrid):

    def __init__(self, network_code, data_or_dims, origin, spacing,
                 phase: Phases = Phases.P, value=0, float_type=__default_float_type__,
                 grid_id: ResourceIdentifier = ResourceIdentifier(),
                 grid_type=GridTypes.VELOCITY_METERS,
                 grid_units=__default_grid_units__,
                 coordinate_system=CoordinateSystem.NED,
                 label: str = __default_grid_label__, **kwargs):

        """

        :param network_code:
        :param data_or_dims:
        :param origin:
        :param spacing:
        :param phase:
        :param value:
        :param float_type:
        :param grid_id:
        :param grid_type:
        :param grid_units:
        param coordinate_system:
        :param label:
        :param kwargs:
        """

        self.network_code = network_code

        if (type(spacing) is int) | (type(spacing) is float):
            spacing = [spacing, spacing, spacing]

        super().__init__(data_or_dims, origin, spacing, phase,
                         value=value, grid_type=grid_type,
                         grid_units=grid_units, coordinate_system=coordinate_system,
                         float_type=float_type,
                         grid_id=grid_id, label=label)

    def fill_checkerboard(self, anomaly_size, base_velocity, velocity_perturbation,
                          n_sigma):
        """
        Fill self.data (3D) with a checkerboard velocity model whose lateral anomaly
        size (x, y) doubles with depth layer, while vertical layer thickness (z)
        remains constant. The first anomaly center near the surface is positioned
        at depth anomaly_size[2] / 2.

        Parameters
        ----------
        anomaly_size : (float, float, float)
            Base physical size (same units as self.spacing) of a checker block in
            (x, y, z).
            Only x and y will double with each successive layer; z is the
            layer thickness.
        base_velocity : float
            Background/base velocity.
        velocity_perturbation : float
            Relative perturbation amplitude (e.g., 0.1 for ±10%).
        n_sigma : float
            Controls Gaussian kernel width per layer: sigma = block_size / n_sigma.
        """
        # Shapes & spacing
        nx, ny, nz = self.data.shape
        spacing = np.array(self.spacing, dtype=float)

        # Convert base physical anomaly size to integer cell steps
        base_steps = (np.array(anomaly_size, dtype=float) / spacing).astype(int)
        if np.any(base_steps < 1):
            raise ValueError("Base anomaly size too small relative to spacing.")

        # Layer thickness (in cells) stays constant
        step_z = int(base_steps[2])
        if step_z < 1:
            raise ValueError("Layer thickness (z) must be >= 1 cell.")

        # Number of depth layers that fit in the domain
        n_layers = int(np.ceil(nz / step_z))

        # The deepest layer has the largest lateral steps (doubling each layer)
        max_step_x = base_steps[0] * (2 ** (n_layers - 1))
        max_step_y = base_steps[1] * (2 ** (n_layers - 1))

        # Pad: one block on each side using the *maximum* lateral size and one layer in z
        pad = np.array([max_step_x, max_step_y, step_z], dtype=int)
        ext_shape = (nx + 2 * pad[0], ny + 2 * pad[1], nz + 2 * pad[2])

        accum = np.zeros(ext_shape, dtype=float)

        # Starting indices for x,y on original-domain boundaries
        start_x_global = pad[0]
        start_y_global = pad[1]

        # For z, shift by half a layer block so first centers at anomaly_size[2]/2
        half_block_z = int(round(step_z / 2))
        # Center index for the first layer (L = 0)
        first_layer_center_z = pad[2] + half_block_z

        # Build per-layer seed + convolution, then add to accum
        for L in range(n_layers):
            # Lateral steps double with depth
            step_x_L = base_steps[0] * (2 ** L)
            step_y_L = base_steps[1] * (2 ** L)

            # Defensive checks: allow large steps but ensure kernels will fit
            if step_x_L < 1 or step_y_L < 1:
                continue  # impossible given above checks, but be safe

            # Seed grid for this layer
            work_L = np.zeros(ext_shape, dtype=float)

            # z-center for this layer (single plane of seeds)
            z_center_L = first_layer_center_z + L * step_z
            if z_center_L < 0 or z_center_L >= ext_shape[2]:
                continue  # layer center outside extended domain

            # x,y centers for this layer
            x_centers = np.arange(start_x_global, ext_shape[0], step_x_L)
            y_centers = np.arange(start_y_global, ext_shape[1], step_y_L)

            if len(x_centers) == 0 or len(y_centers) == 0:
                continue  # nothing to seed

            # Checkerboard signs: alternate in x,y; flip every layer using +L
            xg, yg = np.meshgrid(np.arange(len(x_centers)),
                                 np.arange(len(y_centers)),
                                 indexing='ij')
            s = (xg + yg + L) % 2  # include L so adjacent layers alternate
            sign = np.where(s == 0, 1.0, -1.0)

            # Place impulses at centers with +/- signs
            for i, xi in enumerate(x_centers):
                for j, yj in enumerate(y_centers):
                    work_L[xi, yj, z_center_L] = sign[i, j]

            # Build a Gaussian kernel sized to this layer's block (step_x_L, step_y_L, step_z)
            # Kernel coordinates in physical units for proper sigma scaling
            kx = np.arange(step_x_L) * spacing[0]
            ky = np.arange(step_y_L) * spacing[1]
            kz = np.arange(step_z) * spacing[2]
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

            mu = np.array([step_x_L * spacing[0] / 2.0,
                           step_y_L * spacing[1] / 2.0,
                           step_z * spacing[2] / 2.0], dtype=float)
            sigma = np.array([step_x_L * spacing[0] / float(n_sigma),
                              step_y_L * spacing[1] / float(n_sigma),
                              step_z * spacing[2] / float(n_sigma)], dtype=float)

            # Avoid sigma=0
            sigma[sigma == 0] = 1e-12

            kernel_L = np.exp(-(((KX - mu[0]) ** 2) / (sigma[0] ** 2) +
                                ((KY - mu[1]) ** 2) / (sigma[1] ** 2) +
                                ((KZ - mu[2]) ** 2) / (sigma[2] ** 2)))
            kernel_L /= np.sum(np.abs(kernel_L))

            # Convolve this layer and add to accumulator
            conv_L = fftconvolve(work_L, kernel_L, mode="same")
            accum += conv_L

        # Normalize and crop back to original domain
        m = np.max(np.abs(accum))
        if m > 0:
            accum /= m

        data = accum[
               pad[0]: pad[0] + nx,
               pad[1]: pad[1] + ny,
               pad[2]: pad[2] + nz
               ]

        # Rescale to velocity
        smoothed_data = data * (velocity_perturbation * base_velocity) + base_velocity
        self.data = smoothed_data

    def checkerboard_var_size(self, anomaly_size: Union[Tuple, List],
                              base_velocity: float, velocity_perturbation: float,
                              n_sigma: float):
        """
        Checkerboard model without kernel interference:
        - For depth layer L, block size in (x, y, z) = base_size * (2**L).
        - Each voxel belongs to exactly one block (no overlap),
            so no destructive/constructive
          interference between kernels.
        - Within each block we apply a Gaussian profile centered in the block (x,y,z),
          then normalize per-layer so max |value| = 1 (hence peaks are ±1).
        - First anomaly center near surface at z ≈ anomaly_size[2]/2 (top layer).

        Parameters
        ----------
        anomaly_size : (float, float, float)
            Base physical size of a checker block (x, y, z) for the TOP layer.
        base_velocity : float
        velocity_perturbation : float   # e.g., 0.1 for ±10%
        n_sigma : float                  # sigma = block_size / n_sigma (per axis)
        """
        import numpy as np

        nx, ny, nz = self.data.shape
        dx, dy, dz = map(float, self.spacing)

        # Base steps (cells) for the top layer
        base_steps = (
                    np.array(anomaly_size, dtype=float) / np.array([dx, dy, dz])).astype(
            int)
        if np.any(base_steps < 1):
            raise ValueError("Base anomaly size too small relative to spacing.")
        sx0, sy0, sz0 = map(int, base_steps)

        # Build layer thicknesses in z (cells): doubles each layer
        layer_thicknesses = []
        remaining = nz
        L = 0
        while remaining > 0:
            tL = sz0 * (2 ** L)
            tL = max(1, int(tL))
            layer_thicknesses.append(tL)
            remaining -= tL
            L += 1
        n_layers = len(layer_thicknesses)

        # Precompute index grids for x,y once (2D), and reuse for each z-slab/layer
        ix, iy = np.indices((nx, ny))
        field = np.zeros((nx, ny, nz), dtype=float)

        z_start = 0
        for L in range(n_layers):
            # Block size (cells) for this layer: doubles in all directions
            sx = sx0 * (2 ** L)
            sy = sy0 * (2 ** L)
            sz = sz0 * (2 ** L)

            # z slab for this layer
            z0 = z_start
            z1 = min(nz, z_start + layer_thicknesses[L])
            if z0 >= z1:
                z_start += layer_thicknesses[L]
                continue

            # Choose offsets so first block centers are at ~ half a block from the boundary
            # (matches “first center near surface at anomaly_size[2]/2” for z,
            # and gives symmetry in x,y)
            off_x = sx // 2  # integer half-block
            off_y = sy // 2
            # z center of the slab (in index)
            zc_idx = z0 + max(1, sz) // 2

            # Block indices for each x,y cell (same across the whole slab)
            # Use floor division relative to offsets
            bx = (ix - off_x) // sx
            by = (iy - off_y) // sy

            # Checkerboard sign per (x,y) tile, flip by layer too
            sign_xy = ((bx + by + L) % 2 == 0).astype(float) * 2.0 - 1.0  # ∈ {+1, -1}

            # Local centers (in *index* coordinates) for each (x,y) cell’s block
            cx = off_x + bx * sx + sx / 2.0  # may be fractional if sx is even
            cy = off_y + by * sy + sy / 2.0

            # Distances from local block center, in *physical units*
            dx_phys = (ix - cx) * dx
            dy_phys = (iy - cy) * dy

            # Gaussian sigmas in physical units
            sig_x = max((sx * dx) / float(n_sigma), 1e-12)
            sig_y = max((sy * dy) / float(n_sigma), 1e-12)
            sig_z = max((sz * dz) / float(n_sigma), 1e-12)

            # Lateral Gaussian (2D), no overlap because each point belongs to one block
            Gxy = np.exp(
                -((dx_phys ** 2) / (sig_x ** 2) + (dy_phys ** 2) / (sig_y ** 2)))

            # For z within the slab, include vertical Gaussian around zc_idx (same center for slab)
            z_idx = np.arange(z0, z1)
            dz_phys = (z_idx - zc_idx) * dz
            Gz = np.exp(-(dz_phys ** 2) / (sig_z ** 2))  # shape (z1-z0,)

            # Combine to 3D via outer product: (nx,ny,1) * (1,1,nz_slab)
            layer_vals = (sign_xy * Gxy)[:, :, None] * Gz[None, None, :]

            # --- Per-layer peak normalization: make max|layer| == 1
            peak = np.max(np.abs(layer_vals))
            if peak and np.isfinite(peak):
                layer_vals /= peak

            # Accumulate into field
            field[:, :, z0:z1] = layer_vals

            z_start += layer_thicknesses[L]

        # Map to velocity
        self.data = field * (velocity_perturbation * base_velocity) + base_velocity

    @staticmethod
    def _rho_gardner_gcc(vp_km_s: float):
        """Compute density in g/cc from Vp (km/s) using Gardner (1974).

        Parameters
        ----------
        vp_km_s : float or ndarray
            P-wave velocity in km/s.

        Returns
        -------
        float or ndarray
            Density in g/cc.
        """
        return 1.74 * vp_km_s ** 0.25

    @staticmethod
    def _rho_brocher_gcc(vp_km_s: float):
        """Compute density in g/cc from Vp (km/s) using Brocher (2005).

        Parameters
        ----------
        vp_km_s : float or ndarray
            P-wave velocity in km/s.

        Returns
        -------
        float or ndarray
            Density in g/cc.
        """
        return (
                1.6612 * vp_km_s
                - 0.4721 * vp_km_s ** 2
                + 0.0671 * vp_km_s ** 3
                - 0.0043 * vp_km_s ** 4
                + 0.000106 * vp_km_s ** 5
        )

    def to_density(
            self,
            method: Literal["Gardner", "Brocher"],
            poisson_ratio: float = 0.25,
    ):
        """Convert this velocity grid to a density grid (g/cc).

        If the grid phase is S, Vp is estimated from Vs using the Poisson-ratio
        relation:
            Vp / Vs = sqrt(2 * (1 - ν) / (1 - 2ν)).
        The chosen empirical relation (Gardner or Brocher) then maps Vp (km/s)
        to density (g/cc).

        Parameters
        ----------
        method : {"Gardner", "Brocher"}
            Empirical Vp-to-density relation. Gardner (1974) is broadly used
            for clastics; Brocher (2005) often fits crystalline crust better.
        poisson_ratio : float, optional
            Poisson's ratio ν used only when phase == Phases.S to estimate Vp
            from Vs. Default is 0.25 (Poisson solid).

        Returns
        -------
        DensityGrid3D
            New density grid with density in g/cc.

        Raises
        ------
        ValueError
            If an unsupported method is requested.
        """
        vp_vs = np.sqrt((2.0 * (1.0 - poisson_ratio)) / (1.0 - 2.0 * poisson_ratio))

        if self.phase == Phases.S:
            vp = self.data * vp_vs
        else:
            vp = self.data

        # Ensure Vp is in km/s for the empirical relations
        if self.grid_type == GridTypes.VELOCITY_METERS:
            vp = vp / 1000.0

        if (method == "Gardner") or (method.lower() == "gardner"):
            rho_gcc = self._rho_gardner_gcc(vp)
        elif (method == "Brocher") or (method.lower() == "brocher"):
            rho_gcc = self._rho_brocher_gcc(vp)
        else:
            raise ValueError(
                "Unsupported method. Use 'Gardner' or 'Brocher'."
            )

        # If your DensityGrid3D supports explicit unit tags, set g/cc here.
        return DensityGrid3D(
            self.network_code,
            rho_gcc,
            self.origin,
            self.spacing,
            grid_units=self.grid_units,  # consider a GridUnits.G_PER_CC enum
        )

    def convert_to_vp(
            self,
            poisson_ratio: float = 0.25,
            new_grid: bool = True,
    ) -> Optional["VelocityGrid3D"]:
        """Return or update the grid converted to P-wave velocity (Vp).

        If this grid represents S-wave velocity (Vs), convert to Vp using
        the Poisson-ratio relation:
            Vp / Vs = sqrt(2 * (1 - ν) / (1 - 2ν)).
        If the grid already represents Vp, do nothing.

        Parameters
        ----------
        poisson_ratio : float, optional
            Poisson's ratio ν used to compute Vp from Vs. Default is 0.25.
        new_grid : bool, optional
            If True, return a copy with data converted to Vp and phase set to P.
            If False, convert in place and return None.

        Returns
        -------
        VelocityGrid3D or None
            Converted grid if `new_grid` is True; otherwise None.
        """
        if self.phase == Phases.P:
            logger.info("Already P-wave velocity; no conversion performed.")
            return self.copy() if new_grid else None

        if self.phase == Phases.S:
            vp_vs = np.sqrt((2.0 * (1.0 - poisson_ratio)) / (1.0 - 2.0 * poisson_ratio))
            vp = self.data * vp_vs

            if new_grid:
                out_grid = self.copy()
                out_grid.data = vp
                out_grid.phase = Phases.P
                return out_grid

            self.data = vp
            self.phase = Phases.P
            return None

    def to_rgrid(self, n_secondary: Union[int, Tuple[int, int, int]], threads: int = 1):

        xrnge = np.arange(
            self.origin[0], self.corner[0], self.spacing[0]).astype(np.float64)
        yrnge = np.arange(
            self.origin[1], self.corner[1], self.spacing[1]).astype(np.float64)
        zrnge = np.arange(
            self.origin[2], self.corner[2], self.spacing[2]).astype(np.float64)
        if isinstance(n_secondary, int):
            grid = rgrid.Grid3d(x=xrnge, y=yrnge, z=zrnge, cell_slowness=False,
                                method='SPM', nsnx=n_secondary, nsny=n_secondary,
                                nsnz=n_secondary, translate_grid=True,
                                n_threads=threads)
        if isinstance(n_secondary, Tuple):
            grid = rgrid.Grid3d(x=xrnge, y=yrnge, z=zrnge, cell_slowness=False,
                                method='SPM', nsnx=n_secondary[0], nsny=n_secondary[1],
                                nsnz=n_secondary[2], translate_grid=True,
                                n_threads=threads)

        grid.set_velocity(self.data)
        return grid

    @classmethod
    def from_inventory(cls, network_code: str, inventory: Inventory,
                       spacing: Union[float, Tuple[float, float, float]],
                       padding: Union[float, Tuple[float, float, float]] = 0.2,
                       **kwargs):
        """
        Create a grid from an inventory object.
        :param network_code: Network code
        :param inventory: Inventory object
        :param spacing: Grid spacing in grid units
        :param padding: Padding around the inventory in percent of the model span
        (default: 0.2 -> 20%)

        keyword arguments are additional arguments to pass to the class constructor or
        __init__ method.
        """

        # Get the instrument locations
        locations_x = [instrument.x for instrument in inventory.instruments]
        locations_y = [instrument.y for instrument in inventory.instruments]
        locations_z = [instrument.z for instrument in inventory.instruments]

        # Determine the span of the inventory
        min_coords = np.array(
            [np.min(locations_x), np.min(locations_y), np.min(locations_z)])
        max_coords = np.array(
            [np.max(locations_x), np.max(locations_y), np.max(locations_z)])
        inventory_span = max_coords - min_coords

        # Calculate padding in grid units
        if isinstance(padding, tuple):
            padding_x, padding_y, padding_z = padding
        else:
            padding_x = padding_y = padding_z = padding

        # Calculate the total padding to be added
        total_padding = inventory_span * np.array([padding_x, padding_y, padding_z])

        # Adjust the origin and corner with the padding
        padded_origin = min_coords - total_padding / 2
        padded_corner = max_coords + total_padding / 2

        # Calculate grid dimensions
        grid_dims = np.ceil((padded_corner - padded_origin) / np.array(spacing)).astype(
            int)

        # Create and return the grid object
        return cls(network_code, grid_dims, spacing=spacing, origin=padded_origin,
                   **kwargs)

    @staticmethod
    def get_base_name(network_code, phase):
        """
        return the base name given a network code and a phase
        :param network_code: Code of the network
        :type network_code: str
        :param phase: Phase, either P or S
        :type phase: Phases
        :return: the base name
        """
        return f'{network_code.upper()}.{phase}.mod'

    @classmethod
    def from_ocd(cls, origin, corner, dimensions, val=0):
        pass

    @classmethod
    def from_ocs(cls, origin, corner, spacing, val=0):
        pass

    @classmethod
    def from_ocd(cls, origin, dimensions, spacing, val=0):
        pass

    @classmethod
    def from_layered_model(cls, layered_model, dims, origin,
                           spacing, label=__default_grid_label__, **kwargs):
        """
        Generating a 3D grid model from
        :param layered_model: a LayeredVelocityModel object
        :param dims: dimensions of the grid
        :param origin: origin of the grid
        :param spacing: spacing of the grid
        :param label: label of the grid
        :param kwargs: additional arguments
        :return: a VelocityGrid3D object
        """

        z_min = origin[-1]
        z_max = z_min + spacing[-1] * dims[-1]

        z_interp, v_interp = layered_model.to_1d_model(z_min, z_max, spacing[2])

        data = np.zeros(dims)

        for i, v in enumerate(v_interp):
            data[:, :, i] = v_interp[i]

        return cls(layered_model.network_code, data, origin, spacing,
                   phase=layered_model.phase,
                   float_type=layered_model.float_type,
                   grid_id=layered_model.grid_id, label=label, **kwargs)

    def to_slow_lens(self):
        data = self.spacing[0] / self.data

        return TypedGrid(data, self.origin, self.spacing,
                         self.phase, grid_type='SLOW_LEN',
                         grid_units=self.grid_units,
                         float_type=self.float_type,
                         grid_id=self.grid_id, label=self.label)

    @classmethod
    def from_slow_len(cls, grid: TypedGrid, network_code: str):
        data = np.mean(grid.spacing) / grid.data
        return cls(network_code, data, grid.origin, grid.spacing,
                   phase=grid.phase, float_type=grid.float_type,
                   grid_id=grid.grid_id, label=grid.label)

    def to_time(self, seed: Seed, sub_grid_resolution=0.1,
                *args, **kwargs):
        """
        Eikonal solver based on scikit fast marching solver
        :param seed: numpy array location of the seed or origin of useis wave
         in model coordinates
        (usually location of a station or an event)
        :type seed: numpy.ndarray or list
        :param sub_grid_resolution: resolution of the grid around the seed.
        Propagating the wavefront on a denser grid around the seed,
        significantly improves the travel time accuracy. The value represents
        a fraction of the grid resolution. For instance, assuming a grid with
        spacing of 10m, if the sub_grid_resolution is set to 0.1, the
        resolution around the grid will be 1m.

        :rtype: TTGrid
        """

        _require_skfmm("compute travel times with the fast marching solver")

        if isinstance(seed, list):
            seed = np.array(seed)

        if not self.in_grid(seed.loc):
            logger.warning(f'{seed.label} is outside the grid. '
                           f'The travel time grid will not be calculated')
            return

        origin = self.origin
        shape = self.shape
        spacing = self.spacing

        sub_grid_spacing = spacing * sub_grid_resolution

        # extent = ((4 * sub_grid_spacing) * 1.2 + sub_grid_spacing)

        n_pts_inner_grid = (4 * spacing / sub_grid_spacing * 1.2).astype(int)
        for i in range(0, len(n_pts_inner_grid)):
            if n_pts_inner_grid[i] % 2:
                n_pts_inner_grid[i] += 1

        x_i = np.arange(0, n_pts_inner_grid[0]) * sub_grid_spacing[0]
        y_i = np.arange(0, n_pts_inner_grid[1]) * sub_grid_spacing[1]
        z_i = np.arange(0, n_pts_inner_grid[2]) * sub_grid_spacing[2]

        x_i = x_i - np.mean(x_i) + seed.x
        y_i = y_i - np.mean(y_i) + seed.y
        z_i = z_i - np.mean(z_i) + seed.z

        X_i, Y_i, Z_i = np.meshgrid(x_i, y_i, z_i, indexing='ij')

        coords = np.array([X_i.ravel(), Y_i.ravel(), Z_i.ravel()]).T

        vel = self.interpolate(coords, grid_space=False).reshape(
            X_i.shape)

        phi = np.ones_like(X_i)
        phi[int(np.floor(len(x_i) / 2)), int(np.floor(len(y_i) / 2)),
        int(np.floor(len(z_i) / 2))] = 0

        # TODO change with ttcrpy instead of skfmm
        tt_tmp = skfmm.travel_time(phi, vel, dx=sub_grid_spacing)

        tt_tmp_grid = TTGrid(self.network_code, tt_tmp, [x_i[0], y_i[0], z_i[0]],
                             sub_grid_spacing, seed, self.grid_id,
                             phase=self.phase,
                             float_type=self.float_type, label=self.label)

        # __init__(self, data_or_dims, origin, spacing, seed: Seed,
        # phase: Phases = Phases.P, value: float = 0,
        # float_type: FloatTypes = __default_float_type__,
        # grid_id: ResourceIdentifier = ResourceIdentifier(),
        # grid_units: GridUnits = __default_grid_units__):

        data = self.data

        xe = origin[0] + np.arange(0, shape[0], 1) * spacing[0]
        ye = origin[1] + np.arange(0, shape[1], 1) * spacing[1]
        ze = origin[2] + np.arange(0, shape[2], 1) * spacing[2]

        Xe, Ye, Ze = np.meshgrid(xe, ye, ze, indexing='ij')

        coords = np.array([Xe.ravel(), Ye.ravel(), Ze.ravel()])

        corner1 = np.array([np.min(x_i), np.min(y_i), np.min(z_i)])
        corner2 = np.array([np.max(x_i), np.max(y_i), np.max(z_i)])

        test = ((coords[0, :] >= corner1[0]) & (coords[0, :] <= corner2[0]) &
                (coords[1, :] >= corner1[1]) & (coords[1, :] <= corner2[1]) &
                (coords[2, :] >= corner1[2]) & (coords[2, :] <= corner2[2]))

        Xe_grid = Xe.ravel()[test]
        Ye_grid = Ye.ravel()[test]
        Ze_grid = Ze.ravel()[test]

        X = np.array([Xe_grid, Ye_grid, Ze_grid]).T

        tt_interp = tt_tmp_grid.interpolate(X, grid_space=False,
                                            order=3)[0]

        bias = np.max(tt_interp)

        phi_out = np.ones_like(Xe).ravel()
        phi_out[test] = tt_interp - bias

        phi_out = phi_out.reshape(Xe.shape)

        tt_out = skfmm.travel_time(phi_out, data, dx=spacing)

        # tt_out = tt_out.ravel() + bias
        tt_out = tt_out.ravel() + bias
        tt_out[test] = tt_interp
        tt_out = tt_out.reshape(Xe.shape)

        tt_out_grid = TTGrid(self.network_code, tt_out, self.origin,
                             self.spacing, seed, phase=self.phase,
                             float_type=self.float_type,
                             grid_units=self.grid_units,
                             velocity_model_id=self.grid_id,
                             label=self.label)

        tt_out_grid.data -= tt_out_grid.interpolate(seed.T,
                                                    grid_space=False,
                                                    order=3)[0]

        return tt_out_grid

    def to_time_multi_threaded(self, seeds: SeedEnsemble, cpu_utilisation=0.9,
                               *args, **kwargs):
        """
        Multithreaded version of the Eikonal solver
        based on scikit fast marching solver
        :param seeds: array of seed
        :type seeds: np.ndarray
        :param cpu_utilisation: fraction of the cpu core to be used for the
        processing task (between 0 and 1)
        :type cpu_utilisation: float between 0 and 1
        :param args: arguments to be passed directly to skfmm.travel_time
        function
        :param kwargs: keyword arguments to be passed directly to
        skfmm.travel_time function
        :return: a travel time grid ensemble
        :rtype: TravelTimeEnsemble
        """

        num_threads = int(np.ceil(cpu_utilisation * __cpu_count__))
        # ensuring that the number of threads is comprised between 1 and
        # __cpu_count__
        num_threads = np.max([np.min([num_threads, __cpu_count__]), 1])

        data = []
        for seed in seeds:
            if not self.in_grid(seed.loc):
                logger.warning(f'{seed.label} is outside the grid. '
                               f'The travel time grid will not be calculated')
                continue

        with Pool(num_threads) as pool:
            results = pool.map(self.to_time, seeds)

        tt_grid_ensemble = TravelTimeEnsemble(results)

        return tt_grid_ensemble

    def write_nlloc(self, path='.'):

        super().write_nlloc(path=path)

    def mv(self, origin, destination):
        """
        move the velocity grid files from {origin} to {destination}
        :param origin: origin
        :param destination:
        :return:
        """

        super().mv(self, self.base_name, origin, destination)

    @property
    def base_name(self):
        return self.get_base_name(self.network_code, self.phase)

    def write(self, filename, format='VTK', **kwargs):
        field_name = None
        if format == 'VTK':
            field_name = f'velocity_{self.phase.value}'

        super().write(filename, format=format, field_name=field_name, **kwargs)

    def plot_slice(self, axis: int, slice_position: float, grid_space: bool = False,
                   mask: Optional[dict] = None, ** kwargs):
        field_name = f'{self.phase.value} wave velocity '
        if self.grid_units == GridUnits.METER:
            field_name += ' (m/s)'
        if self.grid_units == GridUnits.KILOMETER:
            field_name += ' (km/s)'
        fig, ax = super().plot_slice(axis, slice_position, grid_space, field_name,
                                     mask, **kwargs)
        return fig, ax


class VelocityGridEnsemble:
    def __init__(self, p_velocity_grid, s_velocity_grid):
        """

        :param p_velocity_grid: p-wave 3D velocity grid
        :type p_velocity_grid: VelocityGrid3D
        :param s_velocity_grid: s-wave 3D velocity grid
        :type s_velocity_grid: VelocityGrid3D

        :NOTE: the p and s velocity grids must have the same dimensions and the same
        label
        """

        if not p_velocity_grid.is_like(s_velocity_grid):
            raise ValueError('The p and s velocity grid must be compatible')

        self.p_velocity_grid = p_velocity_grid
        self.s_velocity_grid = s_velocity_grid
        self.label = p_velocity_grid.label
        self.__i__ = 0

    def __getitem__(self, item):
        if item.upper() == 'P':
            return self.p_velocity_grid

        elif item.upper() == 'S':
            return self.s_velocity_grid

        else:
            raise ValueError(f'{item} is not a valid key. '
                             f'The key value must either be "P" or "S"')

    def __iter__(self):
        self.__i__ = 0
        return self

    def __next__(self):
        if self.__i__ < 2:
            if self.__i__ == '0':
                return self.p_velocity_grid
            elif self.__i__ == '1':
                return self.s_velocity_grid
        else:
            raise StopIteration

    @staticmethod
    def keys():
        return ['P', 'S']

    def write(self, path='.'):
        for key in self.keys():
            self[key].write(filename=path + "_" + key + "wave")

    def to_time_multi_threaded(self, seeds: SeedEnsemble, cpu_utilisation=0.9,
                               *args, **kwargs):
        """
        Multithreaded version of the Eikonal solver
        :param seeds:
        :param cpu_utilisation:
        :param args:
        :param kwargs:
        :return:
        """

        tt_grid_ensemble = TravelTimeEnsemble([])

        for key in self.keys():
            tt_grids = self[key].to_time_multi_threaded(seeds,
                                                        cpu_utilisation=
                                                        cpu_utilisation,
                                                        *args, **kwargs)

            tt_grid_ensemble += tt_grids

        return tt_grid_ensemble

    def to_time(self, seeds: SeedEnsemble, multi_threaded=False, *args, **kwargs):
        if multi_threaded:
            return self.to_time_multi_threaded(seeds)

        else:
            tt_grid_ensemble = TravelTimeEnsemble([])
            for key in self.keys():
                for seed in seeds:
                    tt_grid_ensemble += self[key].to_time(seed)

    @property
    def p(self):
        return self['P']

    @property
    def s(self):
        return self['S']

    @property
    def P(self):
        return self['P']

    @property
    def S(self):
        return self['S']

    @property
    def network(self):
        return self.s.network_code

    @property
    def network_code(self):
        return self.s.network_code

    @property
    def dims(self):
        return self.s.shape

    @property
    def shape(self):
        return self.s.shape

    @property
    def origin(self):
        return self.s.origin

    @property
    def corner(self):
        return self.s.corner

    @property
    def spacing(self):
        return self.s.spacing

    @property
    def grid_type(self):
        return self.s.grid_type

    @property
    def grid_units(self):
        return self.s.grid_units

    @property
    def resource_id(self):
        return self.s.resource_id

    @property
    def coordinate_system(self):
        return self.s.coordinate_system

    @property
    def float_type(self):
        return self.s.float_type


class SeismicPropertyGridEnsemble(VelocityGridEnsemble):
    def __init__(self,
                 p_velocity_grid: VelocityGrid3D,
                 s_velocity_grid: VelocityGrid3D,
                 density_grid: DensityGrid3D):
        """

        :param p_velocity_grid: p-wave 3D velocity grid
        :type p_velocity_grid: VelocityGrid3D
        :param s_velocity_grid: s-wave 3D velocity grid
        :type s_velocity_grid: VelocityGrid3D
        :param density_grid: density grid
        :type density_grid: DensityGrid3D

        :NOTE: the p and s velocity and the density grids must have the same dimensions
        and the same label
        """

        if not p_velocity_grid.is_like(s_velocity_grid) :
            raise ValueError(f'the p and s velocity grids are incompatible\n'
                             f'P velocity: {p_velocity_grid}'
                             f'S velocity: {s_velocity_grid}')

        if not p_velocity_grid.is_like(density_grid):
            raise ValueError(f'the p velocity and the density grids are incompatible\n'
                             f'P velocity: {p_velocity_grid}'
                             f'Density   : {density_grid}')

        self.density_grid = density_grid

        super().__init__(p_velocity_grid, s_velocity_grid)

    def __getitem__(self, item):
        if item.upper() == "P":
            return self.p_velocity_grid
        elif item.upper() == "S":
            return self.s_velocity_grid
        elif item.upper() == "DENSITY":
            return self.density_grid
        else:
            raise ValueError(
                f'{item} is not a valid key. Use "P", "S", or "DENSITY".'
            )

    def __repr__(self):
        return (f'P Velocity: {self.p_velocity_grid}\n'
                f'S Velocity: {self.s_velocity_grid}\n'
                f'Density   : {self.density_grid}')

    @property
    def density(self):
        return self['density']

    def to_surface_velocities(
            self,
            period_min: float = 0.1,
            period_max: float = 10.0,
            n_periods: int = 10,
            logspace: bool = True,
            periods_list: Union[np.ndarray, list] = None,
            phase: "Union[Phases, str]" = Phases.RAYLEIGH,
            multithreading: bool = True,  # kept for API; ignored (runs serial)
            z: "Union[Sequence, np.ndarray]" = None,
            disba_param: "Union[DisbaParam]" = DisbaParam(),
            velocity_type: VelocityType = VelocityType.GROUP
    ):
        """
        Compute surface-wave velocities and return a SurfaceVelocityEnsemble.

        This computes either group or phase velocities for Rayleigh or Love waves at
        requested periods, using DISBA under the hood. Periods can be provided
        explicitly via ``periods_list`` or generated from a range.

        Notes
        -----
        - Computation is performed **serially** (``multithreading`` is ignored).
        - DISBA expects inputs in kilometers and km/s. If this ensemble's
          ``grid_units`` are meters, thickness is converted m→km and velocities m/s→km/s
          for computation; outputs are converted back to match this ensemble's
          ``grid_type``:
            * if ``grid_type == GridTypes.VELOCITY_METERS`` → data in **m/s**
            * otherwise → data in **km/s**
        - Period selection:
          * Use **either** ``periods_list`` **or** the range arguments
            (``period_min``, ``period_max``, ``n_periods``, ``logspace``).
          * When ``periods_list`` is given, range arguments are ignored.

        Parameters
        ----------
        period_min : float, default=0.1
            Minimum period in seconds (used when ``periods_list`` is None).
        period_max : float, default=10.0
            Maximum period in seconds (used when ``periods_list`` is None).
        n_periods : int, default=10
            Number of periods to sample between ``period_min`` and ``period_max``
            (used when ``periods_list`` is None).
        logspace : bool, default=True
            If True, generate log-uniformly spaced periods; else linearly spaced
            (used when ``periods_list`` is None).
        periods_list : array-like of float or None, default=None
            Alternative to the range arguments. Explicit list/array of periods (s).
            Must be a 1D array of shape ``(n,)`` or a column vector ``(n, 1)``.
            Values must be sorted in **non-decreasing** order (ascending; duplicates
            allowed). If provided, ``period_min``, ``period_max``, ``n_periods``, and
            ``logspace`` are ignored.
        phase : Phases or str, default=Phases.RAYLEIGH
            Seismic phase: ``"rayleigh"`` or ``"love"``.
        multithreading : bool, default=True
            Ignored. Present for backward compatibility (computation runs serially).
        z : array-like or None, default=None
            Optional custom vertical coordinates. If None, layer centers are used.
            When ``grid_units == GridUnits.METER``, values are interpreted in meters;
            otherwise in kilometers.
        disba_param : DisbaParam, default=DisbaParam()
            Parameters passed through to DISBA.
        velocity_type : VelocityType, default=VelocityType.GROUP
            Which surface-wave velocity to compute: group or phase.

        Returns
        -------
        SurfaceVelocityEnsemble
            An ensemble with one surface-velocity grid per requested period.

        Raises
        ------
        ValueError
            If ``periods_list`` is provided but not 1D ``(n,)`` or ``(n, 1)``, or if it
            is not sorted in ascending order.
        ValueError
            If ``n_periods < 1`` or if ``period_max <= period_min`` when generating
            periods from a range.
        TypeError
            If argument types are inconsistent with the expected types.

        """
        if not isinstance(disba_param, DisbaParam):
            raise TypeError("disba_param must be type DisbaParam")

        if isinstance(phase, str):
            Phases(phase.upper())
        if isinstance(phase, Phases):
            phase = phase.value.lower()

        if periods_list is None:

            if logspace:
                periods = np.logspace(np.log10(period_min), np.log10(period_max),
                                      n_periods)
            else:
                periods = np.linspace(period_min, period_max, n_periods)
        else:
            periods = np.asarray(periods_list).ravel()
            if not np.all(periods[:-1] <= periods[1:]):
                raise ValueError("periods must be sorted in ascending order")
            n_periods = len(periods)

        algorithm = disba_param.algorithm
        dc = disba_param.dc

        # Prepare output containers (filled in serial loops)
        nx, ny, nz = self.shape
        planes_kms = [np.zeros((nx, ny), dtype=np.float64) for _ in range(n_periods)]

        if z is None:
            # Native layers: build thickness and layer-centered properties
            thickness = self.spacing[2] * np.ones(shape=(nz - 1), dtype=np.float64)

            # Convert to km for Disba if spatial grid is meters
            if self.grid_units == GridUnits.METER:
                thickness *= 1.0e-3

            # Convert velocity data to km/s if stored in m/s
            if self.grid_type == GridTypes.VELOCITY_METERS:
                vel_s = self.s.data.astype(np.float64) * 1.0e-3
                vel_p = self.p.data.astype(np.float64) * 1.0e-3
            else:
                vel_s = self.s.data.astype(np.float64)
                vel_p = self.p.data.astype(np.float64)

            rho = self.density.data.astype(np.float64)

            # Layer-center averages
            layers_s = 0.5 * (vel_s[:, :, 1:] + vel_s[:, :, :-1])
            layers_p = 0.5 * (vel_p[:, :, 1:] + vel_p[:, :, :-1])
            layers_rho = 0.5 * (rho[:, :, 1:] + rho[:, :, :-1])

            # make sure the surface wave type is correct
            if velocity_type not in (VelocityType.PHASE, VelocityType.GROUP):
                raise ValueError("Surface wave velocity must be either group or phase")

            for i in tqdm(range(nx)):
                for j in range(ny):
                    if velocity_type == VelocityType.PHASE:
                        pd = PhaseDispersion(
                            thickness=thickness,
                            velocity_p=layers_p[i, j],
                            velocity_s=layers_s[i, j],
                            density=layers_rho[i, j],
                            algorithm=algorithm,
                            dc=dc,
                        )
                    elif velocity_type == VelocityType.GROUP:
                        pd = GroupDispersion(
                            thickness=thickness,
                            velocity_p=layers_p[i, j],
                            velocity_s=layers_s[i, j],
                            density=layers_rho[i, j],
                            algorithm=algorithm,
                            dc=dc,
                        )
                    cmod = pd(periods, mode=0, wave=phase).velocity  # km/s
                    for k in range(n_periods):
                        planes_kms[k][i, j] = cmod[k]

        else:
            # User-provided vertical coordinates → build layer centers/thickness
            z = np.asarray(z, dtype=np.float64)
            layers_centers = 0.5 * (z[1:] + z[:-1])
            layers_thickness = z[1:] - z[:-1]

            # Interpolate properties at layer centers for each (x, y)
            x = np.arange(self.origin[0], self.corner[0], self.spacing[0])
            y = np.arange(self.origin[1], self.corner[1], self.spacing[1])
            xg, yg, zg = np.meshgrid(x, y, layers_centers, indexing="ij")
            interp_xyz = np.column_stack(
                (xg.reshape(-1), yg.reshape(-1), zg.reshape(-1))
            )

            vel_s = self.s.interpolate(interp_xyz, grid_space=False).astype(np.float64)
            vel_p = self.p.interpolate(interp_xyz, grid_space=False).astype(np.float64)
            rho = self.density.interpolate(interp_xyz, grid_space=False).astype(
                np.float64
            )

            # Convert to Disba units when needed
            if self.grid_units == GridUnits.METER:
                layers_thickness = layers_thickness.astype(np.float64) * 1.0e-3
                if self.grid_type == GridTypes.VELOCITY_METERS:
                    vel_s *= 1.0e-3
                    vel_p *= 1.0e-3
            else:
                if self.grid_type == GridTypes.VELOCITY_METERS:
                    # Spatial grid is km/ft, but data are m/s → convert to km/s
                    vel_s *= 1.0e-3
                    vel_p *= 1.0e-3

            # Fill planes in serial
            for ii in tqdm(range(nx)):
                for jj in range(ny):
                    mask = (interp_xyz[:, 0] == x[ii]) & (interp_xyz[:, 1] == y[jj])
                    if velocity_type == VelocityType.PHASE:
                        pd = PhaseDispersion(
                            thickness=layers_thickness,
                            velocity_p=vel_p[mask],
                            velocity_s=vel_s[mask],
                            density=rho[mask],
                            algorithm=algorithm,
                            dc=dc,
                        )
                    elif velocity_type == VelocityType.GROUP:
                        pd = GroupDispersion(
                            thickness=layers_thickness,
                            velocity_p=vel_p[mask],
                            velocity_s=vel_s[mask],
                            density=rho[mask],
                            algorithm=algorithm,
                            dc=dc,
                        )
                    cmod = pd(periods, mode=0, wave=phase).velocity  # km/s
                    for k in range(n_periods):
                        planes_kms[k][ii, jj] = cmod[k]

        # Convert planes back to the desired data unit and wrap as PhaseVelocity
        if velocity_type == VelocityType.PHASE:
            ensemble = PhaseVelocityEnsemble()
        if velocity_type == VelocityType.GROUP:
            ensemble = GroupVelocityEnsemble()

        for k, per in enumerate(periods):
            if self.grid_type == GridTypes.VELOCITY_METERS:
                data = planes_kms[k] * 1.0e3  # km/s → m/s
                out_grid_type = GridTypes.VELOCITY_METERS
            else:
                data = planes_kms[k]
                out_grid_type = self.grid_type
            if velocity_type == VelocityType.PHASE:
                sv = PhaseVelocity(
                    network_code=self.network_code,
                    data_or_dims=data,
                    period=float(per),
                    phase=Phases(phase.upper()),
                    grid_type=out_grid_type,
                    grid_units=self.grid_units,
                    spacing=self.spacing[:-1],
                    origin=self.origin[:-1],
                    resource_id=ResourceIdentifier(),
                    value=0.0,
                    coordinate_system=self.coordinate_system,
                    label=self.label,
                    float_type=self.float_type if hasattr(self, "float_type")
                    else FloatTypes.FLOAT,

                )
            else:
                sv = GroupVelocity(
                    network_code=self.network_code,
                    data_or_dims=data,
                    period=float(per),
                    phase=Phases(phase.upper()),
                    grid_type=out_grid_type,
                    grid_units=self.grid_units,
                    spacing=self.spacing[:-1],
                    origin=self.origin[:-1],
                    resource_id=ResourceIdentifier(),
                    value=0.0,
                    coordinate_system=self.coordinate_system,
                    label=self.label,
                    float_type=self.float_type if hasattr(self, "float_type")
                    else FloatTypes.FLOAT,

                )

            ensemble.append(sv)

        return ensemble

    def transform_to(self, values):
        return self.s.transform_to_grid(values)

    def transform_from(self, values):
        return self.s.transform_from(values)

    def plot_sensitivity_kernel(self, period: float, x: Union[float, int],
                                y: Union[float, int], z: Union[List, np.ndarray],
                                phase: Union[Phases, str] = Phases.RAYLEIGH,
                                disba_param: Union[DisbaParam] = DisbaParam(),
                                grid_space: bool = False):

        if not isinstance(disba_param, DisbaParam):
            raise TypeError(f'disba_param must be type DisbaParam')
        if isinstance(phase, str):
            Phases(phase.upper())
        if isinstance(phase, Phases):
            phase = phase.value.lower()

        algorithm = disba_param.algorithm
        dc = disba_param.dc
        dp = disba_param.dp

        if z is None:
            # calculate the thickness and velocity values of layers
            layers_thickness = self.spacing[2] * np.ones(shape=(self.shape[2] - 1))
            if self.grid_units == GridUnits.METER:
                layers_thickness /= 1.e3
                velocity_s = self.s.data * 1.e-3
                velocity_p = self.p.data * 1.e-3

            else:
                velocity_s = self.s.data
                velocity_p = self.p.data

            # check if the points are inside the grid
            if not grid_space:
                tmp = self.transform_to((x, y, self.origin[2]))
                i = int(tmp[0])
                j = int(tmp[1])
            else:
                i, j = int(x), int(y)
            if not self.s.in_grid([i, j, 0], grid_space=True):
                raise IndexError(f'The point {i, j} is not inside the grid.')

            # seismic parameters of layers (model 1D)
            layers_s = 0.5 * (velocity_s[i, j, 1:] + velocity_s[i, j, :-1])
            layers_p = 0.5 * (velocity_p[i, j, 1:] + velocity_p[i, j, :-1])
            density = 0.5 * (self.density.data[i, j, 1:] + self.density.data[i, j, :-1])
            # calculate the Sensitivity kernel
            ps = PhaseSensitivity(thickness=layers_thickness, velocity_p=layers_p,
                                  velocity_s=layers_s, density=density, dc=dc,
                                  algorithm=algorithm, dp=dp)
            sensitivity = ps(period, mode=0, wave=phase, parameter="velocity_s")
            kernel = sensitivity.kernel

            # interpolate the Sensitivity kernel in nodes
            kernel_nodes = np.zeros(shape=(velocity_s.shape[2], ))
            depth = self.origin[2] + np.arange(velocity_s.shape[2]) * self.spacing[2]
            kernel_nodes[0] = kernel[0]
            kernel_nodes[-1] = kernel[-1]
            kernel_nodes[1:-1] = 0.5 * (kernel[1:] + kernel[:-1])

            # plot the  Sensitivity kernel as function of depth
            ax1 = plt.subplot(1, 2, 1)
            ax1.plot(kernel_nodes, depth, "-ob")
            ax1.set_ylim([depth[-1] + 0.5 * self.spacing[2], depth[0]])
            ax1.set_xlabel("Sensitivity kernel")
            ax1.set_title("Period {0:2.2f} (s)".format(period))
            plt.grid(True)
            ax2 = plt.subplot(1, 2, 2)
            ax2.set_ylim([depth[-1] + 0.5 * self.spacing[2], depth[0]])
            if self.grid_units == GridUnits.METER:
                ax1.set_ylabel("Depth (m)")
                ax2.plot(velocity_s[i, j, :] * 1.e3, depth, "-k")
                ax2.set_xlabel("Velocity (m/s)")
            else:
                ax1.set_ylabel("Depth (km)")
                ax2.plot(velocity_s[i, j, :], depth, "-k")
                ax2.set_xlabel("Velocity (km/s)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            return depth, kernel_nodes

        else:
            # check if the point is inside the grid
            if grid_space:
                x, y, _ = self.transform_from((x, y, 0))
            if not self.s.in_grid([x, y, self.origin[2]], grid_space=False):
                raise IndexError(f'The point {x, y} is not inside the grid.')
            # define layers thickness and centers
            layer_centers = 0.5 * (z[1:] + z[:-1])
            layers_thickness = z[1:] - z[:-1]
            coord_interpolation = np.column_stack((x * np.ones_like(layer_centers),
                                                   y * np.ones_like(layer_centers),
                                                   layer_centers))
            coord_interpolation_z = np.column_stack((x * np.ones_like(z),
                                                     y * np.ones_like(z),
                                                     z))

            # interpolate seismic parameters at the centers of layers
            velocity_s = self.s.interpolate(coord_interpolation, grid_space=False)
            velocity_sz = self.s.interpolate(coord_interpolation_z, grid_space=False)
            velocity_p = self.p.interpolate(coord_interpolation, grid_space=False)
            density = self.density.interpolate(coord_interpolation, grid_space=False)
            if self.grid_units == GridUnits.METER:
                layers_thickness *= 1.e-3
                velocity_s *= 1.e-3
                velocity_p *= 1.e-3
            ps = PhaseSensitivity(thickness=layers_thickness, velocity_p=velocity_p,
                                  velocity_s=velocity_s, density=density, dc=dc,
                                  algorithm=algorithm, dp=dp)
            sensitivity = ps(period, mode=0, wave=phase, parameter="velocity_s")
            kernel = sensitivity.kernel

            # interpolate the Sensitivity kernel in nodes
            kernel_nodes = np.zeros(shape=(len(z), ))
            kernel_nodes[0] = kernel[0]
            kernel_nodes[-1] = kernel[-1]
            kernel_nodes[1:-1] = 0.5 * (kernel[1:] + kernel[:-1])

            # plot kernel
            ax1 = plt.subplot(1, 2, 1)
            ax1.plot(kernel_nodes, z, "-ob")
            ax1.set_ylim([z[-1] + 0.5 * self.spacing[2], z[0]])
            ax1.set_xlabel("Sensitivity kernel")
            ax1.set_title("Period {0:2.2f} (s)".format(period))
            plt.grid(True)
            ax2 = plt.subplot(1, 2, 2)
            ax2.set_ylim([z[-1] + 0.5 * self.spacing[2], z[0]])
            ax2.plot(velocity_sz, z, "-k")
            if self.grid_units == GridUnits.METER:
                ax1.set_ylabel("Depth (m)")
                ax2.set_xlabel("Velocity (m/s)")
            else:
                ax1.set_ylabel("Depth (km)")
                ax2.set_xlabel("Velocity (km/s)")
            plt.grid(True)
            plt.show()
            return z, kernel_nodes

    def write(self, path: Union[Path, str] = '.'):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def read(cls, path: Union[Path, str]):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f'The object in {path} is not a '
                            f'SeismicPropertyGridEnsemble object')
        return obj


class SeededGridType(Enum):
    TIME = 'TIME'
    ANGLE = 'ANGLE'
    AZIMUTH = 'AZIMUTH'
    TAKEOFF = 'TAKEOFF'

    def __str__(self):
        return str(self.value)


class SeededGrid(TypedGrid):
    """
    container for seeded grids (e.g., travel time, azimuth and take off angle)
    """

    __doc__ = f'{TypedGrid.__doc__}\n'

    def __init__(self, network_code, data_or_dims, origin, spacing, seed: Seed,
                 velocity_model_id: ResourceIdentifier,
                 phase: Phases = Phases.P, value: float = 0,
                 grid_units: GridUnits = __default_grid_units__,
                 grid_type: SeededGridType = SeededGridType.TIME,
                 float_type: FloatTypes = __default_float_type__,
                 grid_id: ResourceIdentifier = ResourceIdentifier(),
                 label: str = __default_grid_label__):
        """
        :param network: network code
        :param data_or_dims: data or data dimensions.
        If dimensions not provided a grid with value=value is created
        :param origin: origin of the grid
        :param spacing: the spacing between grid nodes
        :param seed: seed location of the grid - The seed or instrument location
        :param velocity_model_id: velocity model id
        :param phase: Seismic Phase
        :param value: value of the grid when only specifying the dimensions
        :param grid_units: units of measurement used to express values
        :param grid_type: type of grid (e.g., travel time, azimuth, take off angle)
        :param float_type: float type either 'FLOAT' or 'DOUBLE'
        :param grid_id: the grid id
        :param label: label of the grid
        :param coordinate_system: coordinate system either NED or ENU
        """

        self.network = network_code
        self.seed = seed
        self.velocity_model_id = velocity_model_id

        if isinstance(grid_type, str):
            self.grid_type = SeededGridType(grid_type)
        elif isinstance(grid_type, SeededGridType):
            self.grid_type = grid_type
        elif isinstance(grid_type, GridTypes):
            self.grid_type = SeededGridType(grid_type.value)
        else:
            raise TypeError('grid_type must be a SeededGridType object')

        super().__init__(data_or_dims, origin, spacing,
                         phase=phase, value=value,
                         grid_type=grid_type, grid_units=grid_units,
                         float_type=float_type, grid_id=grid_id, label=label,
                         coordinate_system=seed.coordinates.coordinate_system)

        # ensure the data are expressed in the appropriate float_type
        self.data.astype(float_type.value)

    def __repr__(self):
        line = f'{self.grid_type} Grid\n' \
               f'    origin     : {self.origin}\n' \
               f'    spacing    : {self.spacing}\n' \
               f'    dimensions : {self.shape}\n' \
               f'    seed label : {self.seed_label}\n' \
               f'    seed       : {self.seed}'
        return line

    @property
    def network_code(self):
        return self.network

    @property
    def station_code(self):
        return self.seed.station

    @property
    def location_code(self):
        return self.seed.location

    @property
    def instrument_code(self):
        return self.seed.label

    @property
    def seed_label(self):
        return self.seed.short_label

    @property
    def seed_units(self):
        return self.seed.units.value

    @property
    def waveform_id(self):
        return WaveformStreamID(network_code=self.network_code,
                                station_code=self.station_code,
                                location_code=self.location_code)

    # @property
    # def coordinate_system(self):
    #     return self.seed.coordinates.coordinate_system

    @staticmethod
    def get_base_name(network_code, phase, seed_label, grid_type):

        if not isinstance(grid_type, SeededGridType):
            try:
                if isinstance(grid_type, GridTypes):
                    grid_type = SeededGridType(grid_type.value)
                else:
                    grid_type = SeededGridType(grid_type)
            except ValueError:
                raise ValueError(f'{grid_type} is not a valid grid type')

        base_name = f'{network_code}.{phase}.{seed_label}.' \
                    f'{grid_type.value.lower()}'
        return base_name

    @property
    def base_name(self):
        base_name = self.get_base_name(self.network_code, self.phase.upper(),
                                       self.seed_label, self.grid_type)
        return base_name

    def write_nlloc(self, path='.'):
        self._write_grid_data(path=path)
        self._write_grid_header(path=path)
        self._write_grid_model_id(path=path)

    def _write_grid_data(self, path='.'):

        Path(path).mkdir(parents=True, exist_ok=True)

        with open(Path(path) / (self.base_name + '.buf'), 'wb') \
                as out_file:
            if self.float_type.value == 'float32':
                out_file.write(self.data.astype(float32).tobytes())

            elif self.float_type.value == 'float64':
                out_file.write(self.data.astype(np.float64).tobytes())

    def _write_grid_header(self, path='.'):

        # convert 'METER' to 'KILOMETER'
        if self.grid_units.value == 'METER':
            origin = self.origin / 1000
            spacing = self.spacing / 1000
        else:
            origin = self.origin
            spacing = self.spacing

        line1 = f'{self.shape[0]:d} {self.shape[1]:d} {self.shape[2]:d}  ' \
                f'{origin[0]:f} {origin[1]:f} {origin[2]:f}  ' \
                f'{spacing[0]:f} {spacing[1]:f} {spacing[2]:f}  ' \
                f'{self.grid_type}\n'

        with open(Path(path) / (self.base_name + '.hdr'), 'w') as out_file:
            out_file.write(line1)

            if self.grid_type.value in ['TIME', 'ANGLE']:

                if self.seed_units is None:
                    logger.warning(f'seed_units are not defined. '
                                   f'Assuming same units as grid ('
                                   f'{self.grid_units}')
                if self.grid_units.value == 'METER':
                    seed = np.array(self.seed.coordinates.loc) / 1000

                line2 = u"%s %f %f %f\n" % (self.seed_label,
                                            seed[0], seed[1], seed[2])
                out_file.write(line2)

            out_file.write(u'TRANSFORM  NONE\n')

    def _write_grid_model_id(self, path='.'):
        with open(Path(path) / (self.base_name + '.mid'), 'w') as out_file:
            out_file.write(f'{self.model_id}')


def adjust_gradients_for_coordinate_system(gds, coordinate_system):
    """
    Adjust gradients based on the coordinate system.
    :param gds: Gradients (tuple of numpy arrays).
    :param coord_system: Coordinate system (instance of CoordinateSystem).
    :return: Adjusted gradients.
    """
    coord_system = coordinate_system
    if coord_system == CoordinateSystem.NED:
        return {'north': gds[0], 'east': gds[1], 'down': gds[2]}
    elif coord_system == CoordinateSystem.ENU:
        return {'north': gds[1], 'east': gds[0], 'down': -gds[2]}
    elif coord_system == CoordinateSystem.NEU:
        return {'north': gds[0], 'east': gds[1], 'down': -gds[2]}
    elif coord_system == CoordinateSystem.END:
        return {'north': gds[1], 'east': gds[0], 'down': gds[2]}
    else:
        raise ValueError("Invalid coordinate system")


class TTGrid(SeededGrid):
    def __init__(self, network_code, data_or_dims, origin, spacing, seed: Seed,
                 velocity_model_id: ResourceIdentifier,
                 phase: Phases = Phases.P, value: float = 0,
                 grid_units: GridUnits = __default_grid_units__,
                 float_type: FloatTypes = __default_float_type__,
                 grid_id: ResourceIdentifier = ResourceIdentifier(),
                 label=__default_grid_label__):
        super().__init__(network_code, data_or_dims, origin, spacing, seed,
                         velocity_model_id=velocity_model_id, phase=phase,
                         value=value, grid_type=GridTypes.TIME,
                         grid_units=grid_units,
                         float_type=float_type,
                         grid_id=grid_id,
                         label=label)

    def to_azimuth(self):
        """
        This function calculate the azimuth for every
        grid point given a travel time grid calculated using an Eikonal solver
        :return: azimuth angles grids
        """

        gds = np.gradient(self.data)
        tmp = adjust_gradients_for_coordinate_system(gds, self.coordinate_system)
        north = tmp['north']
        east = tmp['east']

        azimuth = np.arctan2(east, north) * 180 / np.pi
        azimuth = np.mod(azimuth, 360)  # Ensuring azimuth is within [0, 360] range

        return AngleGrid(self.network_code, azimuth, self.origin, self.spacing,
                         self.seed, phase=self.phase, float_type=self.float_type,
                         grid_id=ResourceIdentifier(), grid_type=GridTypes.AZIMUTH,
                         velocity_model_id=self.velocity_model_id)

    def to_takeoff(self):
        """
        This function calculate the takeoff angle for every grid point given a
        travel time grid calculated using an Eikonal solver
        :return: takeoff angles grid
        .Note: The convention for the takeoff angle is that 0 degree is down.
        """
        gds = np.gradient(self.data)
        tmp = adjust_gradients_for_coordinate_system(gds, self.coordinate_system)

        east = tmp['east']
        north = tmp['north']
        down = tmp['down']

        hor = np.sqrt(north ** 2 + east ** 2)
        takeoff = np.arctan2(hor, down) * 180 / np.pi
        takeoff = np.clip(takeoff, 0, 180)
        # takeoff is zero pointing down
        return AngleGrid(self.network_code, takeoff, self.origin, self.spacing,
                         self.seed, phase=self.phase, float_type=self.float_type,
                         grid_id=ResourceIdentifier(), grid_units=self.grid_units,
                         grid_type=GridTypes.TAKEOFF,
                         velocity_model_id=self.velocity_model_id)

    def to_azimuth_point(self, coord, grid_space=False, mode='nearest',
                         order=1, **kwargs):
        """
        calculate the azimuth angle at a particular point on the grid for a
        given seed location
        :param coord: coordinates at which to calculate the takeoff angle
        :param grid_space: true if the coordinates are expressed in
        grid space (indices can be fractional) as opposed to model space
        (x, y, z)
        :param mode: interpolation mode
        :param order: interpolation order
        :return: takeoff angle at the location coord
        """

        return self.to_azimuth().interpolate(coord,
                                             grid_space=grid_space,
                                             mode=mode, order=order,
                                             **kwargs)[0]

    def to_takeoff_point(self, coord, grid_space=False, mode='nearest',
                         order=1, **kwargs):
        """
        calculate the takeoff angle at a particular point on the grid for a
        given seed location
        :param coord: coordinates at which to calculate the takeoff angle
        :param grid_space: true if the coordinates are expressed in
        grid space (indices can be fractional) as opposed to model space
        (x, y, z)
        :param mode: interpolation mode
        :param order: interpolation order
        :return: takeoff angle at the location coord
        """
        return self.to_takeoff().interpolate(coord,
                                             grid_space=grid_space,
                                             mode=mode, order=order,
                                             **kwargs)[0]

    def ray_tracer(self, start, grid_space=False, max_iter=1000,
                   arrival_id=None):
        """
        This function calculates the ray between a starting point (start) and an
        end point, which should be the seed of the travel_time grid, using the
        gradient descent method.
        :param start: the starting point (usually event location)
        :type start: tuple, list or numpy.array
        :param grid_space: true if the coordinates are expressed in
        grid space (indices can be fractional) as opposed to model space
        (x, y, z)
        :param max_iter: maximum number of iteration
        :param arrival_id: id of the arrival associated to the ray if
        applicable
        :rtype: numpy.ndarray
        """

        return ray_tracer(self, start, grid_space=grid_space, max_iter=max_iter,
                          arrival_id=arrival_id, velocity_model_id=self.model_id)

    @classmethod
    def from_velocity(cls, seed, seed_label, velocity_grid):
        return velocity_grid.to_time(seed, seed_label)

    def write_nlloc(self, path='.'):
        return super().write_nlloc(path=path)

    @property
    def instrument(self):
        return self.seed_label

    @property
    def location(self):
        return self.seed.location_code

    def write(self, filename, format='VTK', **kwargs):
        """Write the grid data to disk.

        :param filename: Path where the grid should be serialised.
        :type filename: str
        :param format: Output format understood by :meth:`Grid.write`.
        :type format: str
        :param kwargs: Additional keyword arguments forwarded to the parent writer.
        :type kwargs: dict
        :returns: None
        :rtype: None
        """
        field_name = None
        if format == 'VTK':
            field_name = 'Travel Time'

        super().write(filename, format=format, field_name=field_name, **kwargs)

    def plot(self):
        if self.ndim == 2:
            fig = plt.figure()
            ax = fig.add_subplot()
            im = ax.imshow(self.data.T, origin='lower',
                           extent=(self.origin[0], self.corner[0],
                                   self.origin[1], self.corner[1]),
                           cmap="seismic")
            ax.plot(self.seed.x, self.seed.y, "o", color="green")

            cb = fig.colorbar(im, ax=ax, orientation='vertical')
            if self.grid_units == GridUnits.METER:
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                cb.set_label('Travel time (s)', rotation=270,
                             labelpad=10)
            if self.grid_units == GridUnits.KILOMETER:
                ax.set_xlabel("X (km)")
                ax.set_ylabel("Y (km)")
                cb.set_label('Travel time (s)', rotation=270, labelpad=10)
            plt.show()
        else:
            logger.warning("not implemented for 3D grids")


class TravelTimeEnsemble:
    def __init__(self, travel_time_grids):
        """
        Combine a list of travel time grids together providing meta
        functionality (multithreaded ray tracing, sorting, travel-time
        calculation for a specific location etc.). It is assumed that
        all grids are compatible, i.e., that all the grids have the same
        origin, spacing and dimensions.
        :param travel_time_grids: a list of TTGrid objects

        :NOTE: The travel time grids must all have the same grid labels if not,
        the object will not be created.
        """

        self.travel_time_grids = []
        for travel_time_grid in travel_time_grids:
            if travel_time_grid.label != travel_time_grids[0].label:
                raise ValueError('all travel time grids must have the same '
                                 'label')
            self.travel_time_grids.append(travel_time_grid)

        self.__i__ = 0

        for tt_grid in self.travel_time_grids:
            try:
                assert tt_grid.check_compatibility(travel_time_grids[0])
            except:
                raise AssertionError('grids are not all compatible')

    def __len__(self):
        return len(self.travel_time_grids)

    def __add__(self, other):
        if isinstance(other, TTGrid):
            self.travel_time_grids.append(other)
        elif isinstance(other, TravelTimeEnsemble):
            for travel_time_grid in other.travel_time_grids:
                self.travel_time_grids.append(travel_time_grid)

        return TravelTimeEnsemble(self.travel_time_grids)

    def __iter__(self):
        self.__i__ = 0
        return self

    def __next__(self):
        if self.__i__ < len(self):
            result = self.travel_time_grids[self.__i__]
            self.__i__ += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.travel_time_grids[item]
        if isinstance(item, str):
            for tt_grid in self.travel_time_grids:
                if tt_grid.seed_label == item:
                    return tt_grid

    def __repr__(self):
        line = f'Number of travel time grids: {len(self)}'
        return line

    # @classmethod
    # def from_files(cls, path, format='PICKLE'):
    #     """
    #     create a travel time ensemble from files located in a directory
    #     :param path: the base path to the directory containing the travel time
    #     files.
    #     :return:
    #     """
    #     tt_grids = []
    #     for fle in Path(path).glob('*time*.hdr'):
    #         path = fle.parent
    #         base_name = '.'.join(fle.name.split('.')[:-1])
    #         fname = str(Path(path) / base_name)
    #         tt_grid = read_grid(fname, format='NLLOC',
    #                             float_type=__default_float_type__)
    #         tt_grids.append(tt_grid)
    #
    #     return cls(tt_grids)

    @classmethod
    def from_files(cls, path, format='PICKLE'):
        """
        create a travel time ensemble from files located in a directory
        :param path: the base path to the directory containing the travel time
        files.
        :return:
        """
        tt_grids = []
        for file in Path(path).glob('*'):
            tt_grid = read_grid(file, format=format,
                                float_type=__default_float_type__)
            tt_grids.append(tt_grid)

        return cls(tt_grids)

    def select(self, instrument_codes: Optional[List[str]] = None,
               phase: Union[Phases, 'str'] = None):
        """
        return a list of grid corresponding to seed_labels.
        :param instrument_codes: seed labels of the travel time grids to return
        :param phase: the phase {'P' or 'S'}, both if None.
        :return: a list of travel time grids
        :rtype: TravelTimeEnsemble
        """

        if (instrument_codes is None) and (phase is None):
            return self

        tmp = []
        if instrument_codes is None:
            instrument_codes = np.unique(self.seeds)

        if phase is None:
            phases = [Phases.P.value, Phases.S.value]
        else:
            phases = [phase.value if isinstance(phase, Phases) else Phases(phase).value
                      for phase in phase]

        returned_grids = []
        for travel_time_grid in self.travel_time_grids:
            if travel_time_grid.seed.label in instrument_codes:
                if isinstance(travel_time_grid.phase, Phases):
                    grid_phase = travel_time_grid.phase.value
                else:
                    grid_phase = travel_time_grid.phase
                if grid_phase in phases:
                    returned_grids.append(travel_time_grid)

        return TravelTimeEnsemble(returned_grids)

    def sort(self, ascending: bool = True):
        """
        sorting the travel time grid by seed_label
        :param ascending: if true the grids are sorted in ascending order
        :param ascending: bool
        :return: sorted travel time grids.
        :rtype: TravelTimeEnsemble
        """

        i = np.sort(self.seed_labels)

        if not ascending:
            i = i[-1::-1]

        sorted_tt_grids = np.array(self.travel_time_grids)[i]

        return TravelTimeEnsemble(sorted_tt_grids)

    def travel_time(self, seed, grid_space: bool = False,
                    seed_labels: Optional[list] = None,
                    phase: Optional[list] = None):
        """
        calculate the travel time at a specific point for a series of location
        ids
        :param seed: travel time seed
        :param grid_space: true if the coordinates are expressed in
        grid space (indices can be fractional) as opposed to model space
        (x, y, z)
        :param seed_labels: a list of locations from which to calculate the
        travel time.
        :param phase: a list of phases for which the travel time need to be
        calculated
        :return: a list of dictionary containing the travel time and location id
        """

        if isinstance(seed, list):
            seed = np.array(seed)

        if grid_space:
            seed = self.travel_time_grids[0].transform_from(seed)

        if not self.travel_time_grids[0].in_grid(seed):
            raise ValueError('seed is outside the grid')

        tt_grids = self.select(instrument_codes=seed_labels)

        tts = []
        labels = []
        phases = []
        for tt_grid in tt_grids:
            labels.append(tt_grid.seed.label)
            tts.append(tt_grid.interpolate(seed.T,
                                           grid_space=False)[0])
            phases.append(tt_grid.phase)

        tts_dict = {}
        for phase in np.unique(phases):
            tts_dict[phase] = {}

        for label, tt, phase in zip(labels, tts, phases):
            tts_dict[phase][label] = tt

        return tts_dict

    def write_nlloc(self, path='.'):
        for tt_grid in self.travel_time_grids:
            tt_grid.write_nlloc(path=path)

    def write(self, path='.', format='PICKLE'):
        """
        write the travel time grids to disk
        :param path: path to the directory where the travel time grids are to
        be written
        :param format: format in which the travel time grids are to be written
        :return:
        """

        if format == 'PICKLE':
            for tt_grid in self.travel_time_grids:
                filename = Path(path) / f'{tt_grid.seed_label}.{tt_grid.phase}.pickle'
                tt_grid.write(filename, format=format)
        elif format == 'NLLOC':
            self.write_nlloc(path=path)
        else:
            raise ValueError(f'{format} is not a valid format')

    def angles(self, seed, grid_space: bool = False,
               seed_labels: Optional[list] = None,
               phase: Optional[list] = None, **kwargs):
        """
        calculate the azimuth at a specific point for a series of location
        ids
        :param seed: travel time seed
        :param grid_space: true if the coordinates are expressed in
        grid space (indices can be fractional) as opposed to model space
        (x, y, z)
        :param seed_labels: a list of locations from which to calculate the
        travel time.
        :param phase: a list of phases for which the travel time need to be
        calculated
        :return: a list of dictionary containing the azimuth and location id
        """

        if isinstance(seed, list):
            seed = np.array(seed)

        if grid_space:
            seed = self.travel_time_grids[0].transform_from(seed)

        if not self.travel_time_grids[0].in_grid(seed):
            raise ValueError('seed is outside the grid')

        tt_grids = self.select(instrument_codes=seed_labels)

        azimuths = []
        takeoffs = []
        labels = []
        phases = []
        for tt_grid in tt_grids:
            labels.append(tt_grid.seed_label)
            azimuths.append(tt_grid.to_azimuth_point(seed.T,
                                                     grid_space=False,
                                                     **kwargs))
            takeoffs.append(tt_grid.to_takeoff_point(seed.T,
                                                     grid_space=False,
                                                     **kwargs))
            phases.append(tt_grid.phase)

        azimuth_dict = {}
        takeoff_dict = {}
        for phase in np.unique(phases):
            azimuth_dict[phase] = {}
            takeoff_dict[phase] = {}

        for label, azimuth, takeoff, phase in zip(labels, azimuths, takeoffs,
                                                  phases):
            takeoff_dict[phase][label] = takeoff
            azimuth_dict[phase][label] = azimuth

        angle_dict = {}
        angle_dict['takeoff'] = takeoff_dict
        angle_dict['azimuth'] = azimuth_dict

        return angle_dict

    def ray_tracer(self, starts, instrument_codes=None, multithreading=False,
                   cpu_utilisation=0.9, grid_space=False, max_iter=1000):
        """

        :param starts: origin of the ray, usually the location of the events
        :param instrument_codes: a list of seed labels
        :param grid_space: true if the coordinates are expressed in
        grid space (indices can be fractional) as opposed to model space
        (x, y, z)
        :param multithreading: if True use multithreading
        :param max_iter: maximum number of iteration
        :param cpu_utilisation: fraction of core to use, between 0 and 1.
        The number of core to be use is bound between 1 and the total number of
        cores
        :return: a list of rays
        :rtype: list
        """

        if instrument_codes is None:
            travel_time_grids = self
        else:
            if isinstance(instrument_codes, str):
                instrument_codes = [instrument_codes]
            travel_time_grids = self.select(instrument_codes=instrument_codes)

        kwargs = {'grid_space': grid_space,
                  'max_iter': max_iter}

        # consider the case where starts is a single point and not an array of points
        if starts.ndim == 1:
            starts = np.array([starts])

        if multithreading:

            ray_tracer_func = partial(ray_tracer, **kwargs)

            num_threads = int(np.ceil(cpu_utilisation * __cpu_count__))
            # ensuring that the number of threads is comprised between 1 and
            # __cpu_count__
            num_threads = np.max([np.min([num_threads, __cpu_count__]), 1])

            data = []
            for start in starts:
                for travel_time_grid in travel_time_grids:
                    data.append((travel_time_grid, start))

            with Pool(num_threads) as pool:
                results = pool.starmap(ray_tracer_func, data)

            # for result in results:
            #     result.network = self.travel_time_grids[0].network_code

        else:
            results = []
            for travel_time_grid in travel_time_grids:
                for start in starts:
                    results.append(travel_time_grid.ray_tracer(start, **kwargs))

        return results

    @property
    def seeds(self):
        seeds = []
        for seed_label in self.seed_labels:
            seeds.append(self.select(instrument_codes=seed_label)[0].seed)

        return np.array(seeds)

    @property
    def label(self):
        return self.travel_time_grids[0].label

    @property
    def seed_labels(self):
        seed_labels = []
        for grid in self.travel_time_grids:
            seed_labels.append(grid.seed_label)
        return np.unique(np.array(seed_labels))

    @property
    def labels(self):
        labels = []
        for grid in self.travel_time_grids:
            labels.append(grid.label)
        return np.unique(np.array(labels))

    @property
    def shape(self):
        return self.travel_time_grids[0].shape

    @property
    def origin(self):
        return self.travel_time_grids[0].origin

    @property
    def spacing(self):
        return self.travel_time_grids[0].spacing


class AngleGrid(SeededGrid):

    def __init__(self, network_code, data_or_dims, origin, spacing, seed: Seed,
                 velocity_model_id: ResourceIdentifier,
                 phase: Phases = Phases.P, value: float = 0,
                 grid_units: GridUnits = GridUnits.DEGREES,
                 float_type: FloatTypes = __default_float_type__,
                 grid_type: GridTypes = GridTypes.ANGLE,
                 grid_id: ResourceIdentifier = ResourceIdentifier(),
                 label=__default_grid_label__):
        super().__init__(network_code, data_or_dims, origin, spacing, seed,
                         velocity_model_id=velocity_model_id, phase=phase,
                         value=value, grid_type=grid_type,
                         grid_units=grid_units,
                         float_type=float_type,
                         grid_id=grid_id,
                         label=label)

    def write_nlloc(self, path='.'):
        super().write_nlloc(path=path)


class SurfaceWaveVelocity(Grid):
    def __init__(self, network_code: str, data_or_dims: Union[np.ndarray, List, Tuple],
                 period: float, phase: Phases = Phases.RAYLEIGH,
                 grid_type=GridTypes.VELOCITY_METERS, grid_units=GridUnits.METER,
                 spacing: Union[np.ndarray, List, Tuple] = None,
                 origin: Union[np.ndarray, List, Tuple] = None,
                 resource_id: ResourceIdentifier = ResourceIdentifier(),
                 value: float = 0, coordinate_system: CoordinateSystem = CoordinateSystem.NED,
                 label: str = __default_grid_label__,
                 float_type: FloatTypes = FloatTypes.FLOAT,
                 velocity_type: VelocityType = VelocityType.GROUP):
        """Initialise a phase/group velocity grid for a single period.

        :param network_code: Network code associated with the phase-velocity model.
        :type network_code: str
        :param data_or_dims: The grid values or the grid dimensions used to build the
                             underlying :class:`~uquake.grid.base.Grid`.
        :type data_or_dims: Union[np.ndarray, List, Tuple]
        :param period: Wave period used to compute the phase velocities, in seconds.
        :type period: float
        :param phase: Seismic phase for which the velocities are defined.
        :type phase: Phases
        :param velocity_type: can be either the group or the phase velocity
        :type phase: VelocityType
        :param grid_type: Storage type of the grid values.
        :type grid_type: GridTypes
        :param grid_units: Physical units of the grid's spatial axes.
        :type grid_units: GridUnits
        :param spacing: Grid spacing for each axis. If omitted, inferred from
                        ``data_or_dims`` when possible.
        :type spacing: Union[np.ndarray, List, Tuple], optional
        :param origin: Grid origin expressed in the selected coordinate system.
        :type origin: Union[np.ndarray, List, Tuple], optional
        :param resource_id: Resource identifier attached to the grid metadata.
        :type resource_id: ResourceIdentifier
        :param value: Default fill value used when constructing an empty grid.
        :type value: float
        :param coordinate_system: Coordinate system in which the grid axes are
                                  expressed.
        :type coordinate_system: CoordinateSystem
        :param label: Human-readable label attached to the grid.
        :type label: str
        :param float_type: Floating-point precision used to store grid values.
        :type float_type: FloatTypes
        """
        self.network_code = network_code
        self.period = period
        self.velocity_type = velocity_type
        self.grid_type = grid_type
        self.grid_units = grid_units
        self.phase = phase
        self.float_type = float_type

        super().__init__(data_or_dims, origin=origin, spacing=spacing,
                         resource_id=resource_id, value=value,
                         coordinate_system=coordinate_system,
                         label=label)

    @classmethod
    def from_inventory(cls, network_code: str, inventory: Inventory,
                       spacing: Union[float, Tuple[float, float]], period: float,
                       padding: Union[float, Tuple[float, float]] = 0.2,
                       phase: Phases = Phases.RAYLEIGH,
                       velocity_type: VelocityType = VelocityType.GROUP,
                       **kwargs):
        """
        Create a grid object from a given inventory.

        :param network_code: The network code associated with the inventory.
        :type network_code: str
        :param inventory: The inventory containing instrument locations.
        :type inventory: Inventory
        :param spacing: The spacing of the grid. Can be a single float or a tuple of
                        floats specifying spacing in the x and y directions.
        :type spacing: Union[float, Tuple[float, float]]
        :param period: The period associated with the grid.
        :type period: float
        :param padding: The padding to be added around the inventory span. Can be a
                        single float or a tuple of floats specifying padding in the
                         x and y directions. Default is 0.2.
        :type padding: Union[float, Tuple[float, float]], optional
        :param phase: The phase type, default is Phases.RAYLEIGH.
        :type phase: Phases, optional
        :param velocity_type: The surface wave velocity type, default is
            VelocityType.GROUP.
        :type velocity_type: VelocityType, optional
        :param kwargs: Additional keyword arguments forwarded to the constructor.
        :type kwargs: dict
        :return: A phase-velocity grid sized to the instrument coverage.
        :rtype: PhaseVelocity

        :raises ValueError: If the padding or spacing values are invalid.

        This method calculates grid dimensions and origin from the inventory span
        and requested padding before instantiating a :class:`PhaseVelocity` model.
        """

        locations_x, locations_y, _ = get_coordinates_inventory(inventory, strict=True)

        # Determine the span of the inventory
        min_coords = np.array([np.min(locations_x), np.min(locations_y)])
        max_coords = np.array([np.max(locations_x), np.max(locations_y)])
        inventory_span = max_coords - min_coords

        # Calculate padding in grid units
        if isinstance(padding, tuple):
            padding_x, padding_y = padding
        else:
            padding_x = padding_y = padding

        # Calculate the total padding to be added
        total_padding = inventory_span * np.array([padding_x, padding_y])

        # Adjust the origin and corner with the padding
        padded_origin = min_coords - total_padding / 2
        padded_corner = max_coords + total_padding / 2

        # Calculate grid dimensions
        grid_dims = np.ceil((padded_corner - padded_origin) / np.array(spacing)).astype(
            int)
        # Create and return the grid object
        # Create and return the grid object
        return cls(network_code, grid_dims, spacing=spacing, origin=padded_origin,
                   period=period, phase=phase,
                   coordinate_system=inventory[0][0].coordinates.coordinate_system,
                   **kwargs)

    @classmethod
    def _from_seismic_property_grid_ensemble(cls,
                                             seismic_param: SeismicPropertyGridEnsemble,
                                             period: float, phase: Phases,
                                             z_axis_log: bool = False,
                                             npts_log_scale: int = 30,
                                             disba_param: DisbaParam = DisbaParam(),
                                             velocity_type:
                                             VelocityType = VelocityType.GROUP,
                                             **kwargs):
        """Build a phase/group-velocity grid from a seismic property ensemble.

        :param seismic_param: Collection of seismic property grids used to derive
                              phase velocities.
        :type seismic_param: SeismicPropertyGridEnsemble
        :param period: Wave period, in seconds, at which the dispersion relation is
                       sampled.
        :type period: float
        :param phase: Seismic phase whose phase velocities will be extracted.
        :type phase: Phases
        :param z_axis_log: If ``True``, resample the vertical axis on a logarithmic
                           scale before computing dispersion curves.
        :type z_axis_log: bool, optional
        :param npts_log_scale: Number of samples to use when ``z_axis_log`` is turned on.
        :type npts_log_scale: int, optional
        :param disba_param: Numerical parameters forwarded to
                            :class:`disba.PhaseDispersion`.
        :type disba_param: DisbaParam, optional
        :param velocity_type: The surface wave velocity type, default is
            VelocityType.GROUP.
        :type velocity_type: VelocityType
        :returns: A phase-velocity grid for the requested period and phase.
        :rtype: PhaseVelocity
        """
        if z_axis_log:
            z_max = (seismic_param.spacing[2] * seismic_param.shape[2] +
                     seismic_param.origin[2])
            z = (np.logspace(0, np.log10(10 + 1), npts_log_scale) - 10 ** 0 +
                 seismic_param.origin[2]) * z_max / 10
        else:
            z = None

        surface_velocity = seismic_param.to_surface_velocities(
            periods_list=[period],
            logspace=False,
            phase=phase,
            z=z,
            disba_param=disba_param,
            velocity_type=velocity_type
    )
        surface_velocity = surface_velocity[0]

        return cls(
            network_code=seismic_param.network_code,
            data_or_dims=surface_velocity.data,
            period=period,
            phase=phase,
            grid_type=seismic_param.grid_type,
            grid_units=seismic_param.grid_units,
            spacing=(seismic_param.spacing[0], seismic_param.spacing[1]),
            origin=(seismic_param.origin[0], seismic_param.origin[1]),
            resource_id=seismic_param.resource_id,
            coordinate_system=seismic_param.coordinate_system,
            label=seismic_param.label,
            float_type=seismic_param.float_type,
            velocity_type=velocity_type,
        )

    @property
    def type_velocity(self):
        """Return the velocity type (group or phase)."""
        return self.velocity_type

    @property
    def grid_id(self):
        return self.resource_id

    def to_rgrid(self, n_secondary: Union[int, Tuple[int, int]], cell_slowness=False,
                 threads: int = 1):
        """
        Convert the current object to a rgrid object (see ttcrpy).

        :param n_secondary: Number of secondary nodes for grid refinement. If an integer
                            is provided, it is used for both dimensions. If a tuple is
                            provided, it specifies the number of nodes in the x and y
                            dimensions respectively.
        :type n_secondary: Union[int, Tuple[int, int]]
        :param cell_slowness: If True, the grid will be created with cell slowness.
                              If False, the grid will be created without cell slowness.
                              Default is False.
        :type cell_slowness: bool, optional
        :param threads: The number of threads to use for computation. Default is 1.
        :type threads: int, optional
        :return: The resulting 2D grid object from the rgrid module.
        :rtype: rgrid.Grid2d
        """

        if isinstance(n_secondary, Tuple):
            nsx = n_secondary[0]
            nsy = n_secondary[1]
        else:
            nsx = n_secondary
            nsy = n_secondary

        if cell_slowness:
            xrange = np.arange(self.origin[0] - self.spacing[0] * 0.5,
                               self.corner[0] + self.spacing[0] * 0.5,
                               self.spacing[0]).astype(np.float64)
            yrange = np.arange(self.origin[1] - self.spacing[1] * 0.5,
                               self.corner[1] + self.spacing[0] * 0.5,
                               self.spacing[1]).astype(np.float64)
            grid = rgrid.Grid2d(x=xrange, z=yrange, cell_slowness=True,
                                method='SPM', nsnx=nsx, nsnz=nsy, n_threads=threads)
            grid.set_velocity(self.data)

        else:
            xrange = np.arange(self.origin[0], self.corner[0],
                               self.spacing[0]).astype(np.float64)
            yrange = np.arange(self.origin[1], self.corner[1],
                               self.spacing[1]).astype(np.float64)
            grid = rgrid.Grid2d(x=xrange, z=yrange, cell_slowness=False,
                                method='SPM', nsnx=nsx, nsnz=nsy, n_threads=threads)
            grid.set_velocity(self.data)
        return grid

    def write(self, filename, format='VTK', **kwargs):
        """Write the grid data to a file.

        This method writes the grid data to a file in the specified format.

        :param filename: The name of the file to write the data to.
        :type filename: str
        :param format: The format of the file (default: 'VTK').
        :type format: str
        :param **kwargs: Additional keyword arguments specific to the file format.
        """
        field_name = None
        if format == 'VTK':
            field_name = f'{self.velocity_type} velocity_{self.phase.value}'

        super().write(filename, format=format, field_name=field_name, **kwargs)

    def plot(
            self,
            receivers: Optional[Union[np.ndarray, SeedEnsemble]] = None,
            fig_size: Tuple[float, float] = (10, 8),
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            mask: Optional[dict] = None,
            geographic: bool = False,
            **imshow_kwargs,
    ):
        """Plot the phase/group-velocity grid with optional receiver overlays.

        :param receivers: Receiver positions to superimpose on the map. Provide either
                          a ``(N, 2)`` array of ``(x, y)`` coordinates or a
                          :class:`SeedEnsemble`.
        :type receivers: Optional[Union[np.ndarray, SeedEnsemble]]
        :param fig_size: Size of the matplotlib figure in inches ``(width, height)``.
        :type fig_size: Tuple[float, float]
        :param vmin: Lower bound for the colour scale. Defaults to the 1st percentile
                     when omitted.
        :type vmin: Optional[float]
        :param vmax: Upper bound for the colour scale. Defaults to the 99th percentile
                     when omitted.
        :type vmax: Optional[float]
        :param mask: Definition of regions to hide, following
                     :meth:`~uquake.grid.base.Grid.masked_region_xy` conventions.
        :type mask: Optional[dict]
        :param geographic: If ``True``, align axes with easting/northing instead of
                           grid indices.
        :type geographic: bool
        :param imshow_kwargs: Extra keyword arguments forwarded to
                              :func:`matplotlib.axes.Axes.imshow`.
        :type imshow_kwargs: dict
        :returns: Tuple ``(fig, ax)`` with the generated matplotlib figure and axes.
        :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        """
        if self.data.ndim != 2:
            raise ValueError("SurfaceWaveVelocity.plot currently supports only 2D data.")

        fig, ax = plt.subplots(figsize=fig_size)
        if 'cmap' not in imshow_kwargs:
            imshow_kwargs.setdefault('cmap', 'seismic')

        axis_order_map = {
            CoordinateSystem.NED: (0, 1),
            CoordinateSystem.NEU: (0, 1),
            CoordinateSystem.ENU: (1, 0),
            CoordinateSystem.END: (1, 0),
        }

        if geographic:
            axis_order_for_plot = axis_order_map.get(
                self.coordinate_system, (0, 1)
            )
        else:
            axis_order_for_plot = (1, 0)

        row_axis, col_axis = axis_order_for_plot
        extent = (
            self.origin[col_axis],
            self.corner[col_axis],
            self.origin[row_axis],
            self.corner[row_axis],
        )

        display_data = np.transpose(self.data, axes=axis_order_for_plot)

        cax = ax.imshow(
            display_data,
            origin="lower",
            extent=extent,
            **imshow_kwargs,
        )

        if mask is not None:
            positive_mask = super().masked_region_xy(**mask, ax=ax)
            masked_data = np.where(positive_mask, self.data, np.nan)
            grid_data = np.transpose(masked_data, axes=axis_order_for_plot)

            if geographic and len(ax.images) > 1:
                ax.images.pop()
                positive_mask_plot = np.transpose(positive_mask,
                                                  axes=axis_order_for_plot)
                mask_rgba = to_rgba(mask.get('color', 'w'))
                overlay = np.ones((*positive_mask_plot.shape, 4))
                overlay[..., :3] = mask_rgba[:3]
                overlay[..., 3] = (np.logical_not(positive_mask_plot).astype(float)
                                   * mask_rgba[3])
                ax.imshow(overlay, extent=extent, origin='lower', interpolation='none')
        else:
            grid_data = display_data

        if vmin is None:
            vmin = np.nanpercentile(grid_data, 1)
        if vmax is None:
            vmax = np.nanpercentile(grid_data, 99)
        cax.set_clim(vmin, vmax)
        cb = fig.colorbar(cax)
        cb.update_normal(cax)

        if self.grid_units == GridUnits.METER:
            if geographic:
                ax.set_xlabel("Easting (m)")
                ax.set_ylabel("Northing (m)")
            else:
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")

        if self.grid_type == GridTypes.VELOCITY_METERS:
            cb.set_label(self.velocity_type.value + " VELOCITY " + self.phase.value + " (m/s)",
                         rotation=270, labelpad=10)
        elif self.grid_type == GridTypes.VELOCITY_KILOMETERS:
            cb.set_label(self.velocity_type.value + " VELOCITY " + self.phase.value + " (km/s)",
                         rotation=270, labelpad=10)

        if self.grid_units == GridUnits.KILOMETER:
            if geographic:
                ax.set_xlabel("Easting (km)")
                ax.set_ylabel("Northing (km)")
            else:
                ax.set_xlabel("X (km)")
                ax.set_ylabel("Y (km)")


        ax.set_title("Period = {0:1.2f} s".format(self.period), weight = "bold")

        if isinstance(receivers, np.ndarray):
            ax.plot(receivers[:, 0], receivers[:, 1], "s", color="yellow")

        if isinstance(receivers, SeedEnsemble):
            coordinates = receivers.locs
            ax.plot(coordinates[:, 0], coordinates[:, 1], "s", color="yellow")

        return fig, ax

    def __repr__(self):
        """Return a concise text summary of key grid attributes."""
        repr_str = """
                period :  %0.2f
                spacing: %s
                origin : %s
                shape  : %s
                """ % (self.period, self.spacing, self.origin, self.shape)
        return repr_str

    def __str__(self):
        """Alias to :meth:`__repr__` for readable printing."""
        return self.__repr__()

    def compute_frechet(self, sources: Union[SeedEnsemble, np.ndarray],
                        receivers: Union[SeedEnsemble, np.ndarray],
                        ns: Union[int, Tuple[int, int, int]] = 5,
                        tt_cal: bool = True, cell_slowness: bool = True,
                        threads: int = 1, *, method: Optional[str] = None,
                        sub_grid_resolution: float = 0.25,
                        step_fraction: float = 0.5,
                        ray_max_iter: int = 5000,
                        progress: bool = False,
                        pairwise: bool = False,
                        return_dense: bool = False,
                        batch_size: Optional[int] = None):

        """
        Calculate Frechet derivatives between sources and receivers.

        :param sources: Source locations provided either as a :class:`SeedEnsemble`
                        or as an array of ``(x, y)`` coordinates.
        :type sources: Union[SeedEnsemble, np.ndarray]
        :param receivers: Receiver locations provided either as a
                          :class:`SeedEnsemble` or an array of ``(x, y)`` coordinates.
        :type receivers: Union[SeedEnsemble, np.ndarray]
        :param ns: Number of secondary nodes when using the ``ttcrpy`` backend.
        :type ns: Union[int, Tuple[int, int, int]], optional
        :param tt_cal: If ``True`` also return the matrix of travel times.
        :type tt_cal: bool, optional
        :param cell_slowness: When ``True`` return derivatives with respect to
                              slowness; otherwise derivatives are with respect to
                              velocity.
        :type cell_slowness: bool, optional
        :param threads: Number of worker threads.
        :type threads: int, optional
        :param method: Backend used for the Frechet computation. ``"fmm"`` uses the
                       internal fast-marching solver and gradient-based ray tracing,
                       while ``"ttcrpy"`` falls back to the legacy implementation.
                       When ``None`` the solver defaults to ``"fmm"`` if
                       :mod:`scikit-fmm` is available, otherwise ``"ttcrpy"``.
        :type method: Optional[str], optional
        :param sub_grid_resolution: Fraction of the grid spacing used for the
                                    high-resolution patch surrounding each source
                                    when ``method="fmm"``.
        :type sub_grid_resolution: float, optional
        :param step_fraction: Maximum fraction of the smallest cell size employed
                               when discretising rays for path-length accumulation.
        :type step_fraction: float, optional
        :param ray_max_iter: Maximum number of iterations allowed in the
                             gradient-descent ray tracer.
        :type ray_max_iter: int, optional
        :param progress: Display a progress bar while iterating over sources.
        :type progress: bool, optional
        :param pairwise: When ``True`` treat the input as ordered source/receiver
            pairs and return a sparse ``(N_pairs, N_cells)`` sensitivity matrix
            (plus optional travel times). When ``False`` return the full
            ``(N_sources, N_receivers, N_cells)`` dense array.
        :type pairwise: bool, optional
        :param batch_size: Optional number of source/receiver pairs to process per
            batch when ``pairwise=True``. Each batch is computed independently and
            the resulting matrices are stacked. Defaults to processing all pairs at
            once.
        :returns: When ``tt_cal`` is ``True`` returns ``(frechet, travel_times)``.
                  Otherwise only the Frechet derivatives are returned. ``frechet``
                  is a CSR matrix by default unless ``return_dense`` is ``True``.
        :rtype: Union[sparse.csr_matrix, np.ndarray,
                      Tuple[sparse.csr_matrix, np.ndarray],
                      Tuple[np.ndarray, np.ndarray]]
        """

        if method is None:
            method = "fmm" if _SKFMM_AVAILABLE else "ttcrpy"
        method = method.lower()

        dims = 2  # GroupVelocity or PhaseVelocity grids are defined in (x, y)

        def _extract_coords(entity):
            if isinstance(entity, SeedEnsemble):
                return entity.locs[:, :dims]
            return np.atleast_2d(entity)[:, :dims]

        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError("batch_size must be a positive integer")
        src_coords = _extract_coords(sources)
        rcv_coords = _extract_coords(receivers)

        if batch_size is not None and pairwise and batch_size < src_coords.shape[0]:
            matrices = []
            travel_blocks = [] if tt_cal else None
            for start in range(0, src_coords.shape[0], batch_size):
                end = min(start + batch_size, src_coords.shape[0])
                sub_sources = src_coords[start:end]
                sub_receivers = rcv_coords[start:end]
                result = self.compute_frechet(
                    sub_sources,
                    sub_receivers,
                    ns=ns,
                    tt_cal=tt_cal,
                    cell_slowness=cell_slowness,
                    threads=threads,
                    method=method,
                    sub_grid_resolution=sub_grid_resolution,
                    step_fraction=step_fraction,
                    ray_max_iter=ray_max_iter,
                    progress=progress,
                    pairwise=True,
                    return_dense=return_dense,
                    batch_size=None,
                )
                if tt_cal:
                    sub_matrix, sub_tt = result
                else:
                    sub_matrix = result

                if return_dense:
                    matrices.append(np.asarray(sub_matrix))
                else:
                    matrices.append(sub_matrix.tocsr())

                if tt_cal:
                    travel_blocks.append(np.asarray(sub_tt))

            if return_dense:
                frechet_output = np.vstack(matrices)
            else:
                frechet_output = sparse.vstack(matrices, format="csr")

            if tt_cal:
                travel_output = np.concatenate(travel_blocks)
                return frechet_output, travel_output
            return frechet_output

        if batch_size is not None and not pairwise:
            raise NotImplementedError(
                "batch_size is currently only supported when pairwise=True."
            )

        unique_src_coords, src_inverse, src_rep = _deduplicate_points(src_coords)
        unique_rcv_coords, rcv_inverse, rcv_rep = _deduplicate_points(rcv_coords)

        if pairwise and src_coords.shape[0] != rcv_coords.shape[0]:
            raise ValueError("pairwise=True requires the same number of sources and receivers.")

        if isinstance(sources, SeedEnsemble):
            unique_src_seeds = [sources.seeds[idx] for idx in src_rep]
            unique_sources = SeedEnsemble(unique_src_seeds, units=sources.units)
        else:
            unique_sources = unique_src_coords

        if isinstance(receivers, SeedEnsemble):
            unique_rcv_seeds = [receivers.seeds[idx] for idx in rcv_rep]
            unique_receivers = SeedEnsemble(unique_rcv_seeds, units=receivers.units)
        else:
            unique_receivers = unique_rcv_coords

        n_sources_orig = src_coords.shape[0]
        n_receivers_orig = rcv_coords.shape[0]
        n_unique_sources = unique_src_coords.shape[0]
        n_unique_receivers = unique_rcv_coords.shape[0]

        if n_unique_sources != n_sources_orig:
            logger.info(
                f"Collapsed {n_sources_orig} sources to {n_unique_sources} unique locations."
            )
        if n_unique_receivers != n_receivers_orig:
            logger.info(
                f"Collapsed {n_receivers_orig} receivers to {n_unique_receivers} unique locations."
            )

        n_pairs = src_coords.shape[0]

        swap_axes = (not pairwise) and (n_unique_receivers < n_unique_sources)

        if swap_axes:
            logger.info(
                "Using reciprocity: computing Frechet derivatives"
                " with receivers as sources."
            )

        compute_sources = unique_receivers if swap_axes else unique_sources
        compute_receivers = unique_sources if swap_axes else unique_receivers

        def _entity_count(entity):
            return len(entity.seeds) if isinstance(entity, SeedEnsemble) else\
                np.atleast_2d(entity).shape[0]

        start_time = time.perf_counter()
        logger.info(
            f"Computing Frechet derivatives using '{method}' backend for "
            f"{_entity_count(compute_sources)} unique sources x {_entity_count(compute_receivers)} unique receivers."
        )

        if method == "fmm":
            _require_skfmm("compute Frechet derivatives via the fast marching solver")
            result = self._compute_frechet_fmm(
                sources=compute_sources,
                receivers=compute_receivers,
                tt_cal=tt_cal,
                cell_slowness=cell_slowness,
                threads=threads,
                sub_grid_resolution=sub_grid_resolution,
                step_fraction=step_fraction,
                ray_max_iter=ray_max_iter,
                progress=progress,
            )
            if tt_cal:
                frechet_unique, tt_unique = result
            else:
                frechet_unique = result
                tt_unique = None
        elif method == "ttcrpy":
            frechet_rows, tt_rows = self._compute_frechet_ttcrpy(
                sources=compute_sources,
                receivers=compute_receivers,
                ns=ns,
                tt_cal=tt_cal,
                cell_slowness=cell_slowness,
                threads=threads,
                progress=progress,
            )
            if pairwise:
                row_blocks = []
                tt_pairs = np.empty(n_pairs, dtype=float) if tt_cal else None
                for pair_idx, (src_idx, rcv_idx) in enumerate(zip(src_inverse, rcv_inverse)):
                    row_blocks.append(frechet_rows[src_idx].getrow(rcv_idx))
                    if tt_cal:
                        tt_pairs[pair_idx] = tt_rows[src_idx][rcv_idx]

                if row_blocks:
                    frechet_pairs = sparse.vstack(row_blocks, format="csr")
                else:
                    n_cells = frechet_rows[0].shape[1] if frechet_rows else 0
                    frechet_pairs = sparse.csr_matrix((0, n_cells))

                elapsed = time.perf_counter() - start_time
                logger.info(
                    f"Frechet computation ('{method}' backend) completed in {elapsed:.2f}s "
                    f"for {n_pairs} source/receiver pairs."
                )

                if return_dense:
                    frechet_dense = frechet_pairs.toarray()
                    if tt_cal:
                        return frechet_dense, np.asarray(tt_pairs, dtype=float)
                    return frechet_dense

                if tt_cal:
                    return frechet_pairs, np.asarray(tt_pairs, dtype=float)
                return frechet_pairs

            frechet_unique = np.stack([mat.toarray() for mat in frechet_rows], axis=0)
            tt_unique = np.stack(tt_rows, axis=0) if tt_cal else None
        else:
            raise ValueError("method must be either 'fmm' or 'ttcrpy'.")

        if swap_axes:
            frechet_unique = np.transpose(frechet_unique, (1, 0, 2))
            if tt_unique is not None:
                tt_unique = tt_unique.T

        if pairwise:
            frechet_pairs = frechet_unique[src_inverse, rcv_inverse, :]
            if return_dense:
                if tt_cal:
                    return frechet_pairs, tt_unique[src_inverse, rcv_inverse]
                return frechet_pairs
            frechet_pairs_matrix = sparse.csr_matrix(
                frechet_pairs.reshape(n_pairs, frechet_pairs.shape[-1])
            )
            frechet_pairs_matrix.original_shape = (n_pairs,)
            if tt_cal:
                return frechet_pairs_matrix, tt_unique[src_inverse, rcv_inverse]
            return frechet_pairs_matrix

        frechet_full = frechet_unique[src_inverse][:, rcv_inverse, :]
        if return_dense:
            if tt_cal:
                return frechet_full, tt_unique[src_inverse][:, rcv_inverse]
            return frechet_full

        frechet_matrix = sparse.csr_matrix(
            frechet_full.reshape(n_sources_orig * n_receivers_orig, frechet_full.shape[-1])
        )
        frechet_matrix.original_shape = (n_sources_orig, n_receivers_orig)
        if tt_cal:
            return frechet_matrix, tt_unique[src_inverse][:, rcv_inverse]
        return frechet_matrix

    def compute_frechet_eikonal(
            self,
            sources: Union[SeedEnsemble, np.ndarray],
            receivers: Union[SeedEnsemble, np.ndarray],
            ns: Union[int, Tuple[int, int, int]] = 5,
            tt_cal: bool = True,
            cell_slowness: bool = True,
            threads: int = 1,
            *,
            method: Optional[str] = None,
            sub_grid_resolution: float = 0.25,
            step_fraction: float = 0.5,
            ray_max_iter: int = 5_000,
            progress: bool = False,
            pairwise: bool = False,
            return_rays: bool = False,
            return_dense: bool = False,
            batch_size: Optional[int] = None,
    ):
        """Compute Frechet derivatives using the Estuary ``eikonal`` backend.

        Parameters
        ----------
        return_dense
            When ``True`` materialise the sensitivity matrix as a dense NumPy array;
            otherwise a CSR matrix is returned (the default).
        """

        if _eikonal_compute_frechet is None or EKImageData is None:
            raise ImportError(
                "The 'eikonal-ng' extensions are not available. Build them first or "
                "install the package before calling compute_frechet_eikonal()."
            )

        if return_rays:
            raise NotImplementedError(
                "return_rays=True is not supported by compute_frechet_eikonal()."
            )

        if method is not None and method.lower() not in {"eikonal", "fmm"}:
            raise ValueError(
                "method must be None or 'eikonal' for compute_frechet_eikonal().")

        if progress:
            logger.warning(
                "Progress reporting is not supported by the eikonal backend; ignoring.")

        dims = 2

        def _extract_coords(entity):
            if isinstance(entity, SeedEnsemble):
                return entity.locs[:, :dims]
            return np.atleast_2d(entity)[:, :dims]

        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError("batch_size must be a positive integer")
        src_coords = _extract_coords(sources)
        rcv_coords = _extract_coords(receivers)

        if batch_size is not None and pairwise and batch_size < src_coords.shape[0]:
            matrices = []
            travel_blocks = [] if tt_cal else None
            for start in range(0, src_coords.shape[0], batch_size):
                end = min(start + batch_size, src_coords.shape[0])
                sub_sources = src_coords[start:end]
                sub_receivers = rcv_coords[start:end]
                result = self.compute_frechet_eikonal(
                    sub_sources,
                    sub_receivers,
                    ns=ns,
                    tt_cal=tt_cal,
                    cell_slowness=cell_slowness,
                    threads=threads,
                    method=method,
                    sub_grid_resolution=sub_grid_resolution,
                    step_fraction=step_fraction,
                    ray_max_iter=ray_max_iter,
                    progress=progress,
                    pairwise=True,
                    return_rays=False,
                    return_dense=return_dense,
                    batch_size=None,
                )
                if tt_cal:
                    sub_matrix, sub_tt = result
                else:
                    sub_matrix = result

                if return_dense:
                    matrices.append(np.asarray(sub_matrix))
                else:
                    matrices.append(sub_matrix.tocsr())

                if tt_cal:
                    travel_blocks.append(np.asarray(sub_tt))

            if return_dense:
                frechet_output = np.vstack(matrices)
            else:
                frechet_output = sparse.vstack(matrices, format="csr")

            if tt_cal:
                travel_output = np.concatenate(travel_blocks)
                return frechet_output, travel_output
            return frechet_output

        if batch_size is not None and not pairwise:
            raise NotImplementedError(
                "batch_size is currently only supported when pairwise=True."
            )

        if pairwise and src_coords.shape[0] != rcv_coords.shape[0]:
            raise ValueError(
                "pairwise=True requires the same number of sources and receivers.")

        unique_src_coords, src_inverse, _ = _deduplicate_points(src_coords)
        unique_rcv_coords, rcv_inverse, _ = _deduplicate_points(rcv_coords)

        n_sources = src_coords.shape[0]
        n_receivers = rcv_coords.shape[0]

        if pairwise:
            unique_src_coords = src_coords
            unique_rcv_coords = rcv_coords
            src_inverse = np.arange(n_sources, dtype=int)
            rcv_inverse = np.arange(n_receivers, dtype=int)

        _ = (ns, sub_grid_resolution, ray_max_iter)
        if threads not in (1, None):  # pragma: no cover - advisory warning
            warnings.warn(
                "The 'threads' parameter is ignored by compute_frechet_eikonal().",
                RuntimeWarning,
                stacklevel=2,
            )

        spacing_arr = np.atleast_1d(np.array(self.spacing, dtype=float))
        if spacing_arr.size < dims:
            spacing_arr = np.repeat(spacing_arr[0], dims)
        spacing_arr = spacing_arr[:dims]
        if not np.allclose(spacing_arr, spacing_arr[0]):
            raise ValueError("compute_frechet_eikonal requires isotropic grid spacing.")
        spacing_value = float(spacing_arr[0])

        origin_arr = np.atleast_1d(np.array(self.origin, dtype=float))
        if origin_arr.size < dims:
            origin_arr = np.concatenate([origin_arr, np.zeros(dims - origin_arr.size)])
        origin_vec = origin_arr[:dims]

        velocity_data = np.asarray(self.data, dtype=float)
        if np.any(velocity_data <= 0):
            raise ValueError(
                "Velocity grid must be strictly positive for eikonal computations.")

        if self.grid_type == GridTypes.VELOCITY_KILOMETERS and self.grid_units == GridUnits.METER:
            velocity = velocity_data.copy() * 1e3  # convert to m/s

        elif self.grid_type == GridTypes.VELOCITY_METERS and self.grid_units == GridUnits.KILOMETER:
            velocity = velocity_data.copy() * 1e-3  # convert to km/s

        else:
            raise ValueError("Grid type not supported")

        velocity_grid = EKImageData(velocity, origin=tuple(origin_vec),
                                    spacing=spacing_value)

        # unique_src_grid = self.transform_to_grid(unique_src_coords)
        # unique_rcv_grid = self.transform_to_grid(unique_rcv_coords)
        self.transform_from_grid(unique_src_coords)

        frechet_result = _eikonal_compute_frechet(
            velocity=velocity_grid,
            sources=unique_src_coords,
            receivers=unique_rcv_coords,
            spacing=spacing_value,
            origin=origin_vec,
            rk_step=step_fraction,
            second_order=True,
            cell_slowness=cell_slowness,
            return_travel_times=tt_cal,
            pairwise=pairwise,
            return_rays=False,
            dtype=float,
            return_dense=return_dense,
        )
        if tt_cal:
            frechet_output, travel_output = frechet_result
        else:
            frechet_output = frechet_result
            travel_output = None

        if not return_dense and not sparse.isspmatrix_csr(frechet_output):
            frechet_output = sparse.csr_matrix(frechet_output)

        if travel_output is not None:
            return frechet_output, travel_output
        return frechet_output

    def _compute_frechet_ttcrpy(self,
                                sources: Union[SeedEnsemble, np.ndarray],
                                receivers: Union[SeedEnsemble, np.ndarray],
                                ns: Union[int, Tuple[int, int, int]],
                                tt_cal: bool,
                                cell_slowness: bool,
                                threads: int,
                                progress: bool):

        if isinstance(sources, SeedEnsemble):
            srcs = sources.locs[:, :2]
        else:
            srcs = sources
        srcs = np.atleast_2d(srcs)

        if isinstance(receivers, SeedEnsemble):
            rxs = receivers.locs[:, :2]
        else:
            rxs = receivers
        rxs = np.atleast_2d(rxs)

        n_sources = srcs.shape[0]
        n_receivers = rxs.shape[0]

        worker_threads = threads if isinstance(threads, int) and threads > 0 else 1
        grid = self.to_rgrid(n_secondary=ns, cell_slowness=cell_slowness,
                             threads=worker_threads)
        if hasattr(grid, "set_use_thread_pool"):
            grid.set_use_thread_pool(worker_threads > 1)

        tt_rows = []
        frechet_rows = []

        pbar = tqdm(total=n_sources, desc="Frechet (ttcrpy)", unit="src",
                    disable=not progress)
        try:
            for src in srcs:
                single_src = np.asarray(src, dtype=float).reshape(1, -1)
                tt_single, _, frechet_single = grid.raytrace(
                    source=single_src,
                    rcv=rxs,
                    compute_L=True,
                    return_rays=True,
                )

                tt_processed = np.asarray(tt_single).reshape(n_receivers)

                if sparse.issparse(frechet_single):
                    frechet_processed = frechet_single.tocsr()
                else:
                    frechet_array = np.asarray(frechet_single)
                    if frechet_array.size == 0:
                        frechet_processed = sparse.csr_matrix((n_receivers, 0))
                    else:
                        if frechet_array.ndim == 1:
                            if n_receivers != 1:
                                raise ValueError(
                                    "Unexpected Frechet shape for multiple receivers."
                                )
                            frechet_array = frechet_array.reshape(1, -1)
                        if frechet_array.shape[0] != n_receivers:
                            raise ValueError(
                                f"Inconsistent Frechet derivative shape {frechet_array.shape} "
                                f"for {n_receivers} receivers."
                            )
                        frechet_processed = sparse.csr_matrix(frechet_array)

                tt_rows.append(tt_processed)
                frechet_rows.append(frechet_processed)

                pbar.update()
        finally:
            pbar.close()

        if tt_cal:
            return frechet_rows, tt_rows
        return frechet_rows, tt_rows

    def _compute_frechet_fmm(self,
                              sources: Union[SeedEnsemble, np.ndarray],
                              receivers: Union[SeedEnsemble, np.ndarray],
                              tt_cal: bool,
                              cell_slowness: bool,
                              threads: int,
                              sub_grid_resolution: float,
                              step_fraction: float,
                              ray_max_iter: int,
                              progress: bool):

        if isinstance(sources, SeedEnsemble):
            seeds = sources.seeds
        else:
            seeds = self._build_seeds_from_array(np.atleast_2d(sources))

        if isinstance(receivers, SeedEnsemble):
            receiver_coords = receivers.locs
        else:
            receiver_coords = np.atleast_2d(receivers)

        if len(seeds) == 0 or receiver_coords.size == 0:
            raise ValueError("Both sources and receivers must be provided to compute Frechet derivatives.")

        n_sources = len(seeds)
        n_receivers = receiver_coords.shape[0]
        n_cells = int(self.data.size)

        dtype = np.dtype(self.float_type.value)
        tt = np.zeros((n_sources, n_receivers), dtype=np.float64)
        frechet = np.zeros((n_sources, n_receivers, n_cells), dtype=np.float64)

        velocity_flat = self.data.ravel(order='C').astype(np.float64)
        dims = self.ndim

        def _process_source(args):
            idx, seed = args
            tt_grid = self._build_travel_time_grid_fmm(
                seed,
                sub_grid_resolution=sub_grid_resolution,
            )

            source_tt = np.zeros(n_receivers, dtype=np.float64)
            source_frechet = np.zeros((n_receivers, n_cells), dtype=np.float64)
            seed_loc = np.asarray(seed.loc)

            for j, rec in enumerate(receiver_coords):
                receiver_point = self._prepare_receiver_point(rec, seed_loc, dims)
                if not self.in_grid(receiver_point[:dims], grid_space=False):
                    raise ValueError(f"Receiver {receiver_point[:dims]} is outside the grid bounds.")

                ray = tt_grid.ray_tracer(receiver_point, grid_space=False, max_iter=ray_max_iter)
                if ray is None or getattr(ray, 'nodes', None) is None or len(ray.nodes) < 2:
                    raise RuntimeError("Ray tracing failed to converge for receiver {0}.".format(receiver_point))

                path_nodes = np.asarray(ray.nodes)
                frechet_row = self.path_lengths_from_nodes(path_nodes, step_fraction=step_fraction)

                if not cell_slowness:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        frechet_row = -frechet_row / np.square(velocity_flat)
                        frechet_row[~np.isfinite(frechet_row)] = 0.0

                source_frechet[j, :] = frechet_row

                if hasattr(ray, 'travel_time') and ray.travel_time is not None:
                    source_tt[j] = ray.travel_time
                else:
                    source_tt[j] = tt_grid.interpolate(receiver_point, grid_space=False, order=1)[0]

            return idx, source_frechet, source_tt

        tasks = list(enumerate(seeds))
        max_workers = max(1, int(threads))

        pbar = tqdm(total=n_sources, desc="Frechet (fmm)", unit="src",
                    disable=not progress)
        try:
            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = []
                    for res in executor.map(_process_source, tasks):
                        results.append(res)
                        pbar.update()
            else:
                results = []
                for task in tasks:
                    results.append(_process_source(task))
                    pbar.update()
        finally:
            pbar.close()

        for idx, source_frechet, source_tt in results:
            frechet[idx, :, :] = source_frechet
            tt[idx, :] = source_tt

        frechet = frechet.astype(dtype, copy=False)
        if tt_cal:
            return frechet, tt.astype(dtype, copy=False)
        return frechet

    def _prepare_receiver_point(self, receiver: np.ndarray, seed_loc: np.ndarray,
                                 dims: int) -> np.ndarray:
        receiver = np.asarray(receiver, dtype=float)
        if receiver.size < dims:
            receiver = np.pad(receiver, (0, dims - receiver.size), constant_values=0.0)
        point = seed_loc.copy()
        point[:dims] = receiver[:dims]
        return point

    def _build_seeds_from_array(self, srcs: np.ndarray) -> List[Seed]:
        seeds = []
        dims = self.ndim
        for idx, src in enumerate(np.asarray(srcs, dtype=float)):
            if src.size < dims:
                src = np.pad(src, (0, dims - src.size), constant_values=0.0)
            coords = np.zeros(3, dtype=float)
            coords[:dims] = src[:dims]
            coordinates = Coordinates(coords[0], coords[1], coords[2],
                                      coordinate_system=self.coordinate_system)
            seeds.append(Seed(f'S{idx:04d}', f'L{idx:04d}', coordinates))
        return seeds

    def _build_travel_time_grid_fmm(self, seed: Seed,
                                    sub_grid_resolution: float = 0.25) -> TTGrid:
        _require_skfmm("build auxiliary travel-time grids using the fast marching solver")

        if not self.in_grid(seed.loc[:self.ndim], grid_space=False):
            raise ValueError(f'Source {seed.label} lies outside the grid bounds.')

        origin = np.asarray(self.origin, dtype=float)
        spacing = np.asarray(self.spacing, dtype=float)
        shape = np.asarray(self.shape, dtype=int)
        dims = self.ndim

        sub_spacing = spacing * float(sub_grid_resolution)
        n_pts_inner = np.maximum(4, (4 * spacing / sub_spacing * 1.2).astype(int))
        for dim in range(dims):
            if n_pts_inner[dim] % 2:
                n_pts_inner[dim] += 1

        seed_coords = np.asarray(seed.loc)[:dims]

        local_axes = []
        for dim in range(dims):
            axis = np.arange(n_pts_inner[dim], dtype=float) * sub_spacing[dim]
            axis = axis - np.mean(axis) + seed_coords[dim]
            local_axes.append(axis)

        local_mesh = np.meshgrid(*local_axes, indexing='ij')
        local_coords = np.stack([m.ravel() for m in local_mesh], axis=1)
        local_vel = self.interpolate(local_coords, grid_space=False).reshape(
            [len(axis) for axis in local_axes]
        )

        phi_local = np.ones_like(local_mesh[0])
        centre_idx = tuple(int(np.floor(len(axis) / 2)) for axis in local_axes)
        phi_local[centre_idx] = 0.0

        tt_local = skfmm.travel_time(phi_local, local_vel, dx=tuple(sub_spacing))

        local_origin = [axis[0] for axis in local_axes]
        tt_local_grid = TTGrid(
            self.network_code,
            tt_local,
            local_origin,
            sub_spacing,
            seed,
            phase=self.phase,
            float_type=self.float_type,
            grid_units=self.grid_units,
            velocity_model_id=self.grid_id,
            label=self.label,
        )

        global_axes = [origin[dim] + np.arange(shape[dim], dtype=float) * spacing[dim]
                       for dim in range(dims)]
        global_mesh = np.meshgrid(*global_axes, indexing='ij')
        global_coords = np.stack([m.ravel() for m in global_mesh], axis=1)

        corner_min = np.array([np.min(axis) for axis in local_axes])
        corner_max = np.array([np.max(axis) for axis in local_axes])

        mask = np.ones(global_coords.shape[0], dtype=bool)
        for dim in range(dims):
            mask &= (global_coords[:, dim] >= corner_min[dim]) & (global_coords[:, dim] <= corner_max[dim])

        tt_interp = tt_local_grid.interpolate(global_coords[mask], grid_space=False, order=3)[0]
        bias = float(np.max(tt_interp)) if tt_interp.size else 0.0

        phi_global = np.ones(np.prod(shape), dtype=float)
        phi_global[mask] = tt_interp - bias
        phi_global = phi_global.reshape(tuple(shape))

        tt_full = skfmm.travel_time(phi_global, self.data, dx=tuple(spacing))
        tt_flat = tt_full.ravel() + bias
        tt_flat[mask] = tt_interp
        tt_full = tt_flat.reshape(tuple(shape))

        if dims == 2:
            tt_full = tt_full[:, :, np.newaxis]
            origin_tt = np.concatenate([origin, [0.0]])
            spacing_tt = np.concatenate([spacing, [np.mean(spacing)]])
        else:
            origin_tt = origin
            spacing_tt = spacing

        tt_grid = TTGrid(
            self.network_code,
            tt_full.astype(self.float_type.value),
            origin_tt,
            spacing_tt,
            seed,
            phase=self.phase,
            float_type=self.float_type,
            grid_units=self.grid_units,
            velocity_model_id=self.grid_id,
            label=self.label,
        )

        tt_grid.data -= tt_grid.interpolate(seed.T, grid_space=False, order=3)[0]
        return tt_grid

    def to_time(self, seed: Seed, ns: Union[int, Tuple[int, int, int]] = 5):
        """Generate a travel-time grid for a single source location.

        :param seed: Source definition used as the emitter in the travel-time
                     computation.
        :type seed: Seed
        :param ns: Number of secondary nodes used when constructing the auxiliary
                   ray-tracing grid.
        :type ns: Union[int, Tuple[int, int, int]], optional
        :returns: Travel-time grid derived from the current phase-velocity model.
        :rtype: TTGrid
        """
        grid = self.to_rgrid(n_secondary=ns, cell_slowness=False)
        if self.grid_type == GridTypes.VELOCITY_KILOMETERS:
            grid.set_velocity(1.e3 * self.data)
        else:
            grid.set_velocity(self.data)
        src = np.array([[seed.x, seed.y]])
        grid.raytrace(src, src)
        tt_nodes = grid.get_grid_traveltimes()
        tt_out_grid = TTGrid(self.network_code, data_or_dims=tt_nodes,
                             origin=self.origin, spacing=self.spacing,
                             seed=seed, phase=self.phase,
                             float_type=self.float_type,
                             grid_units=self.grid_units,
                             velocity_model_id=self.grid_id,
                             label=self.label)
        return tt_out_grid

    def to_time_multi_threaded(self, seeds: SeedEnsemble,
                               ns: Union[int, Tuple[int, int, int]] = 5):
        """Generate travel-time grids for multiple sources using thread-level parallelism.

        :param seeds: Collection of source definitions to evaluate.
        :type seeds: SeedEnsemble
        :param ns: Number of secondary nodes used when constructing the auxiliary
                   ray-tracing grid.
        :type ns: Union[int, Tuple[int, int, int]], optional
        :returns: Travel-time grids indexed by the order of the provided seeds.
        :rtype: TravelTimeEnsemble
        """
        grid = self.to_rgrid(n_secondary=ns, threads=len(seeds), cell_slowness=False)
        grid.set_velocity(self.data)
        src = np.zeros(shape=(len(seeds), 2))
        if self.grid_type == GridTypes.VELOCITY_KILOMETERS:
            grid.set_velocity(1.e3 * self.data)
        else:
            grid.set_velocity(self.data)
        for n, s in enumerate(seeds):
            src[n, 0] = s.x
            src[n, 1] = s.y
        grid.set_use_thread_pool(False)
        grid.raytrace(src, src)
        tt_ensemble = []
        for n, s in enumerate(seeds):
            tt_nodes = grid.get_grid_traveltimes(thread_no=n).astype(
                self.float_type.value)
            tt_ensemble.append(TTGrid(self.network_code, data_or_dims=tt_nodes,
                                      origin=self.origin, spacing=self.spacing,
                                      seed=s, phase=self.phase,
                                      float_type=self.float_type,
                                      grid_units=self.grid_units,
                                      velocity_model_id=self.grid_id,
                                      label=self.label))
        return TravelTimeEnsemble(tt_ensemble)

    def __save_rays_vtk__(self, rays, filename):
        """Export traced rays as polylines in VTK format.

        :param rays: Iterable of arrays describing individual ray paths as ``(x, y)``
                     sample coordinates.
        :type rays: Sequence[np.ndarray]
        :param filename: Destination filename passed to :mod:`evtk` (extension optional).
        :type filename: str
        :returns: None
        :rtype: None
        """
        n_rays = len(rays)
        points_per_line = np.zeros(n_rays)
        x = []
        y = []
        for n in range(n_rays):
            points_per_line[n] = rays[n].shape[0]
            points = rays[n]
            x = x + list(points[:, 0])
            y = y + list(points[:, 1])
        z = np.zeros(len(x))
        x = np.array(x)
        y = np.array(y)
        hl.polyLinesToVTK(filename, x, y, z, pointsPerLine=points_per_line)

    def raytracing(self, receivers, method="SPM", save_rays: bool = False,
                   save_tt_grid: list=[], folder=None):
        """Trace inter-receiver rays across the phase-velocity grid.

        :param receivers: Receiver locations as a ``(N, 2)`` or ``(N, 3)`` array; the
                           third coordinate is ignored if provided.
        :type receivers: np.ndarray
        :param method: Ray-tracing method name understood by :mod:`ttcrpy`.
        :type method: str
        :param save_rays: If ``True``, export the traced rays using
                          :meth:`__save_rays_vtk__`.
        :type save_rays: bool
        :param save_tt_grid: Optional arguments controlling the saving of the travel-time
                              grid. The content is passed directly to :mod:`ttcrpy`.
        :type save_tt_grid: list
        :param folder: Output folder used when persisting rays or travel times.
        :type folder: Optional[str]
        :returns: None
        :rtype: None
        """

        if folder is None:
            folder = "."
        if receivers.shape[1] == 3:
            receivers = receivers[:, :2]
        n_rcv = receivers.shape[0]
        rcv = np.zeros((n_rcv * (n_rcv - 1) // 2, 2))
        src = np.zeros((n_rcv * (n_rcv - 1) // 2, 2))
        n1 = 0
        n2 = n_rcv - 1
        for n in range(n_rcv):
            rcv[n1:n2, :] = receivers[n + 1:n_rcv, :]
            src[n1:n2, :] = receivers[n, :]
            n1 += n_rcv - n - 1
            n2 = n1 + n_rcv - n - 2
        # build a 2d rgrid
        xrange = np.arange(self.origin[0], self.corner[0], self.spacing[0])
        yrange = np.arange(self.origin[1], self.corner[1], self.spacing[1])
        grid = rgrid.Grid2d(x=xrange, z=yrange, method=method, nsnx=25,
                            nsnz=25, n_threads=n_rcv-1, cell_slowness=False)
        grid.set_velocity(self.data)
        if save_rays:
            tt, rays = grid.raytrace(src, rcv, return_rays=True)
            self.__save_rays_vtk__(rays=rays,
                                   filename=folder + "rays_period{0:2.2f}".format(
                                       self.period))
        else:
            tt = grid.raytrace(src, rcv, return_rays=False)

        if save_tt_grid:
            tt = grid.get_grid_traveltimes(thread_no=0)
            grid_tt = Grid(data_or_dims=tt, origin=self.origin, spacing=self.spacing,
                           resource_id=self.resource_id,
                           coordinate_system=self.coordinate_system, label=self.label)
            grid_tt.write(filename=folder + "tt_rcv1", format='VTK',
                          field_name="travel_time")


class PhaseVelocity(SurfaceWaveVelocity):
    def __init__(self, network_code: str, data_or_dims: Union[np.ndarray, List, Tuple],
                 period: float, phase: Phases = Phases.RAYLEIGH,
                 grid_type=GridTypes.VELOCITY_METERS, grid_units=GridUnits.METER,
                 spacing: Union[np.ndarray, List, Tuple] = None,
                 origin: Union[np.ndarray, List, Tuple] = None,
                 resource_id: ResourceIdentifier = ResourceIdentifier(),
                 value: float = 0, coordinate_system: CoordinateSystem = CoordinateSystem.NED,
                 label: str = __default_grid_label__,
                 float_type: FloatTypes = FloatTypes.FLOAT, **kwargs):
        """Initialise a phase-velocity grid for a single period.

        :param network_code: Network code associated with the phase-velocity model.
        :type network_code: str
        :param data_or_dims: The grid values or the grid dimensions used to build the
                             underlying :class:`~uquake.grid.base.Grid`.
        :type data_or_dims: Union[np.ndarray, List, Tuple]
        :param period: Wave period used to compute the phase velocities, in seconds.
        :type period: float
        :param phase: Seismic phase for which the velocities are defined.
        :type phase: Phases
        :param grid_type: Storage type of the grid values.
        :type grid_type: GridTypes
        :param grid_units: Physical units of the grid's spatial axes.
        :type grid_units: GridUnits
        :param spacing: Grid spacing for each axis. If omitted, inferred from
                        ``data_or_dims`` when possible.
        :type spacing: Union[np.ndarray, List, Tuple], optional
        :param origin: Grid origin expressed in the selected coordinate system.
        :type origin: Union[np.ndarray, List, Tuple], optional
        :param resource_id: Resource identifier attached to the grid metadata.
        :type resource_id: ResourceIdentifier
        :param value: Default fill value used when constructing an empty grid.
        :type value: float
        :param coordinate_system: Coordinate system in which the grid axes are
                                  expressed.
        :type coordinate_system: CoordinateSystem
        :param label: Human-readable label attached to the grid.
        :type label: str
        :param float_type: Floating-point precision used to store grid values.
        :type float_type: FloatTypes
        """
        super().__init__(
            network_code=network_code,
            data_or_dims=data_or_dims,
            period=period,
            phase=phase,
            velocity_type=VelocityType.PHASE,
            grid_type=grid_type,
            grid_units=grid_units,
            spacing=spacing,
            origin=origin,
            resource_id=resource_id,
            value=value,
            coordinate_system=coordinate_system,
            label=label,
            float_type=float_type,
        )

    @classmethod
    def from_inventory(cls, network_code: str, inventory: Inventory,
                       spacing: Union[float, Tuple[float, float]], period: float,
                       padding: Union[float, Tuple[float, float]] = 0.2,
                       phase: Phases = Phases.RAYLEIGH,
                       **kwargs):
        """
        Create a grid object from a given inventory.

        :param network_code: The network code associated with the inventory.
        :type network_code: str
        :param inventory: The inventory containing instrument locations.
        :type inventory: Inventory
        :param spacing: The spacing of the grid. Can be a single float or a tuple of
                        floats specifying spacing in the x and y directions.
        :type spacing: Union[float, Tuple[float, float]]
        :param period: The period associated with the grid.
        :type period: float
        :param padding: The padding to be added around the inventory span. Can be a
                        single float or a tuple of floats specifying padding in the
                         x and y directions. Default is 0.2.
        :type padding: Union[float, Tuple[float, float]], optional
        :param phase: The phase type, default is Phases.RAYLEIGH.
        :type phase: Phases, optional
        :param kwargs: Additional keyword arguments forwarded to the constructor.
        :type kwargs: dict
        :return: A phase-velocity grid sized to the instrument coverage.
        :rtype: PhaseVelocity

        :raises ValueError: If the padding or spacing values are invalid.

        This method calculates grid dimensions and origin from the inventory span
        and requested padding before instantiating a :class:`PhaseVelocity` model.
        """
        return super().from_inventory(network_code=network_code, inventory=inventory,
                                      spacing=spacing, period=period,
                                      padding=padding,
                                      phase=phase,
                                      velocity_type=VelocityType.PHASE,
                                      **kwargs)

    @classmethod
    def from_seismic_property_grid_ensemble(cls,
                                            seismic_param: SeismicPropertyGridEnsemble,
                                            period: float, phase: Phases,
                                            z_axis_log:bool = False,
                                            npts_log_scale: int = 30,
                                            disba_param: DisbaParam = DisbaParam(),
                                            ):
        """Build a phase-velocity grid from a seismic property ensemble.

        :param seismic_param: Collection of seismic property grids used to derive
                              phase velocities.
        :type seismic_param: SeismicPropertyGridEnsemble
        :param period: Wave period, in seconds, at which the dispersion relation is
                       sampled.
        :type period: float
        :param phase: Seismic phase whose phase velocities will be extracted.
        :type phase: Phases
        :param z_axis_log: If ``True``, resample the vertical axis on a logarithmic
                           scale before computing dispersion curves.
        :type z_axis_log: bool, optional
        :param npts_log_scale: Number of samples to use when ``z_axis_log`` is turned on.
        :type npts_log_scale: int, optional
        :param disba_param: Numerical parameters forwarded to
                            :class:`disba.PhaseDispersion`.
        :type disba_param: DisbaParam, optional
        :returns: A phase-velocity grid for the requested period and phase.
        :rtype: PhaseVelocity
        """
        return super()._from_seismic_property_grid_ensemble(
            seismic_param=seismic_param,
            period=period,
            z_axis_log=z_axis_log,
            npts_log_scale=npts_log_scale,
            disba_param=disba_param,
            phase=phase,
            velocity_type=VelocityType.PHASE
        )


class GroupVelocity(SurfaceWaveVelocity):
    def __init__(self, network_code: str, data_or_dims: Union[np.ndarray, List, Tuple],
                 period: float, phase: Phases = Phases.RAYLEIGH,
                 grid_type=GridTypes.VELOCITY_METERS, grid_units=GridUnits.METER,
                 spacing: Union[np.ndarray, List, Tuple] = None,
                 origin: Union[np.ndarray, List, Tuple] = None,
                 resource_id: ResourceIdentifier = ResourceIdentifier(),
                 value: float = 0, coordinate_system: CoordinateSystem = CoordinateSystem.NED,
                 label: str = __default_grid_label__,
                 float_type: FloatTypes = FloatTypes.FLOAT, **kwargs):
        """Initialise a group-velocity grid for a single period.

        :param network_code: Network code associated with the group-velocity model.
        :type network_code: str
        :param data_or_dims: The grid values or the grid dimensions used to build the
                             underlying :class:`~uquake.grid.base.Grid`.
        :type data_or_dims: Union[np.ndarray, List, Tuple]
        :param period: Wave period used to compute the group velocities, in seconds.
        :type period: float
        :param phase: Seismic phase for which the velocities are defined.
        :type phase: Phases
        :param grid_type: Storage type of the grid values.
        :type grid_type: GridTypes
        :param grid_units: Physical units of the grid's spatial axes.
        :type grid_units: GridUnits
        :param spacing: Grid spacing for each axis. If omitted, inferred from
                        ``data_or_dims`` when possible.
        :type spacing: Union[np.ndarray, List, Tuple], optional
        :param origin: Grid origin expressed in the selected coordinate system.
        :type origin: Union[np.ndarray, List, Tuple], optional
        :param resource_id: Resource identifier attached to the grid metadata.
        :type resource_id: ResourceIdentifier
        :param value: Default fill value used when constructing an empty grid.
        :type value: float
        :param coordinate_system: Coordinate system in which the grid axes are
                                  expressed.
        :type coordinate_system: CoordinateSystem
        :param label: Human-readable label attached to the grid.
        :type label: str
        :param float_type: Floating-point precision used to store grid values.
        :type float_type: FloatTypes
        """
        super().__init__(
            network_code=network_code,
            data_or_dims=data_or_dims,
            period=period,
            phase=phase,
            velocity_type=VelocityType.GROUP,
            grid_type=grid_type,
            grid_units=grid_units,
            spacing=spacing,
            origin=origin,
            resource_id=resource_id,
            value=value,
            coordinate_system=coordinate_system,
            label=label,
            float_type=float_type,
        )

    @classmethod
    def from_inventory(cls, network_code: str, inventory: Inventory,
                       spacing: Union[float, Tuple[float, float]], period: float,
                       padding: Union[float, Tuple[float, float]] = 0.2,
                       phase: Phases = Phases.RAYLEIGH,
                       **kwargs):
        """
        Create a group-velocity grid object sized to the instrument coverage.

        :param network_code: The network code associated with the inventory.
        :type network_code: str
        :param inventory: The inventory containing instrument locations.
        :type inventory: Inventory
        :param spacing: The spacing of the grid. Can be a single float or a tuple of
                        floats specifying spacing in the x and y directions.
        :type spacing: Union[float, Tuple[float, float]]
        :param period: The period associated with the grid.
        :type period: float
        :param padding: The padding to be added around the inventory span. Can be a
                        single float or a tuple specifying padding in x and y. Default 0.2.
        :type padding: Union[float, Tuple[float, float]], optional
        :param phase: The phase type, default is Phases.RAYLEIGH.
        :type phase: Phases, optional
        :param kwargs: Additional keyword arguments forwarded to the constructor.
        :type kwargs: dict
        :return: A group-velocity grid sized to the instrument coverage.
        :rtype: GroupVelocity
        """
        return super().from_inventory(
            network_code=network_code,
            inventory=inventory,
            spacing=spacing,
            period=period,
            padding=padding,
            phase=phase,
            velocity_type=VelocityType.GROUP,
            **kwargs,
        )

    @classmethod
    def from_seismic_property_grid_ensemble(cls,
                                            seismic_param: SeismicPropertyGridEnsemble,
                                            period: float, phase: Phases,
                                            z_axis_log: bool = False,
                                            npts_log_scale: int = 30,
                                            disba_param: DisbaParam = DisbaParam(),
                                            **kwargs):
        """Build a group-velocity grid from a seismic property ensemble.

        :param seismic_param: Collection of seismic property grids used to derive velocities.
        :type seismic_param: SeismicPropertyGridEnsemble
        :param period: Wave period (s) at which the dispersion relation is sampled.
        :type period: float
        :param phase: Seismic phase whose group velocities will be extracted.
        :type phase: Phases
        :param z_axis_log: If True, resample the vertical axis on a logarithmic scale.
        :type z_axis_log: bool, optional
        :param npts_log_scale: Number of samples if ``z_axis_log`` is True.
        :type npts_log_scale: int, optional
        :param disba_param: Numerical parameters forwarded to disba.PhaseDispersion.
        :type disba_param: DisbaParam, optional
        :returns: A group-velocity grid for the requested period and phase.
        :rtype: GroupVelocity
        """
        return super()._from_seismic_property_grid_ensemble(
            seismic_param=seismic_param,
            period=period,
            phase=phase,
            z_axis_log=z_axis_log,
            npts_log_scale=npts_log_scale,
            disba_param=disba_param,
            velocity_type=VelocityType.GROUP,
        )


class SurfaceVelocityEnsemble(list):
    """Represents an ensemble of Phase or Group Velocity instances.

     This class extends the built-in list class to represent an ensemble \
      of PhaseVelocity instances.

     """
    def __init__(self, velocity_type: VelocityType = VelocityType.GROUP, *args):
        self.velocity_type = velocity_type
        super().__init__(*args)

    def append(self, surface_velocity):
        if isinstance(surface_velocity, SurfaceWaveVelocity):
            super().append(surface_velocity)
        else:
            print("Only instances of the SurfaceVelocity class can be added to the "
                  "list.")

    def add_surface_velocity(self, surface_velocity):
        self.append(surface_velocity)

    @classmethod
    def _from_seismic_property_grid_ensemble(
            cls, seismic_properties: SeismicPropertyGridEnsemble,
            periods: list, phase: Phases = Phases.RAYLEIGH, z_axis_log:bool = False,
            npts_log_scale: int = 30, disba_param: DisbaParam = DisbaParam(),
            type_velocity: VelocityType = VelocityType.GROUP
    ):
        """
        Construct a SurfaceVelocityEnsemble from a SeismicPropertyGridEnsemble.

        This class method generates a :class:`SurfaceVelocityEnsemble` instance from
        a provided :class:`SeismicPropertyGridEnsemble`. It computes dispersion curves
        for the specified seismic phase and associates them with the given periods.
        The method supports both linear and logarithmic sampling of the depth axis.

        Parameters
        ----------
        seismic_properties : SeismicPropertyGridEnsemble
            The seismic property ensemble (Vs, Vp, density profiles) used as the
            input model for computing surface velocities.
        periods : list of float
            A list of periods (in seconds) at which the velocity is evaluated.
        phase : Phases, optional
            The seismic wave phase to consider (default: ``Phases.RAYLEIGH``).
        z_axis_log : bool, optional
            If ``True``, the depth axis is sampled logarithmically (default: ``False``).
        npts_log_scale : int, optional
            Number of points to use when applying logarithmic depth scaling
            (default: ``30``). Only relevant if ``z_axis_log=True``.
        disba_param : DisbaParam, optional
            Parameters controlling the numerical solver (default: ``DisbaParam()``).
        type_velocity : VelocityType, optional
            The type of velocity to compute, either ``VelocityType.GROUP`` or
            ``VelocityType.PHASE`` (default: ``VelocityType.GROUP``).

        Returns
        -------
        SurfaceVelocityEnsemble
            An ensemble of surface velocities associated with the given seismic
            property grid ensemble and computation settings.

        """
        if z_axis_log:
            z_max = (seismic_properties.spacing[2] * seismic_properties.shape[2] +
                     seismic_properties.origin[2])
            z = (np.logspace(0, np.log10(10 + 1), npts_log_scale) - 10 ** 0 +
                 seismic_properties.origin[2]) * z_max / 10
        else:
            z = None

        cls_obj = seismic_properties.to_surface_velocities(
            periods_list=periods,
            phase=phase,
            multithreading=True,  # kept for API; ignored (runs serial)
            z=z,
            disba_param=disba_param,
            velocity_type=type_velocity,
        )

        return cls_obj

    @property
    def periods(self):
        periods = []
        for surface_velocity in self:
            periods.append(surface_velocity.period)

        return periods

    def transform_to(self, values):
        return self.transform_to_grid(values)

    def transform_to_grid(self, values):
        return self[0].transform_to_grid(values)

    def transform_from(self, values):
        return self[0].transform_from(values)

    def transform_from_grid(self, values):
        return self.transform_from(values)

    def plot_dispersion_curve(self, x: Union[float, int], y: Union[float, int],
                              grid_space: bool = False):
        """
        plot the dispersion curve at a point x, y of the grid. If grid_space is True,
        x, and y represent grid coordinates. If grid_space is False (default), x and
        y represent the coordinates in spatial units (meters, km etc.).

        :param x: x coordinates expressed in grid or model space
        :type x: float or int
        :param y: y coordinates expressed in grid or model space
        :type y: float or int
        :param grid_space: whether the coordinates are expressed in grid or model space
        :type grid_space: bool
        default value (False, model space)
        """
        cmod = []
        if not grid_space:
            tmp = self[0].transform_to((x, y))
            i = int(tmp[0])
            j = int(tmp[1])
        else:
            i, j = int(x), int(y)
        if not self[0].in_grid([i, j], grid_space=True):
            raise IndexError(f'The point {i, j} is not inside the grid.')
        for k in self:
            cmod.append(k.data[i, j])
        periods = self.periods
        plt.semilogx(periods, cmod, "-ob")
        plt.xlabel("Period (s)")
        if self[0].grid_types == GridTypes.VELOCITY_METERS:
            plt.ylabel(f"{self.velocity_type} velocity (m/s)")
        if self[0].grid_types == GridTypes.VELOCITY_KILOMETERS:
            plt.ylabel(f"{self.velocity_type} velocity (km/s)")
        plt.grid(which='major', linewidth=0.8)
        plt.grid(which='minor', linestyle=':', linewidth=0.5)
        plt.show()


class PhaseVelocityEnsemble(SurfaceVelocityEnsemble):
    """Represents an ensemble of PhaseVelocity instances.

     This class extends the built-in list class to represent an ensemble \
      of PhaseVelocity instances.

     """

    def __init__(self, *args):
        super().__init__(velocity_type=VelocityType.PHASE, *args)

    @classmethod
    def from_seismic_property_grid_ensemble(
            cls, seismic_properties: SeismicPropertyGridEnsemble,
            periods: list, phase: Phases = Phases.RAYLEIGH, z_axis_log:bool = False,
            npts_log_scale: int = 30, disba_param: DisbaParam = DisbaParam(),
    ):
        return super()._from_seismic_property_grid_ensemble(
            seismic_properties,
            periods, phase, z_axis_log,
            npts_log_scale, disba_param,
            VelocityType.PHASE)


class GroupVelocityEnsemble(SurfaceVelocityEnsemble):
    """Represents an ensemble of PhaseVelocity instances.

     This class extends the built-in list class to represent an ensemble \
      of GroupVelocity instances.

     """

    def __init__(self, *args):
        super().__init__(velocity_type=VelocityType.GROUP, *args)

    @classmethod
    def from_seismic_property_grid_ensemble(
            cls, seismic_properties: SeismicPropertyGridEnsemble,
            periods: list, phase: Phases = Phases.RAYLEIGH, z_axis_log: bool = False,
            npts_log_scale: int = 30, disba_param: DisbaParam = DisbaParam(),
    ):
        return super()._from_seismic_property_grid_ensemble(
            seismic_properties,
            periods, phase, z_axis_log,
            npts_log_scale, disba_param,
            VelocityType.GROUP)


def get_coordinates_inventory(
    inventory: Inventory,
    strict: bool = False
) -> Tuple[List[float], List[float], Set[CoordinateSystem]]:
    """
    Extract coordinates from an Inventory object

    :param inventory: The Inventory containing instruments with coordinates.
    :param strict: If True, raise error on unknown/missing coordinate systems.
    :return: (locations_x, locations_y, unique_coordinate_systems)
    """

    locations_x = []
    locations_y = []
    coord_systems = set()

    for inst in inventory.instruments:
        coords = getattr(inst, "coordinates", None)
        if coords is None:
            if strict:
                raise ValueError(f"Instrument {inst} has no `.coordinates` attribute.")
            else:
                print(f"Warning: Instrument {inst} has no `.coordinates` — skipping.")
                continue

        coordinate_system = getattr(coords, "coordinate_system", None)
        if coordinate_system is None:
            if strict:
                raise ValueError(f"Instrument {inst} has no `coordinate_system`.")
            else:
                print(f"Warning: Instrument {inst} has no `coordinate_system` — skipping.")
                continue

        coord_systems.add(coordinate_system)

        locations_x.append(coords.x)
        locations_y.append(coords.y)

    if not locations_x or not locations_y:
        raise ValueError("No valid coordinates found in inventory.")

    return locations_x, locations_y, coord_systems


def _phase_velocity_pnt_worker(
    i: int,
    j: int,
    layers_s: "np.ndarray",
    layers_p: "np.ndarray",
    layers_density: "np.ndarray",
    periods: "np.ndarray",
    thickness: "np.ndarray",
    algorithm: str,
    dc: float,
    phase: str,
) -> Tuple[int, int, "np.ndarray"]:
    """Compute phase velocities for one grid cell (i, j) over all periods."""
    velocity_sij = layers_s[i, j]
    velocity_pij = layers_p[i, j]
    density_ij = layers_density[i, j]
    pd = PhaseDispersion(
        thickness=thickness,
        velocity_p=velocity_pij,
        velocity_s=velocity_sij,
        density=density_ij,
        algorithm=algorithm,
        dc=dc,
    )
    cmod = pd(periods, mode=0, wave=phase).velocity
    return i, j, cmod


def _phase_velocity_interp_worker(
    xi: float,
    yj: float,
    interp_xyz: "np.ndarray",
    s_vel: "np.ndarray",
    p_vel: "np.ndarray",
    density: "np.ndarray",
    periods: "np.ndarray",
    layer_thick: "np.ndarray",
    algorithm: str,
    dc: float,
    phase: str,
) -> Tuple[float, float, "np.ndarray"]:
    """Compute phase velocities for one (x, y) profile over all periods."""
    idx = np.where(
        np.logical_and(interp_xyz[:, 0] == xi, interp_xyz[:, 1] == yj)
    )[0]
    velocity_sij = s_vel[idx]
    velocity_pij = p_vel[idx]
    density_ij = density[idx]
    pd = PhaseDispersion(
        thickness=layer_thick,
        velocity_p=velocity_pij,
        velocity_s=velocity_sij,
        density=density_ij,
        algorithm=algorithm,
        dc=dc,
    )
    cmod = pd(periods, mode=0, wave=phase).velocity
    return xi, yj, cmod

