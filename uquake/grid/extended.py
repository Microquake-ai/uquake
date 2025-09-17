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
import numpy as np
from .base import Grid
from pathlib import Path
import matplotlib.pyplot as plt
from loguru import logger
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Optional
from .base import ray_tracer
import shutil
from uquake.grid import read_grid
from scipy.interpolate import interp1d
from enum import Enum
from typing import List
from uquake.core.event import WaveformStreamID
from uquake.core.coordinates import Coordinates, CoordinateSystem
from uquake.core.inventory import Inventory
from uquake.synthetic.inventory import generate_unique_instrument_code
from uquake.core.event import ResourceIdentifier
from .base import __default_grid_label__
from typing import Set, Tuple, Union
from ttcrpy import rgrid
from scipy.signal import fftconvolve
from disba import PhaseDispersion, PhaseSensitivity
from evtk import hl



__cpu_count__ = cpu_count()

valid_phases = ('P', 'S')

# In many cases, where Z is ignored, North-Up-Down and North-East-Up can be treated as the same
NORTH_EAST_SYSTEMS = {CoordinateSystem.NED, CoordinateSystem.NEU}


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
            srce = Seed(instrument.station_code, instrument.location_code, instrument.coordinates)
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

    def fill_checkerboard(self, anomaly_size, base_velocity, velocity_perturbation, n_sigma):
        data = np.zeros(shape=self.data.shape)

        # Convert anomaly size to grid index units and calculate the starting
        # and ending indices.
        step_anomaly = (np.array(anomaly_size) / np.array(self.spacing)).astype(int)

        if step_anomaly[0] >= self.shape[0]:
            raise Exception("Dimension mismatch: anomaly exceeds the first axis "
                            "of the grid!\n")
        if step_anomaly[1] >= self.shape[1]:
            raise Exception("Dimension mismatch: anomaly exceeds the second axis "
                            "of the grid!\n")
        if step_anomaly[2] >= self.shape[2]:
            raise Exception("Dimension mismatch: anomaly exceeds the third axis "
                            "of the grid!\n")

        # Starting indices
        start_x = (self.data.shape[0] % step_anomaly[0] + step_anomaly[0]) // 2 - 1
        start_y = (self.data.shape[1] % step_anomaly[1] + step_anomaly[1]) // 2 - 1
        start_z = (self.data.shape[2] % step_anomaly[2] + step_anomaly[2]) // 2 - 1

        # Generate the range of indices for each dimension.
        x = np.arange(start_x, self.data.shape[0], step_anomaly[0])
        y = np.arange(start_y, self.data.shape[1], step_anomaly[1])
        z = np.arange(start_z, self.data.shape[2], step_anomaly[2])

        # Create a meshgrid to represent all points in the 3D grid space.
        x_grid, y_grid, z_grid = np.meshgrid(
            np.arange(len(x)),
            np.arange(len(y)),
            np.arange(len(z)),
            indexing='ij'
        )
        indices = np.column_stack(
            (x_grid.reshape((-1,)), y_grid.reshape((-1,)), z_grid.reshape((-1,))))

        # Set the perturbation values based on the checkerboard pattern rule.
        # Calculate the sum of the indices
        summ = indices[:, 0] + indices[:, 1] + indices[:, 2]
        ind = np.where(summ % 2 == 0)[0]  # indices with even sum
        data[x[indices[ind, 0]], y[indices[ind, 1]], z[indices[ind, 2]]] = 1
        ind = np.where(summ % 2 == 1)[0]  # indices with odd sum
        data[x[indices[ind, 0]], y[indices[ind, 1]], z[indices[ind, 2]]] = -1

        # Create 3D Gaussian kernel
        kernel_x, kernel_y, kernel_z = np.meshgrid(
            np.arange(step_anomaly[0]) * self.spacing[0],
            np.arange(step_anomaly[1]) * self.spacing[1],
            np.arange(step_anomaly[2]) * self.spacing[2]
        )
        mu = np.array([
            step_anomaly[0] * self.spacing[0] / 2.,
            step_anomaly[1] * self.spacing[1] / 2.,
            step_anomaly[2] * self.spacing[2] / 2.
        ])
        sigma = np.array([
            step_anomaly[0] * self.spacing[0] / n_sigma,
            step_anomaly[1] * self.spacing[1] / n_sigma,
            step_anomaly[2] * self.spacing[2] / n_sigma
        ])
        kernel = np.exp(-(
                (kernel_x - mu[0]) ** 2 / sigma[0] ** 2 + (kernel_y - mu[1]) ** 2 /
                sigma[1] ** 2 + (kernel_z - mu[2]) ** 2 / sigma[2] ** 2))
        kernel = kernel / np.sum(np.abs(kernel))  # normalization

        # Convolution in the frequency domain
        data = fftconvolve(data, kernel, mode="same")
        data /= np.max(np.abs(data))

        # Rescale the data
        smoothed_data = data * velocity_perturbation * base_velocity + base_velocity
        self.data = smoothed_data

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

        # Get the instrument locations - use normalize_inventory_coordinates to ensure x -> Easting, y -> Northing
        locations_easting, locations_northing, locations_elevation, _ = extract_inventory_easting_northing(inventory, strict=True)

        # Determine the span of the inventory
        min_coords = np.array([np.min(locations_easting), np.min(locations_northing), np.min(locations_elevation)])
        max_coords = np.array([np.max(locations_easting), np.max(locations_northing), np.max(locations_elevation)])
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
        if item.upper() == 'P':
            return self.p_velocity_grid

        elif item.upper() == 'S':
            return self.s_velocity_grid

        elif item.upper() == 'DENSITY':
            return self.density_grid

        else:
            raise ValueError(f'{item} is not a valid key. '
                             f'The key value must either be "P" or "S"')

    def __repr__(self):
        return (f'P Velocity: {self.p_velocity_grid}\n'
                f'S Velocity: {self.s_velocity_grid}\n'
                f'Density   : {self.density_grid}')

    @property
    def density(self):
        return self['density']

    def to_phase_velocities(self, period_min: float = 0.1, period_max: float = 10.,
                            n_periods: int = 10, logspace: bool = True,
                            phase: Union[Phases, str] = Phases.RAYLEIGH,
                            multithreading: bool = True,
                            z: Union[List, np.ndarray] = None,
                            disba_param: Union[DisbaParam] = DisbaParam()):

        if not isinstance(disba_param, DisbaParam):
            raise TypeError(f'disba_param must be type DisbaParam')

        if isinstance(phase, str):
            Phases(phase.upper())

        if isinstance(phase, Phases):
            phase = phase.value.lower()

        if logspace:
            periods = np.logspace(np.log10(period_min), np.log10(period_max), n_periods)
        else:
            periods = np.linspace(period_min, period_max, n_periods)

        @dask.delayed
        def phase_velocity_pnt(index_i, index_j, cells_s, cells_p, cells_density,
                               list_periods, layers_thickness_param, algorithm_param,
                               dc_param):
            velocity_sij = cells_s[index_i, index_j]
            velocity_pij = cells_p[index_i, index_j]
            density_profile_ij = cells_density[index_i, index_j]
            phase_disp = PhaseDispersion(thickness=layers_thickness_param,
                                         velocity_p=velocity_pij,
                                         velocity_s=velocity_sij,
                                         density=density_profile_ij,
                                         algorithm=algorithm_param,
                                         dc=dc_param)
            return phase_disp(list_periods, mode=0, wave=phase).velocity

        @dask.delayed
        def phase_velocity_interp(xi, yj, interpolated_points, s_wave_vel,
                                  p_wave_vel, density_model, list_periods,
                                  layers_thickness_param, algorithm_param,
                                  dc_param):
            indices = np.where(np.logical_and(interpolated_points[:, 0] == xi,
                                              interpolated_points[:, 1] == yj
                                              ))[0]
            velocity_sij = s_wave_vel[indices]
            velocity_pij = p_wave_vel[indices]
            density_profile_ij = density_model[indices]
            phase_disp = PhaseDispersion(thickness=layers_thickness_param,
                                         velocity_p=velocity_pij,
                                         velocity_s=velocity_sij,
                                         density=density_profile_ij,
                                         algorithm=algorithm_param,
                                         dc=dc_param)
            return phase_disp(list_periods, mode=0, wave=phase).velocity
        algorithm = disba_param.algorithm
        dc = disba_param.dc

        if z is None:
            thickness = self.spacing[2] * np.ones(shape=(self.shape[2] - 1))
            if self.grid_units == GridUnits.METER:
                thickness *= 1.e-3
                velocity_s = self.s.data * 1.e-3
                velocity_p = self.p.data * 1.e-3
            else:
                velocity_s = self.s.data
                velocity_p = self.p.data
            layers_s = 0.5 * (velocity_s[:, :, 1:] + velocity_s[:, :, :-1])
            layers_p = 0.5 * (velocity_p[:, :, 1:] + velocity_p[:, :, :-1])
            layers_density = 0.5 * (self.density.data[:, :, 1:] +
                                    self.density.data[:, :, :-1])

            if multithreading:
                results = []
                for i in range(self.shape[0]):
                    cmod_ij = []
                    results.append(cmod_ij)
                    for j in range(self.shape[1]):
                        cmod_ij.append(phase_velocity_pnt(i, j, layers_s, layers_p,
                                                          layers_density, periods,
                                                          thickness, algorithm, dc))

                phase_velocity = dask.compute(*results)  # todo see other arguments
                phase_velocity = np.array(phase_velocity)
                phase_velocity = [phase_velocity[:, :, k] for k in range(n_periods)]
            else:
                phase_velocity = [np.zeros(shape=(self.shape[0], self.shape[1]))
                                  for _ in range(n_periods)]
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        velocity_s_ij = layers_s[i, j]
                        velocity_p_ij = layers_p[i, j]
                        density_ij = layers_density[i, j]
                        pd = PhaseDispersion(thickness=thickness,
                                             velocity_p=velocity_p_ij,
                                             velocity_s=velocity_s_ij,
                                             density=density_ij,
                                             algorithm=algorithm,
                                             dc=dc)
                        cmod = pd(periods, mode=0, wave=phase).velocity
                        for k in range(n_periods):
                            phase_velocity[k][i, j] = cmod[k]
        else:
            # define layers thickness and centers
            layers_centers = 0.5 * (z[1:] + z[:-1])
            layers_thickness = z[1:] - z[:-1]
            # interpolation
            x = np.arange(self.origin[0], self.corner[0], self.spacing[0])
            y = np.arange(self.origin[1], self.corner[1], self.spacing[1])
            x_grd, y_grd, z_grd = np.meshgrid(x, y, layers_centers, indexing='ij')
            coord_interpolation = np.column_stack((x_grd.reshape(-1), y_grd.reshape(-1),
                                                   z_grd.reshape(-1)))
            velocity_s = self.s.interpolate(coord_interpolation, grid_space=False)
            velocity_p = self.p.interpolate(coord_interpolation, grid_space=False)
            density = self.density.interpolate(coord_interpolation, grid_space=False)
            if self.grid_units == GridUnits.METER:
                velocity_s *= 1.e-3
                velocity_p *= 1.e-3
                z *= 1.e-3
            if multithreading:
                results = []
                for x_i in x:
                    cmod_ij = []
                    results.append(cmod_ij)
                    for y_j in y:
                        cmod_ij.append(phase_velocity_interp(x_i, y_j,
                                                             coord_interpolation,
                                                             velocity_s, velocity_p,
                                                             density, periods,
                                                             layers_thickness, algorithm,
                                                             dc))

                phase_velocity = dask.compute(*results)  # todo see other arguments
                phase_velocity = np.array(phase_velocity)
                phase_velocity = [phase_velocity[:, :, k] for k in range(n_periods)]

            else:
                # run code in one thread
                phase_velocity = [np.zeros(shape=(self.shape[0], self.shape[1]))
                                  for _ in range(n_periods)]
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        ind = np.where(np.logical_and(coord_interpolation[:, 0] == x[i],
                                                      coord_interpolation[:, 1] == y[j]
                                                      ))[0]
                        velocity_s_ij = velocity_s[ind]
                        velocity_p_ij = velocity_p[ind]
                        density_ij = density[ind]
                        pd = PhaseDispersion(thickness=layers_thickness,
                                             velocity_p=velocity_p_ij,
                                             velocity_s=velocity_s_ij,
                                             density=density_ij,
                                             algorithm=algorithm,
                                             dc=dc)
                        cmod = pd(periods, mode=0, wave=phase).velocity
                        for k in range(n_periods):
                            phase_velocity[k][i, j] = cmod[k]
        return periods, phase_velocity

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


class PhaseVelocity(Grid):
    def __init__(self, network_code: str, data_or_dims: Union[np.ndarray, List, Tuple],
                 period: float, phase: Phases = Phases.RAYLEIGH,
                 grid_type=GridTypes.VELOCITY_METERS, grid_units=GridUnits.METER,
                 spacing: Union[np.ndarray, List, Tuple] = None,
                 origin: Union[np.ndarray, List, Tuple] = None,
                 resource_id: ResourceIdentifier = ResourceIdentifier(),
                 value: float = 0, coordinate_system: CoordinateSystem = CoordinateSystem.NED,
                 label: str = __default_grid_label__,
                 float_type: FloatTypes = FloatTypes.FLOAT):
        """Initialize a PhaseVelocity instance.

            :param network_code: The network code associated with the data.
            :type network_code: str
            :param data_or_dims: The data or dimensions of the grid.
            :type data_or_dims: Union[np.ndarray, List, Tuple]
            :param period: the wave period used to calculate the phase velocity.
            :type period: float
            :param phase: The seismic phase type (default: Phases.RAYLEIGH).
            :type phase: Phases
            :param grid_type: The type of grid (default: GridTypes.VELOCITY_METERS).
            :type grid_type: GridTypes
            :param grid_units: The units of the grid (default: GridUnits.METER).
            :type grid_units: GridUnits
            :param spacing: The spacing of the grid (default: None).
            :type spacing: Union[np.ndarray, List, Tuple], optional
            :param origin: The origin of the grid (default: None).
            :type origin: Union[np.ndarray, List, Tuple], optional
            :param resource_id: The resource identifier for the data (default: ResourceIdentifier()).
            :type resource_id: ResourceIdentifier
            :param value: The value associated with the data (default: 0).
            :type value: float
            :param coordinate_system: The coordinate system of the data (default: CoordinateSystem.NED).
            :type coordinate_system: CoordinateSystem
            :param label: The label of the grid (default: __default_grid_label__).
            :type label: str
            :param float_type: The float precision (default: FloatTypes.FLOAT).
            :type float_type: FloatTypes
        """

        self.network_code = network_code
        self.period = period
        self.grid_type = grid_type
        self.grid_units = grid_units
        self.phase = phase
        self.float_type = float_type

        super().__init__(data_or_dims, origin=origin, spacing=spacing,
                         resource_id=resource_id, value=value,
                         coordinate_system=coordinate_system,
                         label=label)

    @classmethod
    def from_seismic_property_grid_ensemble(cls,
                                            seismic_param: SeismicPropertyGridEnsemble,
                                            period: float, phase: Phases,
                                            z_axis_log:bool = False,
                                            npts_log_scale: int = 30,
                                            disba_param: DisbaParam = DisbaParam()):
        """Create a 2D Phase Velocity model.

        This method constructs a PhaseVelocity instance from a SeismicPropertyGridEnsemble,
        associating it with the specified period, phase, and DisbaParam.

        :param seismic_param: The SeismicPropertyGridEnsemble used to create the \
         PhaseVelocity instance.
        :type seismic_param: SeismicPropertyGridEnsemble
        :param period: the wave period used to calculate the phase velocity.
        :type period: float
        :param phase: The seismic phase.
        :type phase: Phases
        :param disba_param: Parameters of Disba to be used(default: DisbaParam()).
        :type disba_param: DisbaParam, optional
        :return: A PhaseVelocity instance created from the SeismicPropertyGridEnsemble.
        :rtype: PhaseVelocity
        """
        if z_axis_log:
            z_max = (seismic_param.spacing[2] * seismic_param.shape[2] +
                     seismic_param.origin[2])
            z = (np.logspace(0, np.log10(10 + 1), npts_log_scale) - 10 ** 0 +
                 seismic_param.origin[2]) * z_max / 10
        else:
            z = None
        _, phase_velocity = seismic_param.to_phase_velocities(period_min=period,
                                                              period_max=period,
                                                              n_periods=1,
                                                              logspace=False,
                                                              z=z,
                                                              multithreading=True,
                                                              disba_param=disba_param)
        phase_velocity = phase_velocity[0]
        if seismic_param.grid_type == GridTypes.VELOCITY_METERS:
            phase_velocity *= 1.e3

        return cls(
            network_code=seismic_param.network_code,
            data_or_dims=phase_velocity,
            period=period,
            phase=phase,
            grid_type=seismic_param.grid_type,
            grid_units=seismic_param.grid_units,
            spacing=(seismic_param.spacing[0], seismic_param.spacing[1]),
            origin=(seismic_param.origin[0], seismic_param.origin[1]),
            resource_id=seismic_param.resource_id,
            coordinate_system=seismic_param.coordinate_system,
            label=seismic_param.label,
            float_type=seismic_param.float_type
        )

    @classmethod
    def from_inventory(cls, network_code: str, inventory: Inventory,
                       spacing: Union[float, Tuple[float, float]], 
                       period: float,
                       padding: Union[float, Tuple[float, float]] = 0.2,
                       phase: Phases = Phases.RAYLEIGH,
                       **kwargs
                       ):
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
        :param kwargs: Additional keyword arguments.
        :type kwargs: dict
        :return: An instance of the Grid class created from the inventory.
        :rtype: Grid

        :raises ValueError: If the padding or spacing values are invalid.

        This method calculates the grid dimensions and origin by considering the
        span of the inventory and the specified padding. It then creates and
        returns a grid object with the calculated parameters.
        """
       
        locations_easting, locations_northing, _, _ = extract_inventory_easting_northing(inventory, strict=True)

        # Determine the span of the inventory
        min_coords = np.array([np.min(locations_easting), np.min(locations_northing)])
        max_coords = np.array([np.max(locations_easting), np.max(locations_northing)])
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
        grid_dims = np.ceil((padded_corner - padded_origin) / np.array(spacing)).astype(int)

        # Create and return the grid object
        return cls(network_code, grid_dims, spacing=spacing, origin=padded_origin, period=period, phase=phase, **kwargs)

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


    @property
    def grid_id(self):
        return self.resource_id

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
            field_name = f'velocity_{self.phase.value}'

        super().write(filename, format=format, field_name=field_name, **kwargs)

    def plot(
            self,
            receivers: Optional[Union[np.ndarray, SeedEnsemble]] = None,
            fig_size: Tuple[float, float] = (10, 8),
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            mask: Optional[dict] = None,
            **imshow_kwargs,
    ):
        """
        Plot the Phase velocity with optional overlay of receiver positions.

        Parameters
        ----------
        receivers : np.ndarray or SeedEnsemble, optional
            Receiver positions to overlay on the plot. Can be a 2D NumPy array of shape
            (N, 2) containing (x, y) coordinates or a SeedEnsemble object.

        fig_size : tuple of float, default=(10, 8)
            Size of the matplotlib figure in inches (width, height).

        vmin : float, optional
            Minimum value for the colormap. If None, the 1st percentile of the data is used.

        vmax : float, optional
            Maximum value for the colormap. If None, the 99th percentile of the data is used.

        mask : dict, optional
            Dictionary specifying regions to mask out from the plot. Keys and structure
            depend on implementation, e.g., {'polygon': [(x1, y1), ..., (xn, yn)]}.

        **imshow_kwargs
            Additional keyword arguments passed directly to `matplotlib.axes.Axes.imshow.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the plot.

        ax : matplotlib.axes.Axes
            The matplotlib axes object where the grid and overlays are plotted.
        """
        fig, ax = plt.subplots(figsize=fig_size)
        if 'cmap' not in imshow_kwargs:
            imshow_kwargs.setdefault('cmap', 'seismic')

        cax = ax.imshow(
            self.data.T,
            origin="lower",
            extent=(self.origin[0], self.corner[0], self.origin[1], self.corner[1]),
            **imshow_kwargs,
        )

        if mask is not None:
            positive_mask = super().masked_region_xy(**mask,
                                                     ax=ax)
            grid_data = np.where(positive_mask, self.data, np.nan).T
        else:
            grid_data = self.data.T

        if vmin is None:
            vmin = np.nanpercentile(grid_data, 1)
        if vmax is None:
            vmax = np.nanpercentile(grid_data, 99)
        cax.set_clim(vmin, vmax)
        cb = fig.colorbar(cax)
        cb.update_normal(cax)

        if self.grid_units == GridUnits.METER:
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            cb.set_label("Vel " + self.phase.value + " (m/s)", rotation=270, labelpad=10)

        if self.grid_units == GridUnits.KILOMETER:
            ax.set_xlabel("X (km)")
            ax.set_ylabel("Y (km)")
            cb.set_label("Velocity (km/s)", rotation=270, labelpad=10)

        ax.set_title("Period = {0:1.2f} s".format(self.period))

        if isinstance(receivers, np.ndarray):
            ax.plot(receivers[:, 0], receivers[:, 1], "s", color="yellow")

        if isinstance(receivers, SeedEnsemble):
            coordinates = receivers.locs
            ax.plot(coordinates[:, 0], coordinates[:, 1], "s", color="yellow")

        return fig, ax

    def __repr__(self):
        repr_str = """
                period :  %0.2f
                spacing: %s
                origin : %s
                shape  : %s
                """ % (self.period, self.spacing, self.origin, self.shape)
        return repr_str

    def __str__(self):
        return self.__repr__()

    def compute_frechet(self, sources: Union[SeedEnsemble, np.ndarray],
                        receivers: Union[SeedEnsemble, np.ndarray],
                        ns: Union[int, Tuple[int, int, int]] = 5,
                        tt_cal: bool = True, cell_slowness: bool = True,
                        threads: int = 1):


        """
        Calculate the Frechet derivative and travel times.

        :param sources: The source ensemble containing locations.
        :type sources: Union[SeedEnsemble, np.ndarray]
        :param receivers: The receiver ensemble containing locations.
        :type receivers: Union[SeedEnsemble, np.ndarray]
        :param ns: Number of secondary nodes for grid refinement.
                   If an integer is provided, it is used for all dimensions.
                   If a tuple is provided, it specifies the number of nodes in the x, y,
                   and z dimensions respectively.
        :type ns: Union[int, Tuple[int, int, int]], optional
        :param tt_cal: If True, the travel times will also be returned. Default is True.
        :type tt_cal: bool, optional
        :param cell_slowness: If True, the grid will be created with cell slowness.
                              If False, the grid will be created without cell slowness.
                              Default is True.
        :type cell_slowness: bool, optional
        :param threads: The number of threads to use for computation. Default is 1.
        :type threads: int, optional
        :return: Frechet derivative and optionally travel times.
        :rtype: Tuple[np.ndarray, np.ndarray] if tt_cal is True, otherwise np.ndarray
        """

        # Create the grid with specified parameters
        grid = self.to_rgrid(n_secondary=ns, cell_slowness=cell_slowness,
                             threads=threads)

        # Extract source locations

        if isinstance(sources, SeedEnsemble):
            srcs = sources.locs[:, :2]
        else:
            srcs = sources

        # Extract receiver locations
        if isinstance(receivers, SeedEnsemble):
            rxs = receivers.locs[:, :2]
        else:
            rxs = receivers

        # Perform ray tracing
        tt, _, frechet = grid.raytrace(source=srcs, rcv=rxs, compute_L=True,
                                       return_rays=True)
        if tt_cal:
            return frechet, tt
        else:
            return frechet

    def to_time(self, seed: Seed, ns: Union[int, Tuple[int, int, int]] = 5):
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


class PhaseVelocityEnsemble(list):
    """Represents an ensemble of PhaseVelocity instances.

     This class extends the built-in list class to represent an ensemble \
      of PhaseVelocity instances.

     """

    def __init__(self, *args):
        super().__init__(*args)

    def append(self, phase_velocity):
        if isinstance(phase_velocity, PhaseVelocity):
            super().append(phase_velocity)
        else:
            print("Only instances of the PhaseVelocity class can be added to the list.")

    def add_phase_velocity(self, phase_velocity):
        self.append(phase_velocity)

    @classmethod
    def from_seismic_property_grid_ensemble(
            cls, seismic_properties: SeismicPropertyGridEnsemble,
            periods: list, phase: Phases = Phases.RAYLEIGH, z_axis_log:bool = False,
            npts_log_scale: int = 30, disba_param: DisbaParam = DisbaParam()
    ):
        """
        Create a PhaseVelocityEnsemble from a SeismicPropertyGridEnsemble.

        This method constructs a PhaseVelocityEnsemble instance from a \
        SeismicPropertyGridEnsemble, associating it with the specified periods and phase.

        :param seismic_properties: The SeismicPropertyGridEnsemble considered to create \
        the PhaseVelocityEnsemble instance.
        :type seismic_properties: SeismicPropertyGridEnsemble
        :param periods: The list of periods.
        :type periods: list
        :param phase: The seismic phase (default: Phases.RAYLEIGH).
        :type phase: Phases
        :param disba_param: Disba parameters (default: DisbaParam()).
        :type disba_param: DisbaParam, optional
        :return: A PhaseVelocityEnsemble instance created from the SeismicPropertyGridEnsemble.
        :rtype: PhaseVelocityEnsemble
        """
        cls_obj = cls()
        for p in periods:
            cls_obj.append(PhaseVelocity.from_seismic_property_grid_ensemble(
                seismic_properties, p, phase, z_axis_log, npts_log_scale, disba_param))
        return cls_obj

    @property
    def periods(self):
        periods = []
        for phase_velocity in self:
            periods.append(phase_velocity.period)

        return periods

    def transform_to(self, values):
        return self.transform_to_grid(values)

    def transform_to_grid(self, values):
        return self[0].transform_to_grid(values)

    def transform_from(self, values):
        return self[0].transform_from

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
        if self[0].grid_units == GridUnits.METER:
            plt.ylabel("Phase velocity (m/s)")
        if self[0].grid_units == GridUnits.KILOMETER:
            plt.ylabel("Phase velocity (km/s)")
        plt.grid(which='major', linewidth=0.8)
        plt.grid(which='minor', linestyle=':', linewidth=0.5)
        plt.show()


def extract_inventory_easting_northing(
    inventory: Inventory,
    strict: bool = False
) -> Tuple[List[float], List[float], List[float], Set[CoordinateSystem]]:
    """
    Extract instrument coordinates from an inventory in a consistent fashion.
        - Easting becomes X
        - Northing becomes Y

    :param inventory: The Inventory containing instruments with coordinates.
    :param strict: If True, raise error on unknown/missing coordinate systems.
    :return: (locations_easting, locations_northing, locations_elevation, unique_coordinate_systems)
    """
    locations_easting = []
    locations_northing = []
    locations_elevation = []
    coord_systems = set()

    for inst in inventory.instruments:
        coords = getattr(inst, "coordinates", None)
        if coords is None:
            if strict:
                raise ValueError(f"Instrument {inst} has no `.coordinates` attribute.")
            else:
                logger.warning("Instrument {} has no `.coordinates` — skipping.", inst)
                continue

        coordinate_system = getattr(coords, "coordinate_system", None)
        if coordinate_system is None:
            if strict:
                raise ValueError(f"Instrument {inst} has no `coordinate_system`.")
            else:
                logger.warning("Instrument {} has no `coordinate_system` — skipping.", inst)
                continue

        locations_easting.append(coords.easting)
        locations_northing.append(coords.northing)
        locations_elevation.append(coords.elevation)
        coord_systems.add(coordinate_system)

    if not locations_easting or not locations_northing or not locations_elevation:
        raise ValueError("No valid coordinates found in inventory.")

    return locations_easting, locations_northing, locations_elevation, coord_systems
