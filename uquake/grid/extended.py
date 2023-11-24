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
import skfmm
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
from typing import Union, Tuple
import h5py

__cpu_count__ = cpu_count()

valid_phases = ('P', 'S')


class Phases(Enum):
    P = 'P'
    S = 'S'


class GridTypes(Enum):
    VELOCITY = 'VELOCITY'
    VELOCITY_METERS = 'VELOCITY_METERS'
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

    def __str__(self):
        return self.value


valid_grid_types = (
    'VELOCITY',
    'VELOCITY_METERS',
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
    'ANGLE2D'
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

    def __init__(self, seeds: List[Seed] = [], units: GridUnits = GridUnits.METER):
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
        self.seeds = seeds

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

    def __init__(self, data_or_dims, origin, spacing, phase: Phases,
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

        super().mv(self, self.base_name,
                   origin, destination)

    @property
    def base_name(self):
        return self.get_base_name(self.network_code, self.phase)


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

        if p_velocity_grid.dims != s_velocity_grid.dims:
            raise ValueError('p and s velocity grids must have the same '
                             'dimensions')
        if p_velocity_grid.label != s_velocity_grid.label:
            raise ValueError('p and s velocity grids must have the same '
                             'label')

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
            self[key].write(path=path)

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


class SeededGridType(Enum):
    TIME = 'TIME'
    ANGLE = 'ANGLE'

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
                 label: str = __default_grid_label__,
                 coordinate_system: CoordinateSystem = CoordinateSystem.NED):
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
                         coordinate_system=coordinate_system)

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
        return self.seed.station_code

    @property
    def location_code(self):
        return self.seed.location_code

    @property
    def instrument_code(self):
        return self.seed.label

    @property
    def seed_label(self):
        return self.seed.short_label

    @property
    def seed_units(self):
        return self.seed.units.value

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
                out_file.write(self.data.astype(np.float32).tobytes())

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


class TTGrid(SeededGrid):
    def __init__(self, network_code, data_or_dims, origin, spacing, seed: Seed,
                 velocity_model_id: ResourceIdentifier,
                 phase: Phases = Phases.P, value: float = 0,
                 grid_units: GridUnits = __default_grid_units__,
                 float_type: FloatTypes = __default_float_type__,
                 grid_id: ResourceIdentifier = ResourceIdentifier(),
                 label=__default_grid_label__,
                 coordinate_system: CoordinateSystem = CoordinateSystem.NED):

        super().__init__(network_code, data_or_dims, origin, spacing, seed,
                         velocity_model_id=velocity_model_id, phase=phase,
                         value=value, grid_type=GridTypes.TIME,
                         grid_units=grid_units,
                         float_type=float_type,
                         grid_id=grid_id,
                         coordinate_system=coordinate_system,
                         label=label)

    def to_azimuth(self):
        """
        This function calculate the takeoff angle and azimuth for every
        grid point given a travel time grid calculated using an Eikonal solver
        :return: azimuth and takeoff angles grids
        .. Note: The convention for the takeoff angle is that 0 degree is down.
        """

        gds_tmp = np.gradient(self.data)
        gds = [-gd for gd in gds_tmp]

        azimuth = np.arctan2(gds[0], gds[1]) * 180 / np.pi
        # azimuth is zero northwards

        return AngleGrid(self.network_code, azimuth, self.origin, self.spacing,
                         self.seed, phase=self.phase, float_type=self.float_type,
                         grid_id=ResourceIdentifier(), grid_type=GridTypes.AZIMUTH,
                         velocity_model_id=self.velocity_model_id)

    def to_takeoff(self):
        gds_tmp = np.gradient(self.data)
        gds = [-gd for gd in gds_tmp]

        hor = np.sqrt(gds[0] ** 2 + gds[1] ** 2)
        takeoff = np.arctan2(hor, -gds[2]) * 180 / np.pi
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

        return ray_tracer(self, start, grid_space=grid_space,
                          max_iter=max_iter, arrival_id=arrival_id,
                          earth_model_id=self.model_id,
                          network=self.network_code)

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

    def select(self, instruments_code: Optional[List[str]] = None,
               phases: Optional[List[Phases]] = None):
        """
        return a list of grid corresponding to seed_labels.
        :param instruments_code: seed labels of the travel time grids to return
        :param phases: the phase {'P' or 'S'}, both if None.
        :return: a list of travel time grids
        :rtype: TravelTimeEnsemble
        """

        if (instruments_code is None) and (phases is None):
            return self

        tmp = []
        if instruments_code is None:
            instruments_code = np.unique(self.seeds)

        if phases is None:
            phases = [Phases.P.value, Phases.S.value]
        else:
            phases = [phase.value if isinstance(phase, Phases) else Phases(phase).value
                      for phase in phases]

        returned_grids = []
        for travel_time_grid in self.travel_time_grids:
            if travel_time_grid.seed_label in instruments_code:
                if isinstance(travel_time_grid.phase, Phases):
                    phase = travel_time_grid.phase.value
                else:
                    phase = travel_time_grid.phase
                if phase in phases:
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

        tt_grids = self.select(instruments_code=seed_labels)

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

        tt_grids = self.select(instruments_code=seed_labels)

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

    def ray_tracer(self, starts, seed_label=None, multithreading=False,
                   cpu_utilisation=0.9, grid_space=False, max_iter=1000):
        """

        :param starts: origin of the ray, usually the location of the events
        :param seed_label: a list of seed labels
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

        travel_time_grid = self.select(instruments_code=[seed_label])

        kwargs = {'grid_space': grid_space,
                  'max_iter': max_iter}

        if multithreading:

            ray_tracer_func = partial(ray_tracer, **kwargs)

            num_threads = int(np.ceil(cpu_utilisation * __cpu_count__))
            # ensuring that the number of threads is comprised between 1 and
            # __cpu_count__
            num_threads = np.max([np.min([num_threads, __cpu_count__]), 1])

            data = []
            for start in starts:
                data.append((travel_time_grid, start))

            with Pool(num_threads) as pool:
                results = pool.starmap(ray_tracer_func, data)

            for result in results:
                result.network = self.travel_time_grids[0].network_code

        else:
            results = []
            for start in starts:
                results.append(travel_time_grid.ray_tracer(start, **kwargs))

        return results

    @property
    def seeds(self):
        seeds = []
        for seed_label in self.seed_labels:
            seeds.append(self.select(instruments_code=seed_label)[0].seed)

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
                 grid_id: ResourceIdentifier = ResourceIdentifier(),
                 label=__default_grid_label__,
                 coordinate_system: CoordinateSystem = CoordinateSystem.NED):

        super().__init__(network_code, data_or_dims, origin, spacing, seed,
                         velocity_model_id=velocity_model_id, phase=phase,
                         value=value, grid_type=GridTypes.TIME,
                         grid_units=grid_units,
                         float_type=float_type,
                         grid_id=grid_id,
                         coordinate_system=coordinate_system,
                         label=label)

    def write_nlloc(self, path='.'):
        super().write_nlloc(path=path)

