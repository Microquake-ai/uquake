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

from __future__ import annotations


import numpy as np
from uuid import uuid4
from ..core.logging import logger
from pkg_resources import load_entry_point
from ..core.util import ENTRY_POINTS
from pathlib import Path
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from uquake.core.coordinates import CoordinateSystem, Coordinates
from typing import Union, List, Tuple
from uquake.core.inventory import Inventory, Network, Station, Channel
import random
from uquake.core.event import ResourceIdentifier
from copy import deepcopy
from hashlib import sha256

def read_grid(filename, format='PICKLE', **kwargs):
    format = format.upper()

    if format not in ENTRY_POINTS['grid'].keys():
        raise TypeError(f'format {format} is currently not supported '
                        f'for Grid objects')

    format_ep = ENTRY_POINTS['grid'][format]
    read_format = load_entry_point(format_ep.dist.key,
                                   f'uquake.io.grid.{format_ep.name}',
                                   'readFormat')

    return read_format(filename, **kwargs)


__default_grid_label__ = 'Default'


class Grid(object):
    """
    Object containing a regular grid
    """

    def __init__(self, data_or_dims: Union[np.ndarray, List, Tuple],
                 spacing: Union[np.ndarray, List, Tuple] = None,
                 origin: Union[np.ndarray, List, Tuple] = None,
                 resource_id: ResourceIdentifier = ResourceIdentifier(),
                 value: float = 0,
                 coordinate_system: CoordinateSystem = CoordinateSystem.NED,
                 label: str = __default_grid_label__):

        """
        can hold both 2 and 3 dimensional grid
        :param data_or_dims: either a numpy array or a Tuple/List with the grid
        dimensions. If grid dimensions are specified, the grid is initialized
        with value
        :param spacing: a set of two or three values containing the grid spacing
        :type spacing: Tuple, List or numpy.ndarray
        :param origin: a set of two or three values origin of the grid
        :type origin: tuple, List or numpy.ndarray
        :param resource_id: unique identifier for the grid, if set to None,
        :param value: value to fill the grid should dims be specified
        :type value:
        uuid4 is used to define a unique identifier.
        :param coordinate_system: Coordinate system
        :type coordinate_system: ~uquake.core.coordinates.CoordinateSystem
        :param label: Label providing additional information on the grid usage
        :type label: str
        """

        data_or_dims = np.array(data_or_dims)

        if data_or_dims.ndim == 1:
            self.data = np.ones(data_or_dims) * value
        else:
            self.data = data_or_dims

        if resource_id is None:
            self.resource_id = str(uuid4())
        else:
            self.resource_id = resource_id

        if origin is None:
            self.origin = np.zeros(len(self.data.shape))
        else:
            origin = np.array(origin)
            if origin.shape[0] == len(self.data.shape):
                self.origin = origin
            else:
                logger.error(f'origin shape should be {len(self.data.shape)}')
                raise ValueError

        if spacing is None:
            self.spacing = np.ones(len(self.data.shape))
        else:
            spacing = np.array(spacing)
            if spacing.shape[0] == len(self.data.shape):
                self.spacing = spacing
            else:
                logger.error(f'spacing shape should be {len(self.data.shape)}')
                raise ValueError

            self.coordinate_system = coordinate_system

        self.id = id
        self.label = label

    def __hash__(self):
        return hash((tuple(self.data.ravel()), tuple(self.spacing),
                     tuple(self.shape), tuple(self.origin)))

    def __eq__(self, other):
        self.hash == other.hash

    def __getitem__(self, item):
        return self.data[item]

    @property
    def hash(self):
        return self.__hash__()

    @classmethod
    def from_ods(cls, origin, dimensions, spacing, val=0):
        """
        create a grid from origin, dimensions and spacing
        :param origin: grid origin
        :type origin: tuple
        :param dimensions: grid dimension
        :type dimensions: tuple
        :param spacing: spacing between the grid nodes
        :type spacing: float
        :param val: constant value with which to fill the grid
        """

        data = np.ones(tuple(dimensions)) * val
        cls.grid = cls(data, spacing=spacing, origin=origin)

    @classmethod
    def from_ocs(cls, origin, corner, spacing, val=0):
        """
        create a grid from origin, corner and spacing
        :param origin: grid origin (e.g., lower left corner for 2D grid)
        :type origin: tuple or list or numpy.array
        :param corner: grid upper (e.g., upper right corner for 2D grid)
        :type corner: tuple or list or numpy.array
        :param spacing: spacing between the grid nodes
        :type spacing: float
        :param val: constant value with which to fill the grid
        :param buf: buffer around the grid in fraction of grid size
        """
        origin2 = origin
        corner2 = corner

        gshape = tuple([int(np.ceil((c - o) / s))
                        for o, c, s in zip(origin2, corner2, spacing)])
        data = np.ones(gshape) * val
        out = cls(data, spacing=spacing, origin=origin)
        out.fill_homogeneous(val)
        return out

    @classmethod
    def from_ocd(cls, origin, corner, dimensions, val=0):
        """
        create a grid from origin, corner and dimensions
        :param origin: grid origin (e.g., lower left corner for 2D grid)
        :param corner: grid upper (e.g., upper right corner for 2D grid)
        :param dimensions: grid dimensions
        :param val: constant value with which to fill the grid
        :return:
        """

        data = np.ones(dimensions) * val
        spacing = (corner - origin) / (dimensions - 1)
        cls(data, spacing, spacing=spacing, origin=origin)
        return cls

    def __repr__(self):
        repr_str = """
        spacing: %s
        origin : %s
        shape  : %s
        """ % (self.spacing, self.origin, self.shape)
        return repr_str

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return np.all((self.shape == other.shape) &
                      (self.spacing == other.spacing) &
                      np.all(self.origin == other.origin))

    def __mul__(self, other):
        if isinstance(other, Grid):
            if self.check_compatibility(self, other):
                mul_data = self.data * other.data
                return Grid(mul_data, spacing=self.spacing,
                            origin=self.origin)
            else:
                raise ValueError

        else:
            raise TypeError

    def __abs__(self):
        return np.abs(self.data)

    def transform_to(self, values):
        """
        transform model space coordinates into grid space coordinates
        :param values: tuple of model space coordinates
        :type values: tuple
        :rtype: tuple
        """
        coords = (values - self.origin) / self.spacing

        return coords

    def transform_to_grid(self, values):
        """
        transform model space coordinates into grid space coordinates
        :param values: tuple of model space coordinates
        :type values: tuple
        :rtype: tuple
        """

        return self.transform_to(values)

    def transform_from(self, values):
        """
        transform grid space coordinates into model space coordinates
        :param values: tuple of grid space coordinates
        :type values: tuple
        :rtype: tuple
        """
        return values * self.spacing + self.origin

    def transform_from_grid(self, values):
        """
        transform grid space coordinates into model space coordinates
        :param values: tuple of grid space coordinates
        :type values: tuple
        :rtype: tuple
        """

        return self.transform_from(values)

    def check_compatibility(self, other):
        """
        check if two grids are compatible, i.e., have the same shape, spacing
        and origin
        """
        return (np.all(self.shape == other.shape) and
                np.all(self.spacing == other.spacing) and
                np.all(self.origin == other.origin))

    def __get_shape__(self):
        """

        return the shape of the object
        """
        return self.data.shape

    shape = property(__get_shape__)

    def copy(self):
        """
        copy the object using copy.deepcopy
        """
        import copy
        cp = copy.deepcopy(self)
        return cp

    def in_grid(self, point):
        """
        Check if a point is inside the grid
        :param point: the point to check in absolute coordinate (model)
        :type point: tuple, list or numpy array
        :returns: True if point is inside the grid
        :rtype: bool
        """
        corner1 = self.origin
        corner2 = self.origin + self.spacing * np.array(self.shape)

        return np.all((point >= corner1) & (point <= corner2))

    def fill_homogeneous(self, value):
        """
        fill the data with a constant value
        :param value: the value with which to fill the array
        """
        self.data.fill(value)

    def fill_random(self, mean, std, smooth_sigma, seed=None):
        """
        fill the model with random number with a mean of "mean".
        :param mean: mean of the random number
        :param std: standard deviation of the grid
        :param sigma: gaussian smoothing parameter
        :param seed: random seed
        """

        np.random.seed(seed)

        self.data = np.random.randn(self.dims[0], self.dims[1], self.dims[2])
        self.smooth(smooth_sigma)
        self.data = self.data * std / np.std(self.data) + mean

    def generate_points(self, pt_spacing=None):
        """
        Generate points within the grid
        """
        # if pt_spacing is None:
        ev_spacing = self.spacing

        dimensions = np.array(self.shape) * self.spacing / ev_spacing

        xe = np.arange(0, dimensions[0]) * ev_spacing[0] + self.origin[0]
        ye = np.arange(0, dimensions[1]) * ev_spacing[1] + self.origin[1]
        ze = np.arange(0, dimensions[2]) * ev_spacing[2] + self.origin[2]

        Xe, Ye, Ze = np.meshgrid(xe, ye, ze)

        Xe = Xe.reshape(np.prod(Xe.shape))
        Ye = Ye.reshape(np.prod(Ye.shape))
        Ze = Ze.reshape(np.prod(Ze.shape))
        data_e = self.data.reshape(np.prod(self.data.shape))
        return Xe, Ye, Ze, data_e

    def flattens(self):
        return self.generate_points()

    def rbf_interpolation_sensitivity(self, location, epsilon,
                                      threshold=0.1):
        """
        calculate the sensitivity of each element given a location
        :param location: location in model space at which the interpolation
        occurs
        :param epsilon: the standard deviation of the gaussian kernel
        :param threshold: threshold relative to the maximum value below which
        the weights are considered 0.
        :rparam: the sensitivity matrix
        """
        x, y, z, v = self.flattens()

        # calculating the distance between the location and every grid points

        dist = np.linalg.norm([x - location[0], y - location[1],
                               z - location[2]], axis=0)

        sensitivity = np.exp(-(dist / epsilon) ** 2)
        sensitivity[sensitivity < np.max(sensitivity) * threshold] = 0
        sensitivity = sensitivity / np.sum(sensitivity)

        return sensitivity

    def generate_random_points_in_grid(self, n_points=1,
                                       grid_space=False,
                                       seed=None):
        """
        Generate a random set of points within the grid
        :param n_points: number of points to generate (default=1)
        :type n_points: int
        :param grid_space: whether the output is expressed in
        grid coordinates (True) or model coordinates (False)
        (default: False)
        :type grid_space: bool
        :param seed: random seed for reproducibility
        :return: an array of triplet
        """

        np.random.seed(seed)

        points = np.random.uniform(0, 1, (n_points, len(self.data.shape)))

        for i in range(n_points):
            points[i] = points[i] * self.dimensions

        if not grid_space:
            return self.transform_from_grid(points)

        return points

    def interpolate(self, coord, grid_space=True, mode='nearest',
                    order=1, **kwargs):
        """
        This function interpolate the values at a given point expressed
        either in grid or absolute coordinates
        :param coord: Coordinate of the point(s) at which to interpolate
        either in grid or absolute coordinates
        :type coord: list, tuple, numpy.array
        :param grid_space: true if the coordinates are expressed in
        grid space (indices can be float) as opposed to model space
        :param mode: {'reflect', 'constant', 'nearest', 'mirror', 'wrap'},
        optional

        The `mode` parameter determines how the input array is extended
        beyond its boundaries. Default is 'constant'. Behavior for each valid
        value is as follows:

        'reflect' (`d c b a | a b c d | d c b a`)
            The input is extended by reflecting about the edge of the last
            pixel.

        'constant' (`k k k k | a b c d | k k k k`)
            The input is extended by filling all values beyond the edge with
            the same constant value, defined by the `cval` parameter.

        'nearest' (`a a a a | a b c d | d d d d`)
            The input is extended by replicating the last pixel.

        'mirror' (`d c b | a b c d | c b a`)
            The input is extended by reflecting about the center of the last
            pixel.

        'wrap' (`a b c d | a b c d | a b c d`)
            The input is extended by wrapping around to the opposite edge.

        :param order: int, optional
            The order of the spline interpolation, default is 3.
            The order has to be in the range 0-5.
        :type order: int

        :type grid_space: bool
        :rtype: numpy.array
        """

        coord = np.array(coord)

        if not grid_space:
            coord = self.transform_to(coord)

        if len(coord.shape) < 2:
            coord = coord[:, np.newaxis]

        try:
            return map_coordinates(self.data, coord, mode=mode, order=order,
                                   **kwargs)
        except Exception as e:
            # logger.warning(e)
            # logger.info('transposing the coordinate array')
            return map_coordinates(self.data, coord.T, mode=mode, order=order,
                                   **kwargs)

    def fill_from_z_gradient(self, vals, zvals):
        data = self.data
        origin = self.origin
        zinds = [int(self.transform_to([origin[0], origin[1], z_])[2]) for z_
                 in zvals]
        # print(zinds, origin)

        data[:, :, zinds[0]:] = vals[0]
        data[:, :, :zinds[-1]] = vals[-1]

        for i in range(len(zinds) - 1):
            # print(i)
            fill = np.linspace(vals[i + 1], vals[i], zinds[i] - zinds[i + 1])
            data[:, :, zinds[i + 1]:zinds[i]] = fill

    def get_grid_point_coordinates(self, mesh_grid=True):
        """
        """
        x = []
        for i, (dimension, spacing) in \
                enumerate(zip(self.data.shape, self.spacing)):
            v = np.arange(0, dimension) * spacing + self.origin[0]
            x.append(v)

        if not mesh_grid:
            return tuple(x)

        if len(x) == 2:
            return tuple(np.meshgrid(x[0], x[1]))

        if len(x) == 3:
            return tuple(np.meshgrid(x[0], x[1], x[2]))

    def write(self, filename, format='PICKLE', **kwargs):
        """
        write the grid to disk
        :param filename: full path to the file to be written
        :type filename: str
        :param format: output file format
        :type format: str
        """
        format = format.upper()

        Path(filename).parent.mkdirs(parent=True, exist_ok=True)

        if format not in ENTRY_POINTS['grid'].keys():
            raise TypeError(f'format {format} is currently not supported '
                            f'for Grid objects')

        format_ep = ENTRY_POINTS['grid'][format]
        write_format = load_entry_point(format_ep.dist.key,
                                        f'uquake.io.grid.{format_ep.name}',
                                        'writeFormat')

        return write_format(self, filename, **kwargs)

    @classmethod
    def read(cls, filename, format='PICKLE'):
        read_grid(filename, format=format)

    def plot_1D(self, x, y, z_resolution, grid_space=False,
                inventory=None, reverse_y=True):
        """

        :param x: x location
        :param y: y location
        :param z_resolution_m: z resolution in grid units
        :param grid_space:
        :return:
        """

        if not grid_space:
            x, y, z = self.transform_from([x, y, 0])

        zs = np.arange(self.origin[2], self.corner[2], z_resolution)

        coords = []
        for z in zs:
            coords.append(np.array([x, y, z]))

        values = self.interpolate(coords, grid_space=grid_space)

        plt.plot(values, zs)
        if reverse_y:
            plt.gca().invert_yaxis()

        if (inventory):
            z_stas = []
            for network in inventory:
                for station in network:
                    loc = station.loc
                    z_stas.append(loc[2])

            plt.plot([np.mean(values)] * len(z_stas), z_stas, 'kv')



            plt.plot()

            plt.plot()
        plt.show()

    def smooth(self,
               sigma: Union[np.ndarray, List[Union[int, float]], Union[int, float]],
               grid_space: bool = False,
               preserve_statistic: bool = True,
               in_place: bool = True) -> Grid:
        """
        Smooth the data using a Gaussian filter.

        :param sigma: Standard deviation for Gaussian filter.
                      Can be a single number or a numpy array.
        :param grid_space: If True, smoothing is performed in grid space.
        :param preserve_statistic: If True, original mean and standard deviation are preserved.
        :param in_place: If True, modifies the object's data in place. Otherwise, returns a new object.
        :return: self if in_place is True, otherwise a new instance of Grid.
        """

        # Use self's properties or create a copy for a new object.
        obj = self if in_place else deepcopy(
            self)  # You need to import deepcopy from the copy module

        original_mean = np.mean(obj.data)
        original_std = np.std(obj.data)

        if isinstance(sigma, (int, float)):
            sigma = np.array([sigma] * obj.data.ndim)

        if isinstance(sigma, list):
            sigma = np.array(sigma)

        if not grid_space:
            sigma = sigma / obj.spacing

        smoothed_data = gaussian_filter(obj.data, sigma)

        if preserve_statistic:
            smoothed_mean = np.mean(smoothed_data)
            smoothed_std = np.std(smoothed_data)
            obj.data = (smoothed_data - smoothed_mean) * (
                        original_std / smoothed_std) + original_mean
        else:
            obj.data = smoothed_data

        return obj

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return list(self.data.shape)

    @property
    def dims(self):
        return self.shape

    @property
    def dimensions(self):
        return self.shape

    @property
    def corner(self):
        return np.array(self.origin) + np.array(self.shape) * \
               np.array(self.spacing)

    @property
    def checksum(self):
        """
        Compute the SHA-256 checksum of a numpy array.
        """
        if self.data.flags['C_CONTIGUOUS']:
            data_bytes = self.data.data
        else:
            contiguous_data = np.ascontiguousarray(self.data)
            data_bytes = contiguous_data.data
        return sha256(data_bytes).hexdigest()

    def generate_random_inventory_in_grid(self, num_station: int = 1,
                                          ratio_uni_tri: float = 3) -> Inventory:
        """
        Generate a random inventory within the grid.

        :param num_station: Number of stations to place.
        :type num_station: int
        :param ratio_uni_tri: Ratio of uniaxial to tri-axial stations.
        :type ratio_uni_tri: float
        :return: Generated inventory.
        :rtype: Inventory
        """

        # Create a Network
        network = Network(code='XX', description='Generated Network')

        # Generate random points
        points = self.generate_random_points_in_grid(n_points=num_station)

        # Calculate number of uniaxial and triaxial stations based on the ratio

        for i, point in enumerate(points):
            # Create coordinates and a Station
            coordinates = Coordinates(point[0], point[1], point[2],
                                      coordinate_system=self.coordinate_system)

            station = Station(code=f"ST{i:02d}", coordinates=coordinates)

            # Determine if this station should be uniaxial or triaxial
            is_uni = random.choices([True, False], [ratio_uni_tri, 1])[0]

            # Create Channel(s)
            if is_uni:
                channel = Channel(code="HHZ", location_code="00",
                                  coordinates=coordinates)
                station.channels.append(channel)
            else:
                for axis in ["Z", "N", "E"]:
                    channel = Channel(code=f"HH{axis}", location_code=f"00",
                                      coordinates=coordinates)
                    station.channels.append(channel)

            # Append Station to Network
            network.stations.append(station)

        # Create an Inventory
        inventory = Inventory(networks=[network])

        return inventory

    def generate_random_catalog_in_grid(self, num_event: int = 1):
        pass


def angles(travel_time_grid):
    """
    This function calculate the take off angle and azimuth for every grid point
    given a travel time grid calculated using an Eikonal solver
    :param travel_time_grid: travel_time grid
    :type travel_time_grid: ~uquake.core.grid.Grid.
    :rparam: azimuth and takeoff angles grids
    .. Note: The convention for the takeoff angle is that 0 degree is down.
    """

    gds_tmp = np.gradient(travel_time_grid.data)
    gds = [-gd for gd in gds_tmp]

    tmp = np.arctan2(gds[0], gds[1])  # azimuth is zero northwards
    azimuth = travel_time_grid.copy()
    azimuth.type = 'ANGLE'
    azimuth.data = tmp

    hor = np.sqrt(gds[0] ** 2 + gds[1] ** 2)
    tmp = np.arctan2(hor, -gds[2])
    # takeoff is zero pointing down
    takeoff = travel_time_grid.copy()
    takeoff.type = 'ANGLE'
    takeoff.data = tmp

    return azimuth, takeoff


def ray_tracer(travel_time_grid, start, grid_space=False, max_iter=1000,
               arrival_id=None, earth_model_id=None,
               network: str=None):
    """
    This function calculates the ray between a starting point (start) and an
    end point, which should be the seed of the travel_time grid, using the
    gradient descent method.
    :param travel_time_grid: a travel time grid
    :type travel_time_grid: TTGrid
    :param start: the starting point (usually event location)
    :type start: tuple, list or numpy.array
    :param grid_space: true if the coordinates are expressed in
    grid space (indices can be fractional) as opposed to model space
    (x, y, z)
    :param max_iter: maximum number of iteration
    :param arrival_id: id of the arrival associated to the ray if
    applicable
    :type arrival_id: uquake.core.event.ResourceIdentifier
    :param earth_model_id: velocity/earth model id.
    :type earth_model_id: uquake.core.event.ResourceIdentifier
    :param network: network information
    :type network: str
    :rtype: numpy.array
    """

    from uquake.core.event import Ray

    interpolation_order = 1

    if grid_space:
        start = np.array(start)
        start = travel_time_grid.transform_from(start)

    origin = travel_time_grid.origin
    spacing = travel_time_grid.spacing
    end = np.array(travel_time_grid.seed)
    start = np.array(start)

    # calculating the gradient in every dimension at every grid points
    gds = [Grid(gd, origin=origin, spacing=spacing)
           for gd in np.gradient(travel_time_grid.data)]

    dist = np.linalg.norm(start - end)
    cloc = start  # initializing cloc "current location" to start
    spacing = np.linalg.norm(spacing)
    gamma = spacing / 2  # gamma is set to half the grid spacing. This
    # should be
    # sufficient. Note that gamma is fixed to reduce
    # processing time.
    nodes = [start]

    iter_number = 0
    while np.all(dist > spacing / 2):
        if iter_number > max_iter:
            break

        # if dist < spacing * 4:
        #     gamma = spacing / 4

        gvect = np.array([gd.interpolate(cloc, grid_space=False,
                                         order=interpolation_order)[0]
                          for gd in gds])

        if np.linalg.norm(gvect) == 0:
            break

        dr = gamma * gvect / (np.linalg.norm(gvect) + 1e-8)

        if np.linalg.norm(dr) < gamma / 2:
            dr = (dr / np.linalg.norm(dr)) * gamma / 2

        cloc = cloc - dr

        nodes.append(cloc)
        dist = np.linalg.norm(cloc - end)

        iter_number += 1

    nodes.append(end)

    tt = travel_time_grid.interpolate(start, grid_space=False,
                                      order=interpolation_order)[0]

    az = travel_time_grid.to_azimuth_point(start, grid_space=False,
                                           order=interpolation_order)
    toa = travel_time_grid.to_takeoff_point(start, grid_space=False,
                                            order=interpolation_order)

    ray = Ray(nodes=nodes, waveform_id=travel_time_grid.waveform_id,
              arrival_id=arrival_id, phase=travel_time_grid.phase,
              azimuth=az, takeoff_angle=toa, travel_time=tt,
              earth_model_id=earth_model_id, network=network)

    return ray
