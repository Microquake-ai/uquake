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
from uuid import uuid4
from .logging import logger
from copy import deepcopy
from pkg_resources import load_entry_point
from .util import ENTRY_POINTS
from pathlib import Path
from scipy.ndimage.interpolation import map_coordinates


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


class Grid:
    """
    Object containing a regular grid
    """

    def __init__(self, data_or_dims, spacing=None, origin=None,
                 resource_id=None, value=0):

        """
        can hold both 2 and 3 dimensional grid
        :param data_or_dims: either a numpy array or a tuple/list with the grid
        dimensions. If grid dimensions are specified, the grid is initialized
        with value
        :param spacing: Spacing
        :type spacing: typle
        :param origin: tuple, list or array containing the origin of the grid
        :type origin: tuple
        :param resource_id: unique identifier for the grid, if set to None,
        :param value: value to fill the grid should dims be specified
        :type value:
        uuid4 is used to define a unique identifier.
        :type uuid4: str
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
        cls.grid = cls.__init__(data, spacing=spacing, origin=origin)

    @classmethod
    def from_ocs(cls, origin, corner, spacing, val=0):
        """
        create a grid from origin, corner and spacing
        :param origin: grid origin (e.g., lower left corner for 2D grid)
        :type origin: tuple
        :param corner: grid upper (e.g., upper right corner for 2D grid)
        :type corner: tuple
        :param spacing: spacing between the grid nodes
        :type spacing: float
        :param val: constant value with which to fill the grid
        :param buf: buffer around the grid in fraction of grid size
        """
        origin2 = origin
        corner2 = corner

        gshape = tuple([int(np.ceil((c - o) / spacing))
                        for o, c in zip(origin2, corner2)])
        data = np.ones(gshape) * val
        cls.__init__(data, spacing=spacing, origin=origin)
        cls.fill_homogeneous(val)
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

    def transform_from(self, values):
        """
        transform grid space coordinates into model space coordinates
        :param values: tuple of grid space coordinates
        :type values: tuple
        :rtype: tuple
        """
        return values * self.spacing + self.origin

    def check_compatibility(self, other):
        """
        check if two grids are compatible, i.e., have the same shape, spacing
        and origin
        """
        return (self.shape == other.shape) and \
               (self.spacing == other.spacing) and \
            np.all(self.origin == other.origin)

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

    def generate_points(self, pt_spacing=None):
        """
        """
        # if pt_spacing is None:
        ev_spacing = self.spacing

        dimensions = np.array(self.shape) * self.spacing / ev_spacing

        xe = np.arange(0, dimensions[0]) * ev_spacing + self.origin[0]
        ye = np.arange(0, dimensions[1]) * ev_spacing + self.origin[1]
        ze = np.arange(0, dimensions[2]) * ev_spacing + self.origin[2]

        Xe, Ye, Ze = np.meshgrid(xe, ye, ze)

        Xe = Xe.reshape(np.prod(Xe.shape))
        Ye = Ye.reshape(np.prod(Ye.shape))
        Ze = Ze.reshape(np.prod(Ze.shape))
        return Xe, Ye, Ze

    def write(self, filename, format='PICKLE', **kwargs):
        """
        write the grid to disk
        :param filename: full path to the file to be written
        :type filename: str
        :param format: output file format
        :type format: str
        """
        format = format.upper()
        if format not in ENTRY_POINTS['grid'].keys():
            logger.error('format %s is not currently supported for Grid '
                         'objects' % format)
            return

        format_ep = ENTRY_POINTS['grid'][format]
        write_format = load_entry_point(format_ep.dist.key,
                                        'microquake.plugin.grid.%s' % format_ep.name, 'writeFormat')

        write_format(self, filename, **kwargs)

    def interpolate(self, coord, grid_coordinate=True, mode='nearest',
                    order=1, **kwargs):
        """
        This function interpolate the values at a given point expressed
        either in grid or absolute coordinates
        :param coord: Coordinate of the point(s) at which to interpolate
        either in grid or absolute coordinates
        :type coord: list, tuple, numpy.array
        :param grid_coordinate: true if the coordinates are expressed in
        grid space (indices can be float) as opposed to model space
        :type grid_coordinate: bool
        :rtype: numpy.array
        """

        coord = np.array(coord)

        if not grid_coordinate:
            coord = self.transform_to(coord)

        if len(coord.shape) < 2:
            coord = coord[:, np.newaxis]

        try:
            return map_coordinates(self.data, coord, mode=mode, order=order,
                                   **kwargs)
        except Exception as e:
            logger.warning(e)
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

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    @property
    def dims(self):
        return self.data.shape

    @property
    def dimensions(self):
        return self.data.shape


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


def ray_tracer(travel_time, start, grid_coordinates=False, max_iter=1000):
    """
    This function calculates the ray between a starting point (start) and an
    end point, which should be the seed of the travel_time grid, using the
    gradient descent method.
    :param travel_time: travel time grid with a seed defined
    :type travel_time: ~uquake.core.grid.Grid
    :param start: the starting point (usually event location)
    :type start: tuple, list or numpy.array
    :param grid_coordinates: if true grid coordinates (indices,
    not necessarily integer are used, else real world coordinates are used
    (x, y, z) (Default value False)
    :param max_iter: maximum number of iteration
    :rtype: numpy.array
    """

    from .event import Ray

    if grid_coordinates:
        start = np.array(start)
        start = travel_time.transform_from(start)

    origin = travel_time.origin
    spacing = travel_time.spacing
    end = np.array(travel_time.seed)
    start = np.array(start)

    # calculating the gradient in every dimension at every grid points
    gds = [Grid(gd, origin=origin, spacing=spacing)
           for gd in np.gradient(travel_time.data)]

    dist = np.linalg.norm(start - end)
    cloc = start  # initializing cloc "current location" to start
    gamma = spacing / 2    # gamma is set to half the grid spacing. This
    # should be
    # sufficient. Note that gamma is fixed to reduce
    # processing time.
    nodes = [start]

    iter_number = 0
    while dist > spacing / 2:
        if iter_number > max_iter:
            break

        if dist < spacing * 4:
            gamma = spacing / 4

        gvect = np.array([gd.interpolate(cloc, grid_coordinate=False,
                                         order=1)[0] for gd in gds])

        cloc = cloc - gamma * gvect / np.linalg.norm(gvect)
        nodes.append(cloc)
        dist = np.linalg.norm(cloc - end)

        iter_number += 1

    nodes.append(end)

    ray = Ray(nodes=nodes)

    return ray



