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

__valid_phases__ = ('P', 'S')

__valid_grid_types__ = (
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


class ValidTypes:

    def __init__(self, valid_types):

        self.valid_type = valid_types

    def __str__(self):
        return '\n'.join(self.valid_type)
    def __repr__(self):
        return self.__str__(self)


valid_grid_types = ValidTypes(__valid_grid_types__)


class Grid:
    """
    Object containing a regular grid
    """

    def __init__(self, data, spacing=None, origin=None, resource_id=None):

        """
        can hold both 2 and 3 dimensional grid
        :param data: Grid
        :type numpy.array:
        :param spacing: Spacing
        :type tuple:
        :param origin: tuple, list or array containing the origin of the grid
        :type tuple:
        :param resource_id: unique identifier for the grid, if set to None,
        uuid4 is used to define a unique identifier.
        :type str:
        """

        self.data = np.array(data)

        if resource_id is None:
            self.resource_id = str(uuid4())
        else:
            self.resource_id = resource_id

        if origin is None:
            self.origin = np.zeros(len(data.shape))
        else:
            origin = np.array(origin)
            if origin.shape[0] == len(data.shape):
                self.origin = origin
            else:
                logger.error(f'origin shape should be {len(data.shape)}')
                raise ValueError

        if spacing is None:
            self.spacing = np.ones(len(data.shape))
        else:
            spacing = np.array(spacing)
            if spacing.shape[0] == len(data.shape):
                self.spacing = spacing
            else:
                logger.error(f'spacing shape should be {len(data.shape)}')
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

    @property
    def n_dimensions(self):
        return np.min(self.data.shape)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        repr_str = """
        spacing: %s
        origin : %s
        shape  : %s
        seed   : %s
        type   : %s
        """ % (self.spacing, self.origin, self.shape, self.seed, self.type)
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

    def in_grid(self, point):
        """
        Check if a point is inside the grid
        :param point: the point to check
        :type point: tuple, list or numpy array
        :returns: True if point is inside the grid
        :rtype: bool
        """
        corner1 = self.origin
        corner2 = self.origin + self.spacing * np.array(self.shape)

        return np.all((point >= corner1) & (point <= corner2))

    def fill_homogeneous(self, val):
        self.data = np.ones(self.data.shape) * val

    def copy(self):
        return deepcopy(self)

    def get_grid_point_coordinates(self, meshgrid=True):
        """
        """
        x = []
        for i, (dimension, spacing) in \
                enumerate(zip(self.data.shape, self.spacing)):
            v = np.arange(0, dimension) * spacing + self.origin[0]
            x.append(v)

        if not meshgrid:
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

        if format not in ENTRY_POINTS['grid'].keys():
            logger.error('format %s is not currently supported for Grid '
                         'objects' % format)
            return

        format_ep = ENTRY_POINTS['grid'][format]
        write_format = load_entry_point(format_ep.dist.key,
                                        'microquake.plugin.grid.%s' % format_ep.name, 'writeFormat')

        write_format(self, filename, **kwargs)




