# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: grid.py
#  Purpose: plugin for reading and writing GridData object into various format 
#   Author: microquake development team
#    Email: devs@microquake.org
#
# Copyright (C) 2016 microquake development team
# --------------------------------------------------------------------
"""
plugin for reading and writing GridData object into various format 

:copyright:
    microquake development team (devs@microquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
import pandas as pd
from loguru import logger


def read_nll(filename, **kwargs):
    """
    read NLLoc grid file into a GridData object
    :param filename: filename with or without the extension
    :type filename: str
    :rtype: ~microquake.core.data.grid.GridData
    """
    from microquake.core.nlloc import read_NLL_grid
    return read_NLL_grid(filename)


def read_pickle(filename, **kwargs):
    """
    read grid saved in PICKLE format into a GridData object
    :param filename: full path to the filename
    :type filename: str
    :rtype: ~microquake.core.data.grid.GridData
    """
    import numpy as np
    return np.load(filename)


def read_hdf5(filename, **kwargs):
    """
    read a grid file in hdf5 into a microquake.core.data.grid.GridCollection
    object
    :param filename: filename
    :param kwargs: additional keyword argument passed from wrapper.
    :return: microquake.core.data.grid.GridCollection
    """


def write_nll(grid, filename, **kwargs):
    """
    write a GridData object to disk in NLLoc format
    :param filename: full path to file with or without the extension
    :type filename: str
    :param grid: grid to be saved
    :type grid: ~microquake.core.data.grid.GridData
    """

    from microquake.core.nlloc import write_nll_grid
    data = grid.data
    origin = grid.origin
    spacing = grid.spacing
    grid_type = grid.type
    seed = grid.seed
    label = grid.seed_label
    write_nll_grid(filename, data, origin, spacing, grid_type,
        seed=seed, label=label, **kwargs)


def write_pickle(grid, filename, protocol=-1, **kwargs):
    """
    write a GridData object to disk in pickle (.pickle or .npy extension)
    format
    using the pickle module
    :param grid: grid to be saved
    :type grid: ~microquake.core.data.grid.GridData
    :param filename: full path to file with extension
    :type filename: str
    :param protocol: pickling protocol level
    :type protocol: int
    """
    import pickle as pickle
    with open(filename, 'wb') as of:
        pickle.dump(grid, of, protocol=protocol)


def write_csv(grid, filename, **kwargs):
    """
    Write a GridData object to disk in Microquake csv format
    :param grid: grid to be saved
    :param filename: full path to file with extension
    :return:
    """
    data = grid.data
    shape = data.shape
    origin = grid.origin
    spacing = grid.spacing
    grid_type = grid.type
    seed = grid.seed
    seed_label = grid.seed_label
    x = np.arange(0, shape[0]) * spacing + origin[0]
    y = np.arange(0, shape[1]) * spacing + origin[1]
    z = np.arange(0, shape[2]) * spacing + origin[2]

    xg, yg, zg = np.meshgrid(x, y, z)

    flat_data = data.reshape(np.product(shape))
    flat_xg = xg.reshape(np.product(shape))
    flat_yg = yg.reshape(np.product(shape))
    flat_zg = zg.reshape(np.product(shape))

    data_dict = {'value': flat_data,
                 'x': flat_xg,
                 'y': flat_yg,
                 'z': flat_zg}

    with open(filename, 'w') as f_out:
        # writing header
        f_out.write('Microquake grid\n')
        f_out.write(f'grid_type: {grid_type}\n')
        f_out.write(f'spacing: {spacing}')
        f_out.write(f'origin: {origin}')
        f_out.write(f'shape: {shape}')
        f_out.write(f'seed: {seed}\n')
        f_out.write(f'seed_label: {seed_label}\n')
        f_out.write('x, y, z, value\n')
        for f_d, f_x, f_y, f_z in zip(flat_data, flat_xg, flat_yg, flat_zg):
            f_out.write(f'{f_x}, {f_y}, {f_z}, {f_d}\n')


def read_csv(filename, *args, **kwargs):
    """
    Read a microquake grid save in Microquake CSV format
    :param filename: path to file
    :param args:
    :param kwargs:
    :return:
    """
    pass

    # with open(filename, 'r') as f_in:
    #     line = f_in.readline().strip()
    #     if 'Microquake grid' not in line:
    #         logger.error('not a Microquake csv grid')
    #         raise IOError
    #
    #     line = f_in.readline().strip()
    #     grid_type = line.split(':')[1]
    #
    #     line = f_in.readline().strip()
    #     spacing = np.float(line.split(':')[1])
    #
    #     line = f_in.readline().strip()
    #     origin = np.array(eval(line.split(':')[1]))
    #
    #     line = f_in.readline().strip()
    #     shape = np.array(eval(line.split(':')[1]))
    #
    #     line = f_in.readline().strip()
    #     seed = np.array(eval(line.split(':')[1]))
    #
    #     line = f_in.readline().strip()
    #     seed_label = line.split(':')[1]
    #
    #
    # data = np.zeros(shape)


def write_vtk(grid, filename, *args, **kwargs):
    """
    write a GridData object to disk in VTK format (Paraview, MayaVi2,
    etc.) using
    the pyevtk module.
    param filename: full path to file with the extension. Note that the
    extension for vtk image data (grid data) is usually .vti. 
    :type filename; str
    :param grid: grid to be saved
    :type grid: ~microquake.core.data.grid.GridData
    .. NOTE:
        see the imageToVTK function from the pyevtk.hl module for more
        information on possible additional paramter.
    """
    from pyevtk.hl import imageToVTK

    if filename[-4:] in ['.vti', '.vtk']:
        filename = filename[:-4]

    if isinstance(grid.spacing, tuple):
        spacing = grid.spacing[0]
    else:
        spacing = tuple([grid.spacing] * 3)

    origin = tuple(grid.origin)

    cell_data = {grid.type: grid.data}
    imageToVTK(filename, origin, spacing, pointData=cell_data)


