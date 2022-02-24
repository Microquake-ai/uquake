# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: nlloc.py
#  Purpose: plugin for reading and writing GridData object into various format 
#   Author: uquake development team
#    Email: devs@uquake.org
#
# Copyright (C) 2016 uquake development team
# --------------------------------------------------------------------
"""
plugin for reading and writing GridData object into various format 

:copyright:
    uquake development team (devs@uquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from pathlib import Path
from uuid import uuid4
from ...grid.nlloc import (valid_float_types, VelocityGrid3D, TTGrid,
                           AngleGrid, NLLocGrid, __default_float_type__)
import os


def read_pickle(filename, protocol=-1, **kwargs):
    """
    read grid saved in PICKLE format into a GridData object
    :param filename: full path to the filename
    :type filename: str
    :rtype: ~uquake.core.data.grid.Grid
    """
    import pickle
    return pickle.load(open(filename, 'rb'))


def write_pickle(grid, filename, protocol=-1, **kwargs):
    """
    write a GridData object to disk in pickle (.pickle or .npy extension)
    format
    using the pickle module
    :param grid: grid to be saved
    :type grid: ~uquake.core.data.grid.GridData
    :param filename: full path to file with extension
    :type filename: str
    :param protocol: pickling protocol level
    :type protocol: int
    """
    import pickle
    with open(filename, 'wb') as of:
        pickle.dump(grid, of, protocol=protocol)

    return True


def read_hdf5(filename, **kwargs):
    """
    read a grid file in hdf5 into a uquake.core.data.grid.GridCollection
    object
    :param filename: filename
    :param kwargs: additional keyword argument passed from wrapper.
    :return: uquake.core.data.grid.GridCollection
    """
    pass


def write_csv(grid, filename, **kwargs):
    """
    Write a GridData object to disk in uquake csv format
    :param grid: grid to be saved
    :param filename: full path to file with extension
    :return:
    """
    data = grid.data
    shape = grid.shape
    origin = grid.origin
    spacing = grid.spacing

    v = grid.get_grid_point_coordinates()

    flat_data = data.reshape(np.product(shape))

    with open(filename, 'w') as f_out:
        if data.ndim == 3:
            f_out.write('uquake grid\n')
            f_out.write(f'spacing: {spacing}')
            f_out.write(f'origin: {origin}')
            f_out.write(f'shape: {shape}')
            f_out.write('x,y,z,value\n')

            for k in range(grid.shape[2]):
                for j in range(grid.shape[1]):
                    for i in range(grid.shape[0]):
                        f_out.write(f'{v[0][i]},{v[1][j]},{v[2][k]},'
                                    f'{grid.data[i, j, k]}\n')

        elif data.ndim == 2:
            f_out.write('uquake grid\n')
            f_out.write(f'spacing: {spacing}')
            f_out.write(f'origin: {origin}')
            f_out.write(f'shape: {shape}')
            f_out.write('x,y,value\n')
            for j in range(grid.shape[1]):
                for i in range(grid.shape[0]):
                    f_out.write(f'{v[0][i]},{v[0][j]},'
                                f'{grid.data[i, j]}\n')

    return True


def read_csv(filename, *args, **kwargs):
    """
    Read a grid save in uquake CSV format
    :param filename: path to file
    :param args:
    :param kwargs:
    :return:
    """
    pass


def write_vtk(grid, filename, **kwargs):
    """
    write a GridData object to disk in VTK format (Paraview, MayaVi2,
    etc.) using
    the pyevtk module.
    param filename: full path to file with the extension. Note that the
    extension for vtk image data (grid data) is usually .vti. 
    :type filename; str
    :param grid: grid to be saved
    :type grid: ~uquake.core.data.grid.GridData
    .. NOTE:
        see the imageToVTK function from the pyevtk.hl module for more
        information on possible additional paramter.
    """
    import vtk

    if filename[-4:] in ['.vti', '.vtk']:
        filename = filename[:-4]

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(grid.shape)
    image_data.SetSpacing(grid.spacing)
    image_data.SetOrigin(grid.origin)
    image_data.AllocateScalars(vtk.VTK_FLOAT, 1)

    if grid.ndim == 3:
        for z in range(grid.shape[2]):
            for y in range(grid.shape[1]):
                for x in range(grid.shape[0]):
                    image_data.SetScalarComponentFromFloat(x, y, z, 0,
                                                           grid.data[x, y, z])

    if grid.ndim == 2:
        for y in range(grid.shape[1]):
            for x in range(grid.shape[0]):
                image_data.SetScalarComponentFromFloat(x, y, 0,
                                                       grid.data[x, y])

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(f'{filename}.vti')
    writer.SetInputData(image_data)
    writer.Write()
    return True


def read_vtk(filename, *args, **kwargs):
    pass


def read_nlloc(filename, float_type=__default_float_type__):
    """
    read two parts NLLoc files
    :param filename: filename
    :param float_type: float type as defined in NLLoc grid documentation
    """

    if filename.split('.')[-1] in ['hdr', 'buf', 'mid']:
        filename = filename[:-4]

    header_file = Path(f'{filename}.hdr')
    # header_file = Path(path) / f'{base_name}.hdr'
    with open(header_file, 'r') as in_file:
        line = in_file.readline()
        line = line.split()
        shape = tuple([int(line[0]), int(line[1]), int(line[2])])
        origin = np.array([float(line[3]), float(line[4]),
                           float(line[5])]) * 1000
        spacing = np.array([float(line[6]), float(line[7]),
                            float(line[8])]) * 1000

        grid_type = line[9]
        grid_unit = 'METER'

        line = in_file.readline()

        if grid_type in ['ANGLE', 'ANGLE2D', 'TIME', 'TIME2D']:
            line = line.split()
            seed_label = line[0]
            seed = (float(line[1]) * 1000,
                    float(line[2]) * 1000,
                    float(line[3]) * 1000)

        else:
            seed_label = None
            seed = None

    buf_file = Path(f'{filename}.buf')
    # buf_file = Path(path) / f'{base_name}.buf'
    if float_type == 'FLOAT':
        data = np.fromfile(buf_file,
                           dtype=np.float32)
    elif float_type == 'DOUBLE':
        data = np.fromfile(buf_file,
                           dtype=np.float64)
    else:
        msg = f'float_type = {float_type} is not valid\n' \
              f'float_type should be one of the following valid float ' \
              f'types:\n'
        for valid_float_type in valid_float_types:
            msg += f'{valid_float_type}\n'
        raise ValueError(msg)

    data = data.reshape(shape)

    if '.P.' in filename:
        phase = 'P'
    else:
        phase = 'S'

    # reading the model id file
    mid_file = Path(f'{filename}.mid')
    if mid_file.exists():
        with open(mid_file, 'r') as mf:
            model_id = mf.readline().strip()

    else:
        model_id = str(uuid4())

        # (self, base_name, data_or_dims, origin, spacing, phase,
        #  seed=None, seed_label=None, value=0,
        #  grid_type='VELOCITY_METERS', grid_units='METER',
        #  float_type="FLOAT", model_id=None):

    network_code = filename.split(os.path.sep)[-1].split('.')[0]
    if grid_type in ['VELOCITY', 'VELOCITY_METERS']:
        return VelocityGrid3D(network_code, data, origin, spacing, phase=phase,
                              model_id=model_id)

    elif grid_type == 'TIME':
        return TTGrid(network_code, data, origin, spacing, seed,
                      seed_label, phase=phase, model_id=model_id)

    elif grid_type == 'ANGLE':
        return AngleGrid(network_code, data, origin, spacing, seed,
                         seed_label, angle_type='AZIMUTH', phase=phase,
                         model_id=model_id)

    else:
        grid = NLLocGrid(data, origin, spacing, phase,
                         grid_type=grid_type, model_id=model_id,
                         grid_units=grid_unit)

        if grid_type == 'SLOW_LEN':
            return VelocityGrid3D.from_slow_len(grid, network_code)

        return grid
