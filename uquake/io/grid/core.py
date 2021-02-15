# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: grid.py
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
    Write a GridData object to disk in Microquake csv format
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