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
# Filename: extended.py
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
from ...grid.extended import (valid_float_types, VelocityGrid3D, VelocityGridEnsemble,
                              TTGrid, TravelTimeEnsemble, AngleGrid, TypedGrid,
                              __default_float_type__, Phases, FloatTypes, GridTypes,
                              GridUnits, Seed)
from uquake.core.event import ResourceIdentifier
from uquake.core.coordinates import CoordinateSystem, Coordinates
import h5py
from datetime import datetime
from typing import Union, List, Tuple, Dict, Any, Optional
import os
from enum import Enum


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


def write_hdf5(grid, filename, **kwargs):
    """
    write a GridData object to disk in hdf5 format using the h5py module
    """
    write_grid_to_hdf5(grid, filename, **kwargs)


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
    if (float_type.value == 'FLOAT') | (float_type.value == 'float32'):
        data = np.fromfile(buf_file,
                           dtype=np.float32)
    elif (float_type.value == 'DOUBLE') | (float_type.value == 'float64'):
        data = np.fromfile(buf_file,
                           dtype=np.float64)
    else:
        msg = f'float_type = {float_type} is not valid\n' \
              f'float_type should be one of the following valid float ' \
              f'types:\n'
        for valid_float_type in valid_float_types:
            msg += f'{valid_float_type}\n'
        raise ValueError(msg)

    if data.shape != shape:
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
        grid = TypedGrid(data, origin, spacing, phase,
                         grid_type=grid_type, model_id=model_id,
                         grid_units=grid_unit)

        if grid_type == 'SLOW_LEN':
            return VelocityGrid3D.from_slow_len(grid, network_code)

        return grid


class HDF5Compression(Enum):
    """
    Enumerates the compression types supported by the HDF5 format.
    """
    NONE = None
    GZIP = 'gzip'
    LZF = 'lzf'
    SZIP = 'szip'
    BZIP2 = 'bzip2'


def write_grid_to_hdf5(
        file_name: str,
        grid_ensemble: Union[VelocityGridEnsemble, TravelTimeEnsemble],
        compression: Optional[HDF5Compression] = HDF5Compression.NONE,
        compression_opts: Optional[Union[int, tuple]] = None):
    """
    Writes a VelocityGrid3D object to an HDF5 file with the specified structure.

    :param file_name: The name of the HDF5 file.
    :param grid_ensemble: The VelocityGridEnsemble or TravelTimeEnsemble object to be
    written to the file.
    :param compression: The type of compression to use for storing the dataset. If None,
    no compression is applied.
                        Accepts 'gzip', 'lzf', or 'szip'. Defaults to None.
    :param compression_opts: For 'gzip', an integer from 0 to 9 indicating the
                             compression level.
                             For 'szip', it can be a tuple specifying the compression
                             options.
                             Not applicable for 'lzf'. Defaults to None.
    """

    # Open the file and create the dataset with the specified compression

    with h5py.File(file_name, 'w') as h5file:

        # Create the phase groups
        for phase in Phases:

            if isinstance(grid_ensemble, TravelTimeEnsemble):
                if len(grid_ensemble.select(phases=[phase.value])) == 0:
                    continue
                phase_group = h5file.create_group(f'Phase {phase.value}')
                tmp_ensemble = grid_ensemble.select(phases=[phase.value])
                for time_grid in tmp_ensemble:
                    instrument_group = phase_group.create_group(
                        time_grid.instrument_code)
                    instrument_group.attrs[
                        'Velocity_Model_ID'] = str(time_grid.velocity_model_id)

                    # seed = time_grid.seed.__dict__
                    # seed['coordinates'] = seed['coordinates'].to_json()

                    instrument_group.attrs['station'] = time_grid.seed.station
                    instrument_group.attrs['location'] = time_grid.seed.location
                    instrument_group.attrs['coordinates'] = \
                        time_grid.seed.coordinates.loc
                    instrument_group.attrs['coordinate_System'] = \
                        time_grid.seed.coordinates.coordinate_system.value
                    write_grid_attributes_and_data_hdf5(instrument_group, time_grid,
                                                        compression, compression_opts)

            elif isinstance(grid_ensemble, VelocityGridEnsemble):
                if grid_ensemble[phase.value] is None:
                    continue
                phase_group = h5file.create_group(f'Phase {phase.value}')
                v_grid = grid_ensemble[phase.value]
                write_grid_attributes_and_data_hdf5(phase_group, v_grid, compression,
                                                    compression_opts)


def write_grid_attributes_and_data_hdf5(
        group, grid: VelocityGrid3D,
        compression: Optional[HDF5Compression] = HDF5Compression.NONE,
        compression_opts: Optional[Union[int, tuple]] = None):
    """
    Writes the attributes and data of a VelocityGrid3D object to an HDF5 group.
    :param group:
    :param grid:
    :param compression:
    :param compression_opts:
    :return:
    """

    group.attrs['Creation Timestamp'] = datetime.utcnow().isoformat() + "Z"
    group.attrs['Network'] = grid.network_code
    group.attrs['Origin'] = grid.origin
    group.attrs['Spacing'] = grid.spacing
    group.attrs['Dimensions'] = grid.data.shape
    group.attrs['Grid ID'] = str(grid.grid_id)
    group.attrs['Schema Version'] = "1.0"
    group.attrs['Type'] = grid.grid_type.value
    group.attrs['Units'] = grid.grid_units.value  # or the appropriate units
    group.attrs['float_type'] = grid.float_type.value
    group.attrs['Coordinate_System'] = grid.coordinate_system.value
    group.attrs['Label'] = grid.label

    # Add data set
    data = grid.data.astype(grid.float_type.value)
    data_set = group.create_dataset("Data", data=data, dtype=grid.float_type.value,
                                    compression=compression.value,
                                    compression_opts=compression_opts)
    data_set.attrs['Checksum'] = grid.checksum


def read_velocity_grid_from_hdf5(file_name: str):
    """
    Reads an HDF5 file and converts it into a VelocityGridEnsemble
    object.

    :param file_name: Name of the HDF5 file to read from.
    :return: VelocityGrid3D or VelocityGridEnsemble object initialized with data from the
    HDF5 file.
    """

    with h5py.File(file_name, 'r') as h5file:
        # Initialize a dict to hold VelocityGrid3D objects for each phase
        velocity_grids = {}

        # Iterate over potential phase groups
        for phase in Phases:
            phase_group_name = f"/Phase {phase.value}"
            if phase_group_name in h5file:
                phase_group = h5file.get(phase_group_name)

                attributes = get_attribute_and_data_from_group(phase_group)

                attributes['phase'] = phase

                # Read the data set

                # Create the VelocityGrid3D object for the phase
                velocity_grid = VelocityGrid3D(**attributes)
                velocity_grids[phase] = velocity_grid

        # Check if both P and S wave data are present
        if Phases.P in velocity_grids and Phases.S in velocity_grids:
            # Return a VelocityGridEnsemble object containing both
            return VelocityGridEnsemble(velocity_grids[Phases.P],
                                        velocity_grids[Phases.S])
        elif Phases.P in velocity_grids:
            # Return the P-wave VelocityGrid3D object
            return velocity_grids[Phases.P]
        elif Phases.S in velocity_grids:
            # Return the S-wave VelocityGrid3D object
            return velocity_grids[Phases.S]
        else:
            # Handle the case where no recognized phase data is found
            raise ValueError("No valid phase data found in the HDF5 file.")


def get_attribute_and_data_from_group(group):

    data = np.array(group['Data'][:]).astype(
        group.attrs['float_type'])
    # Extract attributes
    attributes = {
        'network_code': group.attrs['Network'],
        'data_or_dims': data,
        'origin': group.attrs['Origin'],
        'spacing': group.attrs['Spacing'],
        'float_type': FloatTypes(group.attrs['float_type']),
        'grid_id': ResourceIdentifier(group.attrs['Grid ID']),
        'type': GridTypes(group.attrs['Type']),
        'units': GridUnits(group.attrs['Units']),
        'coordinate_system':
            CoordinateSystem(group.attrs['Coordinate_System']),
        'label': group.attrs['Label'],
    }

    return attributes


def read_travel_time_ensemble_from_hdf5(file_name):

    travel_time_grids = []
    with h5py.File(file_name, 'r') as f:
        for phase in f:
            for instrument_id in f[phase]:
                instrument_group = f[phase][instrument_id]
                attributes = get_attribute_and_data_from_group(instrument_group)
                attributes['phase'] = Phases.P if ' P' in phase else Phases.S
                attributes['velocity_model_id'] = instrument_group.attrs[
                    'Velocity_Model_ID']
                station_code = instrument_group.attrs['Seed_Station']
                location_code = instrument_group.attrs['Seed_Location']
                x, y, z = instrument_group.attrs['Seed_Coordinates']
                coordinate_system = CoordinateSystem(instrument_group.attrs[
                                                         'Seed_Coordinate_System'])
                coordinates = Coordinates(x, y, z, coordinate_system=coordinate_system)
                seed = Seed(station_code, location_code, coordinates)
                attributes['seed'] = seed
                attributes.pop('type')
                attributes.pop('units')
                tt_grid = TTGrid(**attributes)
                travel_time_grids.append(tt_grid)

    return TravelTimeEnsemble(travel_time_grids)


