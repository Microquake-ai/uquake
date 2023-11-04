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
                              __default_float_type__, Phases)
import h5py
from datetime import datetime
from typing import Union, List, Tuple, Dict, Any, Optional
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
        grid = TypedGrid(data, origin, spacing, phase,
                         grid_type=grid_type, model_id=model_id,
                         grid_units=grid_unit)

        if grid_type == 'SLOW_LEN':
            return VelocityGrid3D.from_slow_len(grid, network_code)

        return grid


def write_velocity_grid_to_hdf5(
        velocity_grid: Union[VelocityGridEnsemble, VelocityGrid3D], file_name: str):
    """
    Writes a VelocityGrid3D object to an HDF5 file with a specified structure.

    :param velocity_grid: VelocityGridEnsemble or VelocityGrid3D object to be written
    to the file.
    :param file_name: Name of the HDF5 file.
    """
    # Open or create the HDF5 file
    with h5py.File(file_name, 'w') as h5file:
        # Create group based on the phase (P or S)
        if isinstance(velocity_grid, VelocityGrid3D):
            phases = [Phases(velocity_grid.phase)]

        else:
            phases = [Phases.P, Phases.S]

        for phase in phases:

            if isinstance(velocity_grid, VelocityGridEnsemble):
                v_grid = velocity_grid[phase.value]
            else:
                v_grid = velocity_grid

            phase_group = h5file.create_group(f"/Phase {phase.value}")

            # Set attributes
            phase_group.attrs['Grid ID'] = str(v_grid.grid_id)
            phase_group.attrs['Schema Version'] = "1.0"
            phase_group.attrs['Creation Timestamp'] = datetime.utcnow().isoformat() + "Z"
            # ISO 8601 format
            phase_group.attrs['Type'] = "VELOCITY"
            phase_group.attrs['Units'] = "m/s"  # or the appropriate units
            phase_group.attrs['Coordinate System'] = "cartesian"  # or another system
            phase_group.attrs['Data Order'] = "Row-major"
            # or "Column-major" based on the data
            phase_group.attrs['Origin'] = v_grid.origin
            phase_group.attrs['Spacing'] = v_grid.spacing
            phase_group.attrs['Dimensions'] = v_grid.data.shape
            phase_group.attrs['Compression'] = "None"
            # or describe the compression if used

        # Add data set
            data = v_grid.data.astype(v_grid.float_type.value)
            data_set = phase_group.create_dataset("Data", data=data,
                                                  dtype=v_grid.float_type.value)
            data_set.attrs['Checksum'] = v_grid.checksum


# import h5py
# from uquake.grid.nlloc import VelocityGrid3D, VelocityGridEnsemble, Phases
# from obspy.core.util.attribdict import AttribDict


def read_velocity_grid_from_hdf5(file_name: str):
    """
    Reads an HDF5 file and converts it into a VelocityGrid3D or VelocityGridEnsemble
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

                # Extract attributes
                attributes = {
                    'grid_id': phase_group.attrs['Grid ID'],
                    'schema_version': phase_group.attrs['Schema Version'],
                    'creation_timestamp': phase_group.attrs['Creation Timestamp'],
                    'type': phase_group.attrs['Type'],
                    'units': phase_group.attrs['Units'],
                    'coordinate_system': phase_group.attrs['Coordinate System'],
                    'data_order': phase_group.attrs['Data Order'],
                    'origin': phase_group.attrs['Origin'],
                    'spacing': phase_group.attrs['Spacing'],
                    'dimensions': phase_group.attrs['Dimensions'],
                    'compression': phase_group.attrs['Compression']
                }

                # Read the data set
                data = phase_group['Data'][:]

                # Create the VelocityGrid3D object for the phase
                velocity_grid = VelocityGrid3D(network_code=attributes['grid_id'],
                                               data_or_dims=data,
                                               origin=attributes['origin'],
                                               spacing=attributes['spacing'],
                                               phase=phase)  # Set the phase here
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


def write_travel_time_ensemble_to_hdf5(travel_times: TravelTimeEnsemble, filename):
    with h5py.File(filename, 'w') as f:
        for label, tt_grid in travel_times.travel_time_grids.items():
            instrument_group = f.create_group(label)
            instrument_group.attrs['Network'] = tt_grid.grid_id[tt_grid.network_code]
            instrument_group.attrs['Grid ID'] = str(tt_grid.grid_id)
            instrument_group.attrs['Velocity Model ID'] = str(tt_grid.velocity_model_id)
            instrument_group.attrs['Schema Version'] = '1.0'
            instrument_group.attrs[
                'Modification Timestamp'] = datetime.utcnow().isoformat()
            instrument_group.attrs['Type'] = 'TIME'
            instrument_group.attrs['Units'] = 'SECOND'
            instrument_group.attrs[
                'Coordinate System'] = 'Geocentric'  # Assuming geocentric for example
            instrument_group.attrs['Data Order'] = 'Row-major'
            instrument_group.attrs['Origin'] = tt_grid.origin
            instrument_group.attrs['Spacing'] = tt_grid.spacing
            instrument_group.attrs['Dimensions'] = tt_grid.data.shape
            instrument_group.attrs[
                'Compression'] = 'None'  # Or specify if any compression is used

            # Writing the data

            data = tt_grid.data.astype(tt_grid.float_type.value)
            data_set = instrument_group.create_dataset(" Data", data=data,
                                                       dtype=tt_grid.float_type.value)


def read_travel_time_ensemble_from_hdf5(filename):
    travel_time_grids = []
    with h5py.File(filename, 'r') as f:
        for instrument_id in f:
            instrument_group = f[instrument_id]
            data = instrument_group['Data'][:]
            attributes = {
                'grid_id': instrument_group.attrs['Grid ID'],
                'velocity_model_id': instrument_group.attrs['Velocity Model ID'],
                'schema_version': instrument_group.attrs['Schema Version'],
                'modification_timestamp': instrument_group.attrs['Modification Timestamp'],
                'type': instrument_group.attrs['Type'],
                'units': instrument_group.attrs['Units'],
                'coordinate_system': instrument_group.attrs['Coordinate System'],
                'data_order': instrument_group.attrs['Data Order'],
                'origin': instrument_group.attrs['Origin'],
                'spacing': instrument_group.attrs['Spacing'],
                'dimensions': instrument_group.attrs['Dimensions'],
                'compression': instrument_group.attrs['Compression']
            }



            tt_grid = TTGrid(
                network=instrument_id,
                data_or_dims=data.shape,
                origin=instrument_group.attrs['Origin'],
                spacing=instrument_group.attrs['Spacing'],
                seed=AttribDict({'label': instrument_id}),
                # Update with actual seed creation
                velocity_model_id=AttribDict({'id': "velocity_model_id_placeholder"}),
                # Adjust as necessary
                phase=Phases.P,  # Update with actual phase if needed
                float_type=FloatTypes(instrument_group['Data'].dtype)
                # Update this accordingly
            )
            travel_time_grids.append(tt_grid)

    return TravelTimeEnsemble(travel_time_grids)


# Usage example
tte = TravelTimeEnsemble(
    [tt_grid1, tt_grid2])  # Assuming tt_grid1 and tt_grid2 are TTGrid objects
write_travel_time_ensemble_to_hdf5(tte, 'travel_time_ensemble.hdf5')
tte_read = read_travel_time_ensemble_from_hdf5('travel_time_ensemble.hdf5')

# Example of using the function
# my_velocity_object = read_velocity_grid_from_hdf5('velocity_grid.hdf5')


