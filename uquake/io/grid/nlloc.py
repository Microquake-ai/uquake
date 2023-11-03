from pathlib import Path
import numpy as np
from uquake.core.logging import logger


def _write_grid_data(grid, base_name, path='.'):

    Path(path).mkdir(parents=True, exist_ok=True)

    with open(Path(path) / (base_name + '.buf'), 'wb') \
            as out_file:
        if grid.float_type == 'FLOAT':
            out_file.write(grid.data.astype(np.float32).tobytes())

        elif grid.float_type == 'DOUBLE':
            out_file.write(grid.data.astype(np.float64).tobytes())


def _write_grid_header(grid, base_name, path='.', seed_label=None,
                       seed=None, seed_units=None):

    # convert 'METER' to 'KILOMETER'
    if grid.grid_units == 'METER':
        origin = grid.origin / 1000
        spacing = grid.spacing / 1000
    else:
        origin = grid.origin
        spacing = grid.spacing

    line1 = f'{grid.shape[0]:d} {grid.shape[1]:d} {grid.shape[2]:d}  ' \
            f'{origin[0]:f} {origin[1]:f} {origin[2]:f}  ' \
            f'{spacing[0]:f} {spacing[1]:f} {spacing[2]:f}  ' \
            f'{grid.grid_type}\n'

    with open(Path(path) / (base_name + '.hdr'), 'w') as out_file:
        out_file.write(line1)

        if grid.grid_type in ['TIME', 'ANGLE']:

            if seed_units is None:
                logger.warning(f'seed_units are not defined. '
                               f'Assuming same units as grid ('
                               f'{grid.grid_units}')
            if grid.grid_units == 'METER':
                seed = seed / 1000

            line2 = u"%s %f %f %f\n" % (seed_label,
                                        seed[0], seed[1], seed[2])
            out_file.write(line2)

        out_file.write(u'TRANSFORM  NONE\n')


def _write_grid_model_id(base_name, model_id, path='.'):
    with open(Path(path) / (base_name + '.mid'), 'w') as out_file:
        out_file.write(f'{model_id}')


def write_nlloc(base_name, path='.'):

    _write_grid_data(base_name, path=path)
    _write_grid_header(base_name, path=path)
    _write_grid_model_id(base_name, path=path)
