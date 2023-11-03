from uquake.grid.extended import VelocityGrid3D, VelocityGridEnsemble, Phases
import numpy as np
from uquake.io.grid.core import write_velocity_grid_to_hdf5, read_velocity_grid_from_hdf5
from obspy.core.util import AttribDict

# Assuming the functions write_velocity_grid_to_hdf5 and read_velocity_grid_from_hdf5
# are already defined as discussed earlier


def create_test_velocity_grid3d(phase):
    """
    Creates a VelocityGrid3D object for testing.
    """
    grid_dimensions = (10, 10, 10)
    grid_spacing = (1.0, 1.0, 1.0)
    grid_origin = (0.0, 0.0, 0.0)
    grid_data = np.random.rand(*grid_dimensions) * 1000 + 5000  # Random velocity values
    network_code = "TestNetwork"

    return VelocityGrid3D(network_code=network_code,
                          data_or_dims=grid_data,
                          origin=grid_origin,
                          spacing=grid_spacing,
                          phase=phase)


def main():
    hdf5_filename = 'test_velocity_grid.hdf5'

    # Create test VelocityGrid3D for P and S waves
    p_velocity_grid = create_test_velocity_grid3d(Phases.P)
    s_velocity_grid = create_test_velocity_grid3d(Phases.S)

    velocity_grids = VelocityGridEnsemble(p_velocity_grid, s_velocity_grid)

    # Write the P-wave velocity grid to HDF5
    write_velocity_grid_to_hdf5(velocity_grids, hdf5_filename)
    print(f"Wrote P-wave velocity grid to {hdf5_filename}.")

    # Read the velocity grid back from HDF5
    read_velocity_grid = read_velocity_grid_from_hdf5(hdf5_filename)
    print(f"Read both P- and S-wave velocity grid from {hdf5_filename}.")
    assert np.allclose(read_velocity_grid['P'].data, p_velocity_grid.data), \
        "Mismatch between written and read P-wave velocity data!"
    print("P-wave velocity grid read successfully and verified.")
    assert np.allclose(read_velocity_grid['S'].data, s_velocity_grid.data), \
        "Mismatch between written and read S-wave velocity data!"
    print("S-wave velocity grid read successfully and verified.")

    write_velocity_grid_to_hdf5(p_velocity_grid, f'p_{hdf5_filename}')
    print(f"Wrote P-wave velocity grid to p_{hdf5_filename}")

    read_p_velocity_grid = read_velocity_grid_from_hdf5(f'p_{hdf5_filename}')

    assert np.allclose(read_p_velocity_grid.data, p_velocity_grid.data)
    print("P-wave velocity grid read successfully and verified.")

    write_velocity_grid_to_hdf5(s_velocity_grid, f's_{hdf5_filename}')
    print(f"Wrote P-wave velocity grid to p_{hdf5_filename}")

    read_s_velocity_grid = read_velocity_grid_from_hdf5(f's_{hdf5_filename}')
    print("S-wave velocity grid read successfully and verified.")

    assert np.allclose(read_s_velocity_grid.data, s_velocity_grid.data)

    # Write the S-wave velocity grid to the same HDF5
    # # write_velocity_grid_to_hdf5(s_velocity_grid, hdf5_filename)
    # print(f"Wrote S-wave velocity grid to {hdf5_filename}.")

    # import ipdb
    # ipdb.set_trace()

    # Verify that the read data matches the original data

    print("All tests passed!")


if __name__ == "__main__":
    main()
