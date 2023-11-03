import unittest
from uquake.grid import extended
import random
import numpy as np
from uquake.grid.base import Grid
from uquake.grid.extended import VelocityGrid3D, VelocityGridEnsemble, Phases
from uquake.io.grid.core import read_velocity_grid_from_hdf5, write_velocity_grid_to_hdf5
from tempfile import NamedTemporaryFile


class TestGridOperations(unittest.TestCase):

    def setUp(self):
        """
        Set up data for the tests.
        """
        self.grid_dims = [10, 10, 10]
        self.grid_space = [1, 1, 1]
        self.grid_origin = [0, 0, 0]
        self.grid_dimensions = self.grid_dims
        self.grid_spacing = self.grid_space
        self.n_seeds = 10
        self.grid = Grid(self.grid_dims, self.grid_space, self.grid_origin, value=1)
        self.p_velocity_data = np.random.rand(
            *self.grid_dimensions) * 1000 + 5000  # Random P-wave velocity values
        self.s_velocity_data = np.random.rand(
            *self.grid_dimensions) * 1000 + 5000  # Random S-wave velocity values
        self.network_code = "TestNetwork"
        self.phase_p = Phases.P
        self.phase_s = Phases.S

    def test_seed_ensemble_in_grid(self):
        """
        Test seed ensemble generation within grid.
        """
        seeds = extended.SeedEnsemble.generate_random_seeds_in_grid(
            self.grid, n_seeds=self.n_seeds)

        self.assertEqual(len(seeds), self.n_seeds)

    def test_layered_velocity_model(self):
        """
        Test layered velocity model creation and 3D grid conversion.
        """
        lvm = extended.LayeredVelocityModel('NT')
        boundaries = [0, 100, 200, 300, 400, 500]
        for boundary in boundaries:
            layer = extended.ModelLayer(boundary, random.random() * 10000)
            lvm.add_layer(layer)

        velocity_3d = lvm.to_3d_grid(
            self.grid_dims, self.grid_origin, self.grid_space)
        self.assertIsInstance(velocity_3d, extended.VelocityGrid3D)

    def test_velocity_grid_smoothing_and_timing(self):
        """
        Test velocity grid smoothing and time grid conversion.
        """
        velocity_3d = extended.VelocityGrid3D(
            'NT', np.random.rand(100, 100, 100) * 1000 + 5000,
            self.grid_origin, self.grid_space)

        velocity_3d.smooth(20)
        self.assertTrue(
            np.allclose(np.mean(velocity_3d.data), 5000, atol=1000))

        seeds = extended.SeedEnsemble.generate_random_seeds_in_grid(
            velocity_3d, n_seeds=self.n_seeds)
        print(len(seeds))
        tt_grid = velocity_3d.to_time(seeds[0])
        tt_grids = velocity_3d.to_time_multi_threaded(seeds)

        self.assertIsInstance(tt_grid, extended.TTGrid)
        self.assertEqual(len(tt_grids), self.n_seeds)

    def create_test_velocity_grid3d(self, phase):
        """
        Creates a VelocityGrid3D object for testing with the given phase.
        """
        if phase == self.phase_p:
            grid_data = self.p_velocity_data
        else:
            grid_data = self.s_velocity_data

        return VelocityGrid3D(network_code=self.network_code,
                              data_or_dims=grid_data,
                              origin=self.grid_origin,
                              spacing=self.grid_spacing,
                              phase=phase)

    def create_test_velocity_grid3d(self, phase):
        """
        Creates a VelocityGrid3D object for testing with the given phase.
        """
        if phase == self.phase_p:
            grid_data = self.p_velocity_data
        else:
            grid_data = self.s_velocity_data

        return VelocityGrid3D(network_code=self.network_code,
                              data_or_dims=grid_data,
                              origin=self.grid_origin,
                              spacing=self.grid_spacing,
                              phase=phase)

    def test_write_and_read_velocity_grid(self):
        """
        Test writing to and reading from an HDF5 file for velocity grid data.
        """
        p_velocity_grid = self.create_test_velocity_grid3d(self.phase_p)
        s_velocity_grid = self.create_test_velocity_grid3d(self.phase_s)

        velocity_ensemble = VelocityGridEnsemble(p_velocity_grid, s_velocity_grid)

        with NamedTemporaryFile() as temp_file:
            # Write to a temporary file
            write_velocity_grid_to_hdf5(velocity_ensemble, temp_file.name)

            # Read the velocity grid back from the temporary file
            read_velocity_ensemble = read_velocity_grid_from_hdf5(temp_file.name)

            # Verify the P-wave grid
            np.testing.assert_array_almost_equal(
                read_velocity_ensemble['P'].data, p_velocity_grid.data, decimal=3,
                err_msg="Mismatch between written and read P-wave velocity data!")

            # Verify the S-wave grid
            np.testing.assert_array_almost_equal(
                read_velocity_ensemble['S'].data, s_velocity_grid.data, decimal=3,
                err_msg="Mismatch between written and read S-wave velocity data!")

            # # Verify checksum for P-wave grid
            # checksum = p_velocity_grid.checksum
            # self.assertEqual(read_velocity_ensemble['P'].checksum, checksum,
            #                  "Checksum mismatch for P-wave grid data.")
            #
            # # Verify checksum for S-wave grid
            # checksum = s_velocity_grid.checksum
            # self.assertEqual(read_velocity_ensemble['S'].checksum, checksum,
            #                  "Checksum mismatch for S-wave grid data.")


if __name__ == '__main__':
    unittest.main()
