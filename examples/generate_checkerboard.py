from uquake.grid.extended import VelocityGrid3D, VelocityGridEnsemble, SeedEnsemble
from uquake.core import read_inventory
import numpy as np

# create a grid
p_velocity_data = np.random.rand(80, 80, 30) * 0 + 50000
s_velocity_data = np.random.rand(80, 80, 30) * 0 + 1500
smoothing_kernel = 10  # standard deviation of the gaussian smoothing kernel in grid space units
#p_velocity = VelocityGrid3D("A1, p_velocity_data, spacing=(10, 10, 10), origin=(0, 0, 0)).smooth(10)
#s_velocity = VelocityGrid3D(s_velocity_data, spacing=(10, 10, 10), origin=(0, 0, 0)).smooth(10)
grid = VelocityGrid3D(network_code="TestNetwork",spacing=[50, 50, 50],data_or_dims=s_velocity_data, origin=(0., 0., 0.),
                      phase="P")
grid.fill_checkerboard(anomaly_size=[800, 800, 400 ], base_velocity=1500., velocity_perturbation=0.15, n_sigma=3)
grid3d = grid.to_rgrid(1)
grid3d.to_vtk({"Velocity": grid.data}, "Domain")