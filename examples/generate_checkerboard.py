from uquake.grid.extended import VelocityGrid3D, VelocityGridEnsemble, SeedEnsemble
from uquake.core import read_inventory
import numpy as np

# create a grid
p_velocity_data = np.random.rand(80, 80, 30) * 0 + 5000
s_velocity_data = np.random.rand(80, 80, 30) * 0 + 1500


grid_s = VelocityGrid3D(network_code="TestNetwork",spacing=[50, 50, 50],data_or_dims=s_velocity_data, origin=(0., 0., 0.),
                      phase="S")
grid_p = VelocityGrid3D(network_code="TestNetwork",spacing=[50, 50, 50],data_or_dims=p_velocity_data, origin=(0., 0., 0.),
                      phase="S")
grid_s.fill_checkerboard(anomaly_size=[400, 400, 400 ], base_velocity=1500., velocity_perturbation=0.15, n_sigma=3)
grid_p.fill_checkerboard(anomaly_size=[400, 400, 400 ], base_velocity=1500., velocity_perturbation=0.15, n_sigma=3)
#grid_s.to_vtk({"Velocity": grid_s.data}, "Domain")