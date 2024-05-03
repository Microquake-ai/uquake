import numpy as np

from uquake.grid.extended import (VelocityGrid3D, Phases,DensityGrid3D, PhaseVelocity,
                                  SeismicPropertyGridEnsemble, GridUnits, GridTypes,
                                  PhaseVelocityEnsemble, DisbaParam, Seed, SeedEnsemble)
from uquake.core.coordinates import Coordinates

dst_unit = "m"
label = 'test'
disba_param = DisbaParam(dc=0.001, dp=0.010)
if dst_unit == "m":
    unit = GridUnits('METER')
    grtype = GridTypes("VELOCITY_METERS")
    alpha = 1.
else:
    unit = GridUnits('KILOMETER')
    alpha = 1.e-3
    grtype = GridTypes("VELOCITY_KILOMETERS")

p_velocity = VelocityGrid3D('NT', [90, 80, 60],
                            [100. * alpha, 100. * alpha, 0. * alpha],
                            [50 * alpha, 50. * alpha, 25. * alpha], Phases.P,
                            label=label, value=5000, grid_units=unit, grid_type=grtype)
s_velocity = VelocityGrid3D('NT', [90, 80, 60],
                            [100 * alpha, 100 * alpha, 0 * alpha],
                            [50 * alpha, 50 * alpha, 25 * alpha], Phases.S,
                            label=label, value=1500, grid_units=unit, grid_type=grtype)

p_velocity.fill_checkerboard(anomaly_size=[900 * alpha, 900 * alpha, 450 * alpha],
                             base_velocity=5000 * alpha, velocity_perturbation=0.2,
                             n_sigma=3.5)


s_velocity.fill_checkerboard(anomaly_size=[900 * alpha, 900 * alpha, 450 * alpha],
                             base_velocity=1500 * alpha, velocity_perturbation=0.2,
                             n_sigma=3.5)

density = DensityGrid3D('NT', [90, 80, 60],
                        [100 * alpha, 100 * alpha, 0 * alpha],
                        [50 * alpha, 50 * alpha, 25 * alpha],
                        label='test', value=2.7, grid_units=unit)

spge = SeismicPropertyGridEnsemble(p_velocity, s_velocity, density)

phase_velocity = PhaseVelocity.from_seismic_property_grid_ensemble(
    seismic_param=spge, period=0.1, phase=Phases("RAYLEIGH"), disba_param=disba_param)
#phase_velocity.plot()
phase_velocity.write(filename="/Users/mahernasr/Out_uquake/Rayleigh", format="VTK")
rcv = phase_velocity.generate_random_points_in_grid(n_points=1, grid_space=False,
                                                      seed=1)

# genrate receivers
x = np.arange(s_velocity.origin[0] + s_velocity.spacing[0],
              s_velocity.corner[0] - s_velocity.spacing[0],
              10 * s_velocity.spacing[0])
y = np.arange(s_velocity.origin[1] + s_velocity.spacing[1],
              s_velocity.corner[1] - s_velocity.spacing[1],
              10 * s_velocity.spacing[1])
X, Y = np.meshgrid(x, y)
X = X.reshape((-1, 1))
Y = Y.reshape((-1, 1))

seeds_list = []
for n in range(len(X)):
    seed_instance = Seed(coordinates=Coordinates(X[n], Y[n], s_velocity.origin[0]),
                     station_code="TEST", location_code="No_location")
    seeds_list.append(seed_instance)
seeds = SeedEnsemble(seeds_list)

#phase_velocity.to_time(seed=seed_instance, method="SPM", ns=12)
tt_ensemble = phase_velocity.to_time_multi_threaded(seeds=seeds, method="SPM", ns =5)
#tt_grid="all", folder="/Users/mahernasr/out_uquake/model_")