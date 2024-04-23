from uquake.grid.extended import (VelocityGrid3D, DensityGrid3D, Phases,
                                  SeismicPropertyGridEnsemble)

label = 'test'

p_velocity = VelocityGrid3D('NT', [10, 10, 10],
                            [100, 100, 100], [10, 10, 10], Phases.P,
                            label=label, value=5000)

s_velocity = VelocityGrid3D('NT', [10, 10, 10],
                            [100, 100, 100], [10, 10, 10], Phases.S,
                            label=label, value=5000)

p_velocity.fill_checkerboard(anomaly_size=200, base_velocity=5000,
                             velocity_perturbation=0.2, n_sigma=2)

s_velocity.fill_checkerboard(anomaly_size=200, base_velocity=5000,
                             velocity_perturbation=0.2, n_sigma=2)


density = DensityGrid3D('NT', [10, 10, 10], [10, 10, 10],
                        [10, 10, 10], label='test', value=2700)

spge = SeismicPropertyGridEnsemble(p_velocity, s_velocity, density)
spge.to_phase_velocities(period_min=0.1, period_max=10, n_periods=10,
                          logspace=True)

# z_vel = p_velocity.interpolate(coords, grid_space=False, mode='nearest', order=1)
