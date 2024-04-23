from uquake.grid.extended import (VelocityGrid3D, DensityGrid3D, Phases,
                                  SeismicPropertyGridEnsemble)

label = 'test'

p_velocity = VelocityGrid3D('NT', [80, 80, 30],
                            [100, 100, 100], [50, 50, 50], Phases.P,
                            label=label, value=5000, grid_units='METER')

s_velocity = VelocityGrid3D('NT', [80, 80, 30],
                            [100, 100, 100], [50, 50, 50], Phases.S,
                            label=label, value=1500)

p_velocity.fill_checkerboard(anomaly_size=200, base_velocity=5000,
                             velocity_perturbation=0.2, n_sigma=2)

s_velocity.fill_checkerboard(anomaly_size=200, base_velocity=1500,
                             velocity_perturbation=0.2, n_sigma=2)


density = DensityGrid3D('NT', [80, 80, 30], [100, 100, 100],
                        [50, 50, 50], label='test', value=2700)

spge = SeismicPropertyGridEnsemble(p_velocity, s_velocity, density)
# spge.to_phase_velocities(period_min=0.1, period_max=10, n_periods=10,
#                           logspace=True)
