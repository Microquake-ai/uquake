import time
import matplotlib.pyplot as plt
from uquake.io.grid.core import write_vtk


from uquake.grid.extended import (VelocityGrid3D, Phases,
                                  SeismicPropertyGridEnsemble,  GridUnits)

label = 'test'
unit = GridUnits('METER')
p_velocity = VelocityGrid3D('NT', [80, 80, 30],
                            [100., 100., 100.], [50., 50., 50.], Phases.P,
                            label=label, value=5000, grid_units=unit)
s_velocity = VelocityGrid3D('NT', [80, 80, 30],
                            [100, 100, 100], [50, 50, 50], Phases.S,
                            label=label, value=1500, grid_units=unit)
p_velocity.write("vs.vtk", format="pickle", field_name="Vp")
# p_velocity.fill_checkerboard(anomaly_size=[850, 850, 350], base_velocity=5000,
#                              velocity_perturbation=0.2, n_sigma=4)
# write_vtk(p_velocity, "/Users/mahernasr/Out_uquake/vs", "Velp")
#
# s_velocity.fill_checkerboard(anomaly_size=[850, 850, 350], base_velocity=1500,
#                              velocity_perturbation=0.2, n_sigma=4)
#
# write_vtk(s_velocity, "/Users/mahernasr/Out_uquake/vs", "Vels")
#
# density = DensityGrid3D('NT', [80, 80, 30], [100, 100, 100],
#                         [50, 50, 50], label='test', value=2.7, grid_units=unit)
#
# spge = SeismicPropertyGridEnsemble(p_velocity, s_velocity, density)
#
# start_time = time.time()
#
# periods, phase_velocity = spge.to_phase_velocities(period_min=0.1, period_max=10,
#                                                    n_periods=10, logspace=True)
#
# end_time = time.time()
# runtime = end_time - start_time
# print("Runtime:", runtime, "seconds")
# p = 0
# plt.imshow(phase_velocity[p])
# plt.xlabel("X")
# plt.ylabel("Y")
# cb = plt.colorbar()
# cb.ax.set_title('Vel Rayleigh (km/s)',fontsize=8)
# plt.title("period {0:1.2f} s".format(periods[p]))
# plt.show()