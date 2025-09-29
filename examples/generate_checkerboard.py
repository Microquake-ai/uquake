import time
import numpy as np


from uquake.grid.extended import (VelocityGrid3D, Phases,DensityGrid3D, PhaseVelocity,
                                  SeismicPropertyGridEnsemble, GridUnits, GridTypes,
                                  PhaseVelocityEnsemble, DisbaParam,
                                  SurfaceWaveVelocity,VelocityType)

dst_unit = "m"
label = 'test'
disba_param = DisbaParam(dc=0.001, dp=0.001)
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

s_velocity.write(filename="/Users/mahernasr/Out_uquake/vs", format="VTK")
#
spge = SeismicPropertyGridEnsemble(p_velocity, s_velocity, density)
#
periods = np.logspace(np.log10(0.1), np.log10(10), 1)


def nasr_mercier_space(start, stop, npoints):
    return (np.logspace(0, np.log10(10 + 1), npoints) - 10 ** 0 + start) * stop / 10

z = [0, 10., 20, 40., 80., 150., 200., 250., 300., 350., 450, 550., 650., 750, 850, 1000., 1150.,
     1300., 1450.,]
z = np.array(z) * alpha + 0. * alpha


z = (np.logspace(0, np.log10(10 + 1), 30) - 1) * 1450 / 10



surface = SurfaceWaveVelocity.from_seismic_property_grid_ensemble(
    seismic_param=spge,
    period=0.1,
    z_axis_log=False,
    disba_param=DisbaParam(),
    velocity_type=VelocityType.GROUP,
    phase=Phases.RAYLEIGH
)
#     PhaseVelocityEnsemble
# start_time = time.time()
# Phe = PhaseVelocityEnsemble.from_seismic_property_grid_ensemble(
#     spge, periods=list(periods), phase=Phases.RAYLEIGH, disba_param=disba_param)
# end_time = time.time()
# runtime = end_time - start_time
# print("Runtime:", runtime, "seconds")
# Phe.plot_dispersion_curve(9.8, 0.50)



#
# phase_velocity = PhaseVelocity.from_seismic_property_grid_ensemble(
#     seismic_param=spge, period=0.2, phase=Phases("RAYLEIGH"), disba_param=disba_param)
# phase_velocity.write(filename="/Users/mahernasr/Out_uquake/Rayleigh", format="VTK")
# phase_velocity.plot()