import unittest
import numpy as np

from uquake.grid.extended import (
    SurfaceWaveVelocity, GridUnits, GridTypes,
    Phases, CoordinateSystem, DisbaParam, FloatTypes, VelocityType, VelocityGrid3D,
    DensityGrid3D, SeismicPropertyGridEnsemble, PhaseVelocity, GroupVelocity,
    SurfaceVelocityEnsemble, PhaseVelocityEnsemble, GroupVelocityEnsemble
)


class TestSurfaceVelocityEnsembleFromSeismicPropertyGridEnsemble(unittest.TestCase):
    def setUp(self):
        # Small, fast model (meters)
        self.nx, self.ny, self.nz = 16, 12, 8
        self.origin_m = [100.0, 200.0, 0.0]
        self.spacing_m = [50.0, 50.0, 25.0]
        self.periods = [0.5, 1.0, 2.0]
        self.phase = Phases.RAYLEIGH
        self.label = "test-ensemble"
        self.network = "NT"
        self.coord_sys = CoordinateSystem.NED
        self.float_type = FloatTypes.FLOAT

        self.unit_m = GridUnits("METER")
        self.grtype_m = GridTypes("VELOCITY_METERS")

        p_base, s_base, rho_base = 5000.0, 1500.0, 2.7
        self.p_velocity_m = VelocityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            self.origin_m, self.spacing_m, Phases.P,
            label=self.label, value=p_base,
            grid_units=self.unit_m, grid_type=self.grtype_m,
            coordinate_system=self.coord_sys, float_type=self.float_type,
        )
        self.s_velocity_m = VelocityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            self.origin_m, self.spacing_m, Phases.S,
            label=self.label, value=s_base,
            grid_units=self.unit_m, grid_type=self.grtype_m,
            coordinate_system=self.coord_sys, float_type=self.float_type,
        )
        self.density_m = DensityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            self.origin_m, self.spacing_m,
            label=self.label, value=rho_base,
            grid_units=self.unit_m, coordinate_system=self.coord_sys,
            float_type=self.float_type,
        )
        self.spge_m = SeismicPropertyGridEnsemble(self.p_velocity_m, self.s_velocity_m, self.density_m)

        # pick public or underscored factory for ensemble
        self._from_ensemble = getattr(
            SurfaceVelocityEnsemble,
            "from_seismic_property_grid_ensemble",
            getattr(SurfaceVelocityEnsemble, "_from_seismic_property_grid_ensemble"),
        )

    def test_build_group_velocity_ensemble_meters_matches_direct_call_and_scaling(self):
        """Meters grid => each SurfaceWaveVelocity equals direct result × 1e3; periods preserved."""
        disba_param = DisbaParam(dc=0.001, dp=0.001)

        ens = self._from_ensemble(
            seismic_properties=self.spge_m,
            periods=self.periods,
            phase=self.phase,
            z_axis_log=False,
            npts_log_scale=30,
            disba_param=disba_param,
            type_velocity=VelocityType.GROUP,
        )

        # Basic structure
        self.assertIsInstance(ens, SurfaceVelocityEnsemble)
        self.assertEqual(len(ens), len(self.periods))
        self.assertEqual(ens.periods, self.periods)  # preserve order

        # Element-wise checks
        for p, sv in zip(self.periods, ens):
            self.assertIsInstance(sv, SurfaceWaveVelocity)
            self.assertEqual(sv.period, p)
            self.assertEqual(sv.phase, self.phase)
            self.assertEqual(sv.grid_units, self.unit_m)
            self.assertEqual(sv.grid_type, self.grtype_m)
            self.assertEqual(tuple(sv.origin), (self.origin_m[0], self.origin_m[1]))
            self.assertEqual(tuple(sv.spacing), (self.spacing_m[0], self.spacing_m[1]))
            self.assertEqual(sv.data.shape, (self.nx, self.ny))

            # Direct calculation for this period (group velocity)
            pv = self.spge_m.to_surface_velocities(
                period_min=p, period_max=p, n_periods=1, logspace=False,
                phase=self.phase, z=None, disba_param=disba_param,
                velocity_type=VelocityType.GROUP,
            )[0]
            # Scaling rule: meters => ×1e3
            self.assertTrue(np.allclose(sv.data, pv.data * 1e3))

    def test_build_phase_velocity_ensemble_kilometers_no_scaling(self):
        """Kilometers grid => no ×1e3 scaling; values match direct results."""
        # Convert geometry to kilometers
        origin_km = [v * 1e-3 for v in self.origin_m]
        spacing_km = [v * 1e-3 for v in self.spacing_m]
        unit_km = GridUnits("KILOMETER")
        grtype_km = GridTypes("VELOCITY_KILOMETERS")

        p_vel_km = VelocityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            origin_km, spacing_km, Phases.P,
            label=self.label, value=5000.0,
            grid_units=unit_km, grid_type=grtype_km,
            coordinate_system=self.coord_sys, float_type=self.float_type,
        )
        s_vel_km = VelocityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            origin_km, spacing_km, Phases.S,
            label=self.label, value=1500.0,
            grid_units=unit_km, grid_type=grtype_km,
            coordinate_system=self.coord_sys, float_type=self.float_type,
        )
        rho_km = DensityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            origin_km, spacing_km,
            label=self.label, value=2.7,
            grid_units=unit_km, coordinate_system=self.coord_sys,
            float_type=self.float_type,
        )
        spge_km = SeismicPropertyGridEnsemble(p_vel_km, s_vel_km, rho_km)

        disba_param = DisbaParam(dc=0.001, dp=0.001)

        ens = self._from_ensemble(
            seismic_properties=spge_km,
            periods=self.periods,
            phase=self.phase,
            z_axis_log=False,
            npts_log_scale=30,
            disba_param=disba_param,
            type_velocity=VelocityType.PHASE,  # test PHASE path too
        )

        self.assertEqual(len(ens), len(self.periods))
        self.assertEqual(ens.periods, self.periods)

        for p, sv in zip(self.periods, ens):
            # Direct phase velocity in km should match (no ×1e3)
            pv = spge_km.to_surface_velocities(
                period_min=p, period_max=p, n_periods=1, logspace=False,
                phase=self.phase, z=None, disba_param=disba_param,
                velocity_type=VelocityType.PHASE,
            )[0]
            self.assertTrue(np.allclose(sv.data, pv.data))
            self.assertEqual(tuple(sv.origin), (origin_km[0], origin_km[1]))
            self.assertEqual(tuple(sv.spacing), (spacing_km[0], spacing_km[1]))
            self.assertEqual(sv.grid_units, unit_km)
            self.assertEqual(sv.grid_type, grtype_km)

    def test_log_depth_sampling_forwards_z_array(self):
        """z_axis_log=True => ensure a z array is forwarded (via probe wrapper)."""
        class SPGEProbe(SeismicPropertyGridEnsemble):
            def __init__(self, inner):
                self.__dict__.update(inner.__dict__)
                self._inner = inner
                self.captured = []

            def to_surface_velocities(self, **kwargs):
                # capture each call
                self.captured.append(kwargs.copy())
                return self._inner.to_surface_velocities(**kwargs)

        probe = SPGEProbe(self.spge_m)
        npts = 25
        disba_param = DisbaParam()

        ens = self._from_ensemble(
            seismic_properties=probe,
            periods=self.periods,
            phase=self.phase,
            z_axis_log=True,
            npts_log_scale=npts,
            disba_param=disba_param,
            type_velocity=VelocityType.GROUP,
        )

        self.assertEqual(len(ens), len(self.periods))
        # We made one call per period; each must have a non-None z of length npts
        self.assertEqual(len(probe.captured), len(self.periods))
        for call in probe.captured:
            self.assertIn("z", call)
            self.assertIsNotNone(call["z"])
            self.assertEqual(len(call["z"]), npts)


class TestPhaseVelocityEnsembleFromSeismicPropertyGridEnsemble(unittest.TestCase):
    def setUp(self):
        # Small, fast 3D model (meters)
        self.nx, self.ny, self.nz = 12, 10, 6
        self.origin_m = [100.0, 200.0, 0.0]
        self.spacing_m = [50.0, 50.0, 25.0]
        self.periods = [0.5, 1.0, 2.0]
        self.phase = Phases.RAYLEIGH
        self.label = "pve-ensemble"
        self.network = "NT"
        self.coord_sys = CoordinateSystem.NED
        self.float_type = FloatTypes.FLOAT

        self.unit_m = GridUnits("METER")
        self.grtype_m = GridTypes("VELOCITY_METERS")

        p_base, s_base, rho_base = 5000.0, 1500.0, 2.7
        self.p_velocity_m = VelocityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            self.origin_m, self.spacing_m, Phases.P,
            label=self.label, value=p_base,
            grid_units=self.unit_m, grid_type=self.grtype_m,
            coordinate_system=self.coord_sys, float_type=self.float_type,
        )
        self.s_velocity_m = VelocityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            self.origin_m, self.spacing_m, Phases.S,
            label=self.label, value=s_base,
            grid_units=self.unit_m, grid_type=self.grtype_m,
            coordinate_system=self.coord_sys, float_type=self.float_type,
        )
        self.density_m = DensityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            self.origin_m, self.spacing_m,
            label=self.label, value=rho_base,
            grid_units=self.unit_m, coordinate_system=self.coord_sys,
            float_type=self.float_type,
        )
        self.spge_m = SeismicPropertyGridEnsemble(self.p_velocity_m, self.s_velocity_m, self.density_m)

    def test_phasevelocityensemble_meters_matches_direct_and_scaling(self):
        """Meters grid: ensemble elements equal direct phase velocity × 1e3; metadata & order preserved."""
        disba_param = DisbaParam(dc=0.001, dp=0.001)

        ens = PhaseVelocityEnsemble.from_seismic_property_grid_ensemble(
            seismic_properties=self.spge_m,
            periods=self.periods,
            phase=self.phase,
            z_axis_log=False,
            npts_log_scale=30,
            disba_param=disba_param,
        )

        # Type & ensemble-level velocity_type
        self.assertIsInstance(ens, PhaseVelocityEnsemble)
        self.assertEqual(ens.velocity_type, VelocityType.PHASE)

        # Length and periods property
        self.assertEqual(len(ens), len(self.periods))
        self.assertEqual(ens.periods, self.periods)

        # Each element should be a SurfaceWaveVelocity with velocity_type=PHASE and correct metadata
        for p, sv in zip(self.periods, ens):
            self.assertIsInstance(sv, SurfaceWaveVelocity)
            self.assertEqual(sv.velocity_type, VelocityType.PHASE)
            self.assertEqual(sv.period, p)
            self.assertEqual(sv.phase, self.phase)
            self.assertEqual(sv.grid_units, self.unit_m)
            self.assertEqual(sv.grid_type, self.grtype_m)
            self.assertEqual(tuple(sv.origin), (self.origin_m[0], self.origin_m[1]))
            self.assertEqual(tuple(sv.spacing), (self.spacing_m[0], self.spacing_m[1]))
            self.assertEqual(sv.data.shape, (self.nx, self.ny))

            # Direct computation for this period (PHASE)
            direct = self.spge_m.to_surface_velocities(
                period_min=p, period_max=p, n_periods=1, logspace=False,
                phase=self.phase, z=None, disba_param=disba_param,
                velocity_type=VelocityType.PHASE,
            )[0].data

            # Scaling rule: meters => ×1e3
            self.assertTrue(np.allclose(sv.data, direct * 1e3))


    def test_phasevelocityensemble_kilometers_no_scaling(self):
        """Kilometer grids: no ×1e3 scaling; values match direct results;
         metadata is km."""
        # Convert spatial geometry to kilometers
        origin_km = [v * 1e-3 for v in self.origin_m]
        spacing_km = [v * 1e-3 for v in self.spacing_m]
        unit_km = GridUnits("KILOMETER")
        grtype_km = GridTypes("VELOCITY_KILOMETERS")

        p_vel_km = VelocityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            origin_km, spacing_km, Phases.P,
            label=self.label, value=5000.0,
            grid_units=unit_km, grid_type=grtype_km,
            coordinate_system=self.coord_sys, float_type=self.float_type,
        )
        s_vel_km = VelocityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            origin_km, spacing_km, Phases.S,
            label=self.label, value=1500.0,
            grid_units=unit_km, grid_type=grtype_km,
            coordinate_system=self.coord_sys, float_type=self.float_type,
        )
        rho_km = DensityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            origin_km, spacing_km,
            label=self.label, value=2.7,
            grid_units=unit_km, coordinate_system=self.coord_sys,
            float_type=self.float_type,
        )
        spge_km = SeismicPropertyGridEnsemble(p_vel_km, s_vel_km, rho_km)

        disba_param = DisbaParam(dc=0.001, dp=0.001)

        ens = PhaseVelocityEnsemble.from_seismic_property_grid_ensemble(
            seismic_properties=spge_km,
            periods=self.periods,
            phase=self.phase,
            z_axis_log=False,
            npts_log_scale=30,
            disba_param=disba_param,
        )

        self.assertEqual(len(ens), len(self.periods))
        self.assertEqual(ens.velocity_type, VelocityType.PHASE)
        self.assertEqual(ens.periods, self.periods)

        for p, sv in zip(self.periods, ens):
            direct = spge_km.to_surface_velocities(
                period_min=p, period_max=p, n_periods=1, logspace=False,
                phase=self.phase, z=None, disba_param=disba_param,
                velocity_type=VelocityType.PHASE,
            )[0].data

            # No scaling for kilometers
            self.assertTrue(np.allclose(sv.data, direct))
            self.assertEqual(tuple(sv.origin), (origin_km[0], origin_km[1]))
            self.assertEqual(tuple(sv.spacing), (spacing_km[0], spacing_km[1]))
            self.assertEqual(sv.grid_units, unit_km)
            self.assertEqual(sv.grid_type, grtype_km)

    def test_phasevelocityensemble_log_depth_sampling_forwards_z_each_period(self):
        """z_axis_log=True → every call forwards a non-None z array of length npts."""
        class SPGEProbe(SeismicPropertyGridEnsemble):
            def __init__(self, inner):
                self.__dict__.update(inner.__dict__)
                self._inner = inner
                self.captured = []

            def to_surface_velocities(self, **kwargs):
                self.captured.append(kwargs.copy())
                return self._inner.to_surface_velocities(**kwargs)

        probe = SPGEProbe(self.spge_m)
        npts = 24

        ens = PhaseVelocityEnsemble.from_seismic_property_grid_ensemble(
            seismic_properties=probe,
            periods=self.periods,
            phase=self.phase,
            z_axis_log=True,
            npts_log_scale=npts,
            disba_param=DisbaParam(),
        )

        self.assertEqual(len(ens), len(self.periods))
        self.assertEqual(len(probe.captured), len(self.periods))
        for call in probe.captured:
            self.assertIn("z", call)
            self.assertIsNotNone(call["z"])
            self.assertEqual(len(call["z"]), npts)


class TestGroupVelocityEnsembleFromSeismicPropertyGridEnsemble(unittest.TestCase):
    def setUp(self):
        # Small, fast 3D model (meters)
        self.nx, self.ny, self.nz = 12, 10, 6
        self.origin_m = [100.0, 200.0, 0.0]
        self.spacing_m = [50.0, 50.0, 25.0]
        self.periods = [0.5, 1.0, 2.0]
        self.phase = Phases.RAYLEIGH
        self.label = "gve-ensemble"
        self.network = "NT"
        self.coord_sys = CoordinateSystem.NED
        self.float_type = FloatTypes.FLOAT

        self.unit_m = GridUnits("METER")
        self.grtype_m = GridTypes("VELOCITY_METERS")

        p_base, s_base, rho_base = 5000.0, 1500.0, 2.7
        self.p_velocity_m = VelocityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            self.origin_m, self.spacing_m, Phases.P,
            label=self.label, value=p_base,
            grid_units=self.unit_m, grid_type=self.grtype_m,
            coordinate_system=self.coord_sys, float_type=self.float_type,
        )
        self.s_velocity_m = VelocityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            self.origin_m, self.spacing_m, Phases.S,
            label=self.label, value=s_base,
            grid_units=self.unit_m, grid_type=self.grtype_m,
            coordinate_system=self.coord_sys, float_type=self.float_type,
        )
        self.density_m = DensityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            self.origin_m, self.spacing_m,
            label=self.label, value=rho_base,
            grid_units=self.unit_m, coordinate_system=self.coord_sys,
            float_type=self.float_type,
        )
        self.spge_m = SeismicPropertyGridEnsemble(self.p_velocity_m, self.s_velocity_m, self.density_m)

    # -------------------- METERS: should apply ×1e3 scaling --------------------

    def test_groupvelocityensemble_meters_matches_direct_and_scaling(self):
        """Meters grid: ensemble elements equal direct GROUP velocity × 1e3; metadata & order preserved."""
        disba_param = DisbaParam(dc=0.001, dp=0.001)

        ens = GroupVelocityEnsemble.from_seismic_property_grid_ensemble(
            seismic_properties=self.spge_m,
            periods=self.periods,
            phase=self.phase,
            z_axis_log=False,
            npts_log_scale=30,
            disba_param=disba_param,
        )

        # Ensemble type & velocity_type
        self.assertIsInstance(ens, GroupVelocityEnsemble)
        self.assertEqual(ens.velocity_type, VelocityType.GROUP)

        # Length and periods ordering
        self.assertEqual(len(ens), len(self.periods))
        self.assertEqual(ens.periods, self.periods)

        # Per-element checks + numerical agreement (with ×1e3)
        for p, sv in zip(self.periods, ens):
            self.assertIsInstance(sv, SurfaceWaveVelocity)
            self.assertEqual(sv.velocity_type, VelocityType.GROUP)
            self.assertEqual(sv.period, p)
            self.assertEqual(sv.phase, self.phase)
            self.assertEqual(sv.grid_units, self.unit_m)
            self.assertEqual(sv.grid_type, self.grtype_m)
            self.assertEqual(tuple(sv.origin), (self.origin_m[0], self.origin_m[1]))
            self.assertEqual(tuple(sv.spacing), (self.spacing_m[0], self.spacing_m[1]))
            self.assertEqual(sv.data.shape, (self.nx, self.ny))

            direct = self.spge_m.to_surface_velocities(
                period_min=p, period_max=p, n_periods=1, logspace=False,
                phase=self.phase, z=None, disba_param=disba_param,
                velocity_type=VelocityType.GROUP,
            )[0].data

            self.assertTrue(np.allclose(sv.data, direct * 1e3))

    # -------------------- KILOMETERS: should NOT scale --------------------

    def test_groupvelocityensemble_kilometers_no_scaling(self):
        """Kilometer grids: no ×1e3 scaling; values match direct results; metadata is km."""
        origin_km = [v * 1e-3 for v in self.origin_m]
        spacing_km = [v * 1e-3 for v in self.spacing_m]
        unit_km = GridUnits("KILOMETER")
        grtype_km = GridTypes("VELOCITY_KILOMETERS")

        p_vel_km = VelocityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            origin_km, spacing_km, Phases.P,
            label=self.label, value=5000.0,
            grid_units=unit_km, grid_type=grtype_km,
            coordinate_system=self.coord_sys, float_type=self.float_type,
        )
        s_vel_km = VelocityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            origin_km, spacing_km, Phases.S,
            label=self.label, value=1500.0,
            grid_units=unit_km, grid_type=grtype_km,
            coordinate_system=self.coord_sys, float_type=self.float_type,
        )
        rho_km = DensityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            origin_km, spacing_km,
            label=self.label, value=2.7,
            grid_units=unit_km, coordinate_system=self.coord_sys,
            float_type=self.float_type,
        )
        spge_km = SeismicPropertyGridEnsemble(p_vel_km, s_vel_km, rho_km)

        disba_param = DisbaParam(dc=0.001, dp=0.001)

        ens = GroupVelocityEnsemble.from_seismic_property_grid_ensemble(
            seismic_properties=spge_km,
            periods=self.periods,
            phase=self.phase,
            z_axis_log=False,
            npts_log_scale=30,
            disba_param=disba_param,
        )

        self.assertEqual(len(ens), len(self.periods))
        self.assertEqual(ens.velocity_type, VelocityType.GROUP)
        self.assertEqual(ens.periods, self.periods)

        for p, sv in zip(self.periods, ens):
            direct = spge_km.to_surface_velocities(
                period_min=p, period_max=p, n_periods=1, logspace=False,
                phase=self.phase, z=None, disba_param=disba_param,
                velocity_type=VelocityType.GROUP,
            )[0].data

            # No scaling for kilometers
            self.assertTrue(np.allclose(sv.data, direct))
            self.assertEqual(tuple(sv.origin), (origin_km[0], origin_km[1]))
            self.assertEqual(tuple(sv.spacing), (spacing_km[0], spacing_km[1]))
            self.assertEqual(sv.grid_units, unit_km)
            self.assertEqual(sv.grid_type, grtype_km)

    # -------------------- z_axis_log=True forwards z for EACH period --------------------

    def test_groupvelocityensemble_log_depth_sampling_forwards_z_each_period(self):
        """z_axis_log=True → every call forwards a non-None z array of length npts."""
        class SPGEProbe(SeismicPropertyGridEnsemble):
            def __init__(self, inner):
                self.__dict__.update(inner.__dict__)
                self._inner = inner
                self.captured = []

            def to_surface_velocities(self, **kwargs):
                self.captured.append(kwargs.copy())
                return self._inner.to_surface_velocities(**kwargs)

        probe = SPGEProbe(self.spge_m)
        npts = 24

        ens = GroupVelocityEnsemble.from_seismic_property_grid_ensemble(
            seismic_properties=probe,
            periods=self.periods,
            phase=self.phase,
            z_axis_log=True,
            npts_log_scale=npts,
            disba_param=DisbaParam(),
        )

        self.assertEqual(len(ens), len(self.periods))
        self.assertEqual(len(probe.captured), len(self.periods))
        for call in probe.captured:
            self.assertIn("z", call)
            self.assertIsNotNone(call["z"])
            self.assertEqual(len(call["z"]), npts)


if __name__ == "__main__":
    unittest.main()
