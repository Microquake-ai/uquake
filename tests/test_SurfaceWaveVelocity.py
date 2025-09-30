# test_surface_wave_velocity_ctor_unittest.py
import unittest
import numpy as np

from uquake.grid.extended import (
    SurfaceWaveVelocity, GridUnits, GridTypes,
    Phases, CoordinateSystem, DisbaParam, FloatTypes, VelocityType, VelocityGrid3D,
    DensityGrid3D, SeismicPropertyGridEnsemble, PhaseVelocity, GroupVelocity
)


class TestSurfaceWaveVelocityCtor(unittest.TestCase):
    # ------------- helpers -------------

    def _same_shape_lenient(self, obj, expected_dims):
        has_shape = hasattr(obj, "shape")
        has_data = hasattr(obj, "data") and getattr(obj, "data") is not None
        self.assertTrue(has_shape or has_data, "Grid must expose .shape or non-None .data")

        if has_shape:
            shp = tuple(obj.shape)
        else:
            shp = tuple(obj.data.shape)

        self.assertEqual(np.prod(shp), np.prod(expected_dims), "Total size mismatch")

    def _assert_float_dtype_matches_float_type(self, data_dtype, float_type):
        ft_name = getattr(float_type, "name", str(float_type)).upper()
        if ft_name in {"FLOAT"}:
            # Accept either float32 or float64 depending on constructor path
            self.assertIn(
                np.dtype(data_dtype),
                (np.float32, np.float64),
                f"Expected float32 or float64 for {ft_name}, got {data_dtype}",
            )
        elif ft_name in {"FLOAT32"}:
            self.assertEqual(np.dtype(data_dtype), np.float32)
        elif ft_name in {"FLOAT64", "DOUBLE"}:
            self.assertEqual(np.dtype(data_dtype), np.float64)
        else:
            self.assertEqual(np.dtype(data_dtype).kind, "f")

    def _assert_common_attrs(self, grid, meta, origin, spacing):
        self.assertEqual(grid.network_code, meta["network_code"])
        self.assertEqual(grid.period, meta["period"])
        self.assertEqual(grid.phase, meta["phase"])
        self.assertEqual(grid.velocity_type, meta["velocity_type"])
        self.assertEqual(grid.grid_type, meta["grid_type"])
        self.assertEqual(grid.grid_units, meta["grid_units"])
        self.assertEqual(grid.float_type, meta["float_type"])

        for attr in ("origin", "spacing", "label", "coordinate_system"):
            self.assertTrue(hasattr(grid, attr))

        self.assertEqual(len(grid.origin), 2)
        self.assertEqual(len(grid.spacing), 2)
        self.assertTrue(np.allclose(grid.origin, origin))
        self.assertTrue(np.allclose(grid.spacing, spacing))
        self.assertEqual(grid.label, meta["label"])
        self.assertEqual(grid.coordinate_system, meta["coordinate_system"])

    # ------------- setup -------------

    def setUp(self):
        self.dims = [10, 12]        # 2D grid
        self.origin = [100.0, 200.0]
        self.spacing = [50.0, 50.0]

        self.meta = dict(
            network_code="NT",
            period=2.5,
            phase=Phases.RAYLEIGH,
            velocity_type=VelocityType.GROUP,
            grid_type=GridTypes.VELOCITY_METERS,
            grid_units=GridUnits.METER,
            coordinate_system=CoordinateSystem.NED,
            label="unit-test-grid",
            float_type=FloatTypes.FLOAT,
            value=450.0,
        )

    # ------------- tests -------------

    def test_ctor_with_dimensions_lists_parametrized(self):
        for phase in (Phases.RAYLEIGH, Phases.LOVE):
            for vtype in (VelocityType.GROUP, VelocityType.PHASE):
                with self.subTest(phase=phase, velocity_type=vtype):
                    grid = SurfaceWaveVelocity(
                        network_code=self.meta["network_code"],
                        data_or_dims=self.dims,
                        period=self.meta["period"],
                        phase=phase,
                        grid_type=self.meta["grid_type"],
                        grid_units=self.meta["grid_units"],
                        spacing=self.spacing,
                        origin=self.origin,
                        value=self.meta["value"],
                        coordinate_system=self.meta["coordinate_system"],
                        label=self.meta["label"],
                        float_type=self.meta["float_type"],
                        velocity_type=vtype,
                    )
                    meta_local = dict(self.meta, phase=phase, velocity_type=vtype)
                    self._assert_common_attrs(grid, meta_local, self.origin, self.spacing)
                    self._same_shape_lenient(grid, self.dims)
                    if hasattr(grid, "data") and grid.data is not None:
                        self.assertEqual(grid.data.size, np.prod(self.dims))
                        self._assert_float_dtype_matches_float_type(grid.data.dtype, self.meta["float_type"])

    def test_ctor_with_numpy_data(self):
        data = np.full((8, 9), fill_value=900.0, dtype=np.float32)
        grid = SurfaceWaveVelocity(
            network_code=self.meta["network_code"],
            data_or_dims=data,
            period=self.meta["period"],
            phase=self.meta["phase"],
            grid_type=self.meta["grid_type"],
            grid_units=self.meta["grid_units"],
            spacing=self.spacing,
            origin=self.origin,
            coordinate_system=self.meta["coordinate_system"],
            label=self.meta["label"],
            float_type=self.meta["float_type"],
            velocity_type=self.meta["velocity_type"],
        )
        self._assert_common_attrs(grid, self.meta, self.origin, self.spacing)
        self.assertEqual(tuple(grid.data.shape), (8, 9))
        self._assert_float_dtype_matches_float_type(grid.data.dtype, self.meta["float_type"])
        self.assertTrue(np.all(grid.data == 900.0))

    def test_ctor_minimal_required_args_with_dimensions(self):
        grid = SurfaceWaveVelocity(
            network_code="NT",
            data_or_dims=self.dims,
            period=1.0,
            phase=Phases.RAYLEIGH,
            grid_type=GridTypes.VELOCITY_METERS,
            grid_units=GridUnits.METER,
            spacing=[1.0, 1.0],
            origin=[0.0, 0.0],
            velocity_type=VelocityType.GROUP,
        )
        self.assertEqual(grid.network_code, "NT")
        self.assertEqual(grid.period, 1.0)
        self.assertEqual(grid.phase, Phases.RAYLEIGH)
        self.assertEqual(grid.velocity_type, VelocityType.GROUP)
        self.assertEqual(grid.grid_units, GridUnits.METER)
        self.assertEqual(grid.grid_type, GridTypes.VELOCITY_METERS)
        self.assertTrue(np.allclose(grid.origin, [0.0, 0.0]))
        self.assertTrue(np.allclose(grid.spacing, [1.0, 1.0]))
        self._same_shape_lenient(grid, self.dims)


class TestFromSeismicPropertyGridEnsemble(unittest.TestCase):
    def setUp(self):
        self.nx, self.ny, self.nz = 16, 12, 8
        self.origin = [100.0, 200.0, 0.0]
        self.spacing = [50.0, 50.0, 25.0]
        self.period = 0.5
        self.phase = Phases.RAYLEIGH
        self.velocity_type = VelocityType.GROUP
        self.label = "test-ensemble"
        self.network = "NT"
        self.coord_sys = CoordinateSystem.NED
        self.float_type = FloatTypes.FLOAT

        self.unit = GridUnits("METER")
        self.grtype = GridTypes("VELOCITY_METERS")

        self.p_velocity = VelocityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            self.origin, self.spacing, Phases.P,
            label=self.label, value=5000.0,
            grid_units=self.unit, grid_type=self.grtype,
            coordinate_system=self.coord_sys, float_type=self.float_type,
        )
        self.s_velocity = VelocityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            self.origin, self.spacing, Phases.S,
            label=self.label, value=1500.0,
            grid_units=self.unit, grid_type=self.grtype,
            coordinate_system=self.coord_sys, float_type=self.float_type,
        )
        self.density = DensityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            self.origin, self.spacing,
            label=self.label, value=2.7,
            grid_units=self.unit, coordinate_system=self.coord_sys,
            float_type=self.float_type,
        )
        self.spge = SeismicPropertyGridEnsemble(self.p_velocity, self.s_velocity, self.density)

        # Use whichever method is defined
        self._from = getattr(SurfaceWaveVelocity, "from_seismic_property_grid_ensemble",
                             getattr(SurfaceWaveVelocity, "_from_seismic_property_grid_ensemble"))

    def test_from_ensemble_matches_direct_call_and_scaling(self):
        disba_param = DisbaParam(dc=0.001, dp=0.001)

        pv_list = self.spge.to_surface_velocities(
            period_min=self.period,
            period_max=self.period,
            n_periods=1,
            logspace=False,
            phase=self.phase,
            z=None,
            disba_param=disba_param,
            velocity_type=self.velocity_type,
        )
        pv = pv_list[0]

        out = self._from(
            seismic_param=self.spge,
            period=self.period,
            z_axis_log=False,
            disba_param=disba_param,
            velocity_type=self.velocity_type,
            phase=self.phase,
        )

        self.assertIsInstance(out, SurfaceWaveVelocity)
        self.assertEqual(out.period, self.period)
        self.assertEqual(out.phase, self.phase)
        self.assertEqual(out.velocity_type, self.velocity_type)
        self.assertEqual(out.grid_units, self.unit)
        self.assertEqual(out.grid_type, self.grtype)
        self.assertEqual(out.label, self.label)
        self.assertEqual(out.network_code, self.network)
        self.assertEqual(out.coordinate_system, self.coord_sys)
        self.assertEqual(tuple(out.origin), (self.origin[0], self.origin[1]))
        self.assertEqual(tuple(out.spacing), (self.spacing[0], self.spacing[1]))
        self.assertEqual(out.data.shape, (self.nx, self.ny))
        self.assertTrue(np.allclose(out.data, pv.data * 1e3))

    def test_from_ensemble_with_log_z_builds_array(self):
        class SPGEProbe(SeismicPropertyGridEnsemble):
            def __init__(self, inner):
                self.__dict__.update(inner.__dict__)
                self._inner = inner
                self.captured = None

            def to_surface_velocities(self, **kwargs):
                self.captured = kwargs.copy()
                return self._inner.to_surface_velocities(**kwargs)

        probe = SPGEProbe(self.spge)
        npts = 30

        out = self._from(
            seismic_param=probe,
            period=self.period,
            z_axis_log=True,
            npts_log_scale=npts,
            disba_param=DisbaParam(),
            velocity_type=self.velocity_type,
            phase=self.phase,
        )

        self.assertIsNotNone(probe.captured)
        z = probe.captured["z"]
        self.assertIsNotNone(z)
        self.assertEqual(len(z), npts)

        self.assertEqual(out.period, self.period)
        self.assertEqual(out.phase, self.phase)
        self.assertEqual(out.velocity_type, self.velocity_type)
        self.assertEqual(out.data.shape, (self.nx, self.ny))


class TestPhaseVelocityFromSeismicPropertyGridEnsemble(unittest.TestCase):
    def setUp(self):
        # Small grid (meters)
        self.nx, self.ny, self.nz = 16, 12, 8
        self.origin_m = [100.0, 200.0, 0.0]
        self.spacing_m = [50.0, 50.0, 25.0]

        self.period = 0.5
        self.phase = Phases.RAYLEIGH
        self.label = "test-ensemble"
        self.network = "NT"
        self.coord_sys = CoordinateSystem.NED
        self.float_type = FloatTypes.FLOAT

        self.unit_m = GridUnits("METER")
        self.grtype_m = GridTypes("VELOCITY_METERS")

        # Uniform baseline properties
        p_base = 5000.0
        s_base = 1500.0
        rho_base = 2.7

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
        self.spge_m = SeismicPropertyGridEnsemble(self.p_velocity_m, self.s_velocity_m,
                                                  self.density_m)

        # Public or underscored factory (allow either API)
        self._from_phase = getattr(
            PhaseVelocity,
            "from_seismic_property_grid_ensemble",
            getattr(PhaseVelocity, "from_seismic_property_grid_ensemble"),
        )

    def test_phasevelocity_from_ensemble_matches_direct_call_and_scaling_meters(self):
        """Meters grid => classmethod output equals direct result × 1e3."""
        disba_param = DisbaParam(dc=0.001, dp=0.001)

        # Direct ensemble call (phase velocity)
        pv_list = self.spge_m.to_surface_velocities(
            period_min=self.period,
            period_max=self.period,
            n_periods=1,
            logspace=False,
            phase=self.phase,
            z=None,
            disba_param=disba_param,
            velocity_type=VelocityType.PHASE,
        )
        self.assertTrue(pv_list and hasattr(pv_list[0], "data"))
        pv_direct = pv_list[0]
        self.assertEqual(pv_direct.data.shape, (self.nx, self.ny))

        # Classmethod under test
        out = self._from_phase(
            seismic_param=self.spge_m,
            period=self.period,
            phase=self.phase,
            z_axis_log=False,
            disba_param=disba_param,
        )

        # Metadata / forwarding
        self.assertEqual(out.period, self.period)
        self.assertEqual(out.phase, self.phase)
        self.assertEqual(out.grid_units, self.unit_m)
        self.assertEqual(out.grid_type, self.grtype_m)
        self.assertEqual(out.label, self.label)
        self.assertEqual(out.network_code, self.network)
        self.assertEqual(out.coordinate_system, self.coord_sys)
        self.assertEqual(tuple(out.origin), (self.origin_m[0], self.origin_m[1]))
        self.assertEqual(tuple(out.spacing), (self.spacing_m[0], self.spacing_m[1]))
        self.assertEqual(out.data.shape, (self.nx, self.ny))

        # Scaling rule for meters
        self.assertTrue(np.allclose(out.data, pv_direct.data * 1e3))

        self.assertTrue(np.isfinite(out.data).all())
        self.assertGreater(out.data.mean(), 0.0)

    def test_phasevelocity_from_ensemble_with_log_z_forwards_array(self):
        """z_axis_log=True => forwards non-None z of requested length."""
        class SPGEProbe(SeismicPropertyGridEnsemble):
            def __init__(self, inner):
                self.__dict__.update(inner.__dict__)
                self._inner = inner
                self.captured = None

            def to_surface_velocities(self, **kwargs):
                self.captured = kwargs.copy()
                return self._inner.to_surface_velocities(**kwargs)

        probe = SPGEProbe(self.spge_m)
        npts = 25

        out = self._from_phase(
            seismic_param=probe,
            period=self.period,
            phase=self.phase,
            z_axis_log=True,
            npts_log_scale=npts,
            disba_param=DisbaParam(),
        )

        self.assertIsNotNone(probe.captured)
        self.assertIn("z", probe.captured)
        self.assertIsNotNone(probe.captured["z"])
        self.assertEqual(len(probe.captured["z"]), npts)

        self.assertEqual(out.data.shape, (self.nx, self.ny))
        self.assertEqual(tuple(out.origin), (self.origin_m[0], self.origin_m[1]))
        self.assertEqual(tuple(out.spacing), (self.spacing_m[0], self.spacing_m[1]))

    def test_phasevelocity_from_ensemble_kilometers_no_scaling(self):
        """Kilometers grid => classmethod output equals direct result (no ×1e3)."""
        # Convert spatial units to km
        origin_km = [v * 1e-3 for v in self.origin_m]
        spacing_km = [v * 1e-3 for v in self.spacing_m]
        unit_km = GridUnits("KILOMETER")
        grtype_km = GridTypes("VELOCITY_KILOMETERS")

        p_vel_km = VelocityGrid3D(
            self.network, [self.nx, self.ny, self.nz],
            origin_km, spacing_km, Phases.P,
            label=self.label, value=5000.0 * 1.0,  # numerical values OK; units are handled by API
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

        pv_list_km = spge_km.to_surface_velocities(
            period_min=self.period,
            period_max=self.period,
            n_periods=1,
            logspace=False,
            phase=self.phase,
            z=None,
            disba_param=disba_param,
            velocity_type=VelocityType.PHASE,
        )
        pv_direct_km = pv_list_km[0]

        out_km = self._from_phase(
            seismic_param=spge_km,
            period=self.period,
            phase=self.phase,
            z_axis_log=False,
            disba_param=disba_param,
        )

        # No scaling expected for kilometers
        self.assertTrue(np.allclose(out_km.data, pv_direct_km.data))
        self.assertEqual(tuple(out_km.origin), (origin_km[0], origin_km[1]))
        self.assertEqual(tuple(out_km.spacing), (spacing_km[0], spacing_km[1]))
        self.assertEqual(out_km.grid_units, unit_km)
        self.assertEqual(out_km.grid_type, grtype_km)


class TestGroupVelocityFromSeismicPropertyGridEnsemble(unittest.TestCase):
    def setUp(self):
        # Small grid (meters)
        self.nx, self.ny, self.nz = 16, 12, 8
        self.origin_m = [100.0, 200.0, 0.0]
        self.spacing_m = [50.0, 50.0, 25.0]

        self.period = 0.5
        self.phase = Phases.RAYLEIGH
        self.label = "test-ensemble"
        self.network = "NT"
        self.coord_sys = CoordinateSystem.NED
        self.float_type = FloatTypes.FLOAT

        self.unit_m = GridUnits("METER")
        self.grtype_m = GridTypes("VELOCITY_METERS")

        # Baseline properties
        p_base = 5000.0
        s_base = 1500.0
        rho_base = 2.7

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

        # Public or underscored factory (support either API)
        self._from_group = getattr(
            GroupVelocity,
            "from_seismic_property_grid_ensemble",
            getattr(GroupVelocity, "_from_seismic_property_grid_ensemble"),
        )

    def test_groupvelocity_from_ensemble_matches_direct_call_and_scaling_meters(self):
        """Meters grid => classmethod output equals direct result × 1e3."""
        disba_param = DisbaParam(dc=0.001, dp=0.001)

        # Direct ensemble call (group velocity)
        gv_list = self.spge_m.to_surface_velocities(
            period_min=self.period,
            period_max=self.period,
            n_periods=1,
            logspace=False,
            phase=self.phase,
            z=None,
            disba_param=disba_param,
            velocity_type=VelocityType.GROUP,
        )
        self.assertTrue(gv_list and hasattr(gv_list[0], "data"))
        gv_direct = gv_list[0]
        self.assertEqual(gv_direct.data.shape, (self.nx, self.ny))

        # Classmethod under test
        out = self._from_group(
            seismic_param=self.spge_m,
            period=self.period,
            phase=self.phase,
            z_axis_log=False,
            disba_param=disba_param,
        )

        # Metadata / forwarding
        self.assertEqual(out.period, self.period)
        self.assertEqual(out.phase, self.phase)
        self.assertEqual(out.grid_units, self.unit_m)
        self.assertEqual(out.grid_type, self.grtype_m)
        self.assertEqual(out.label, self.label)
        self.assertEqual(out.network_code, self.network)
        self.assertEqual(out.coordinate_system, self.coord_sys)
        self.assertEqual(tuple(out.origin), (self.origin_m[0], self.origin_m[1]))
        self.assertEqual(tuple(out.spacing), (self.spacing_m[0], self.spacing_m[1]))
        self.assertEqual(out.data.shape, (self.nx, self.ny))

        # Scaling rule for meters
        self.assertTrue(np.allclose(out.data, gv_direct.data * 1e3))

        self.assertTrue(np.isfinite(out.data).all())
        self.assertGreater(out.data.mean(), 0.0)

    def test_groupvelocity_from_ensemble_with_log_z_forwards_array(self):
        """z_axis_log=True => forwards non-None z of requested length."""
        class SPGEProbe(SeismicPropertyGridEnsemble):
            def __init__(self, inner):
                self.__dict__.update(inner.__dict__)
                self._inner = inner
                self.captured = None

            def to_surface_velocities(self, **kwargs):
                self.captured = kwargs.copy()
                return self._inner.to_surface_velocities(**kwargs)

        probe = SPGEProbe(self.spge_m)
        npts = 25

        out = self._from_group(
            seismic_param=probe,
            period=self.period,
            phase=self.phase,
            z_axis_log=True,
            npts_log_scale=npts,
            disba_param=DisbaParam(),
        )

        self.assertIsNotNone(probe.captured)
        self.assertIn("z", probe.captured)
        self.assertIsNotNone(probe.captured["z"])
        self.assertEqual(len(probe.captured["z"]), npts)

        self.assertEqual(out.data.shape, (self.nx, self.ny))
        self.assertEqual(tuple(out.origin), (self.origin_m[0], self.origin_m[1]))
        self.assertEqual(tuple(out.spacing), (self.spacing_m[0], self.spacing_m[1]))

    def test_groupvelocity_from_ensemble_kilometers_no_scaling(self):
        """Kilometers grid => classmethod output equals direct result (no ×1e3)."""
        # Convert spatial units to km
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

        gv_list_km = spge_km.to_surface_velocities(
            period_min=self.period,
            period_max=self.period,
            n_periods=1,
            logspace=False,
            phase=self.phase,
            z=None,
            disba_param=disba_param,
            velocity_type=VelocityType.GROUP,
        )
        gv_direct_km = gv_list_km[0]

        out_km = self._from_group(
            seismic_param=spge_km,
            period=self.period,
            phase=self.phase,
            z_axis_log=False,
            disba_param=disba_param,
        )

        # No scaling expected for kilometers
        self.assertTrue(np.allclose(out_km.data, gv_direct_km.data))
        self.assertEqual(tuple(out_km.origin), (origin_km[0], origin_km[1]))
        self.assertEqual(tuple(out_km.spacing), (spacing_km[0], spacing_km[1]))
        self.assertEqual(out_km.grid_units, unit_km)
        self.assertEqual(out_km.grid_type, grtype_km)


if __name__ == "__main__":
    unittest.main()







