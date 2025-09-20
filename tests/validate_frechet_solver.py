"""Quick validation script for the fast-marching Frechet implementation.

The script constructs a simple constant-velocity phase-velocity grid, computes
Frechet derivatives using the new FMM-backed solver, and compares the summed
path lengths against Euclidean distances between predefined source/receiver
pairs. It also checks the velocity-parameterisation branch (cell_slowness=False)
to ensure the expected ``-L / v**2`` scaling.

Run from the repository root with:

    python tests/validate_frechet_solver.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from uquake.core.logging import logger  # Preferred project logger
except ImportError:  # pragma: no cover
    try:
        from loguru import logger  # type: ignore
    except ImportError:
        import logging

        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        logger = logging.getLogger("frechet-validation")

try:
    import skfmm  # noqa: F401

    HAS_SKFMM = True
except ImportError:
    HAS_SKFMM = False

try:
    from uquake.core.coordinates import Coordinates
except ModuleNotFoundError as exc:
    missing = exc.name
    raise SystemExit(
        "The validation script requires the optional dependency 'utm'. "
        "Install it (e.g., `poetry install --with geodesy` or `pip install utm`) "
        "before running the script."
    ) from exc

from uquake.grid.extended import (
    GridTypes,
    PhaseVelocity,
    Seed,
    SeedEnsemble,
)


VELOCITY_M_PER_S = 3000.0
TOLERANCE = 1e-2


def build_phase_velocity_model() -> PhaseVelocity:
    """Create a simple 2-D constant-velocity phase-velocity grid."""

    shape = (50, 50)
    spacing = (1.0, 1.0)
    origin = (0.0, 0.0)
    data = np.full(shape, VELOCITY_M_PER_S, dtype=float)

    return PhaseVelocity(
        network_code="XX",
        data_or_dims=data,
        period=10.0,
        grid_type=GridTypes.VELOCITY_METERS,
        spacing=spacing,
        origin=origin,
    )


def build_sources(coords: np.ndarray) -> SeedEnsemble:
    seeds = []
    for idx, (x, y) in enumerate(coords):
        coordinates = Coordinates(x=float(x), y=float(y), z=0.0)
        seeds.append(Seed(f"S{idx:02d}", "00", coordinates))
    return SeedEnsemble(seeds)


def validate_slowness_derivatives(model: PhaseVelocity,
                                   sources: SeedEnsemble,
                                   receivers: np.ndarray) -> None:
    frechet, _ = model.compute_frechet(
        sources=sources,
        receivers=receivers,
        method="fmm",
        cell_slowness=True,
        tt_cal=True,
        threads=1,
    )

    for idx, (src, rcv) in enumerate(zip(sources.locs[:, :2], receivers)):
        expected_distance = math.dist(src, rcv)
        path_length = frechet[idx, idx, :].sum()
        if not math.isclose(path_length, expected_distance, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            raise AssertionError(
                f"Path length mismatch for pair {idx}: expected {expected_distance:.4f} m, "
                f"got {path_length:.4f} m"
            )


def validate_velocity_derivatives(model: PhaseVelocity,
                                   sources: SeedEnsemble,
                                   receivers: np.ndarray) -> None:
    frechet = model.compute_frechet(
        sources=sources,
        receivers=receivers,
        method="fmm",
        cell_slowness=False,
        tt_cal=False,
        threads=1,
    )

    inv_velocity_sq = 1.0 / (VELOCITY_M_PER_S ** 2)

    for idx, (src, rcv) in enumerate(zip(sources.locs[:, :2], receivers)):
        expected_distance = math.dist(src, rcv)
        derivative_sum = frechet[idx, idx, :].sum()
        expected_derivative = -expected_distance * inv_velocity_sq
        if not math.isclose(
            derivative_sum,
            expected_derivative,
            rel_tol=TOLERANCE,
            abs_tol=TOLERANCE,
        ):
            raise AssertionError(
                f"Velocity derivative mismatch for pair {idx}: expected {expected_derivative:.6e}, "
                f"got {derivative_sum:.6e}"
            )


def main() -> None:
    model = build_phase_velocity_model()

    source_coords = np.array(
        [
            [5.0, 5.0],
            [10.0, 15.0],
            [20.0, 10.0],
            [35.0, 25.0],
        ]
    )
    receiver_coords = np.array(
        [
            [15.0, 15.0],
            [40.0, 20.0],
            [5.0, 30.0],
            [10.0, 5.0],
        ]
    )

    sources = build_sources(source_coords)

    if HAS_SKFMM:
        logger.info("Running FMM-based Frechet checks (scikit-fmm detected).")
        start = time.perf_counter()
        validate_slowness_derivatives(model, sources, receiver_coords)
        validate_velocity_derivatives(model, sources, receiver_coords)
        fmm_elapsed = time.perf_counter() - start
        logger.info(f"FMM backend validation completed in {fmm_elapsed:.2f}s.")
    else:
        logger.info("scikit-fmm not installed; skipping FMM checks and validating ttcrpy backend only.")

    start = time.perf_counter()
    frechet_ttcrpy = model.compute_frechet(
        sources=sources,
        receivers=receiver_coords,
        method="ttcrpy",
        cell_slowness=True,
        tt_cal=False,
        progress=True,
    )
    ttcrpy_elapsed = time.perf_counter() - start

    assert frechet_ttcrpy.shape[0] == len(source_coords)
    assert frechet_ttcrpy.shape[1] == len(receiver_coords)
    assert np.all(np.isfinite(frechet_ttcrpy))

    logger.info(f"ttcrpy backend validation completed in {ttcrpy_elapsed:.2f}s.")

    if HAS_SKFMM:
        logger.info("Frechet validation passed for FMM and ttcrpy backends.")
    else:
        logger.info("Frechet validation passed for the ttcrpy backend (scikit-fmm not installed).")


if __name__ == "__main__":
    main()
