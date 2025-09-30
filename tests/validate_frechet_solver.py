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
from scipy import sparse

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
    _deduplicate_points,
)


VELOCITY_M_PER_S = 3000.0
TOLERANCE = 1e-2
NUM_SOURCES = 40
NUM_RECEIVERS = 40


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
    frechet, tt = model.compute_frechet(
        sources=sources,
        receivers=receivers,
        method="fmm",
        cell_slowness=True,
        tt_cal=True,
        threads=1,
        pairwise=True,
    )

    for idx, (src, rcv, row, tt_val) in enumerate(zip(sources.locs[:, :2], receivers, frechet, tt)):
        expected_distance = math.dist(src, rcv)
        path_length = row.sum()
        if not math.isclose(path_length, expected_distance, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            raise AssertionError(
                f"Path length mismatch for pair {idx}: expected {expected_distance:.4f} m, "
                f"got {path_length:.4f} m"
            )
        if not math.isclose(tt_val, expected_distance / VELOCITY_M_PER_S, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            raise AssertionError("Travel-time sanity check failed.")


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
        pairwise=True,
    )

    inv_velocity_sq = 1.0 / (VELOCITY_M_PER_S ** 2)

    for idx, (src, rcv, row) in enumerate(zip(sources.locs[:, :2], receivers, frechet)):
        expected_distance = math.dist(src, rcv)
        derivative_sum = row.sum()
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

    rng = np.random.default_rng(2025)

    base_source_count = max(1, NUM_SOURCES // 2)
    base_receiver_count = max(1, NUM_RECEIVERS // 2)

    base_sources = rng.uniform(low=5.0, high=45.0, size=(base_source_count, 2))
    base_receivers = rng.uniform(low=5.0, high=45.0, size=(base_receiver_count, 2))

    source_duplicates = base_sources[: NUM_SOURCES - base_source_count]
    receiver_duplicates = base_receivers[: NUM_RECEIVERS - base_receiver_count]

    source_coords = np.vstack([base_sources, source_duplicates])
    receiver_coords = np.vstack([base_receivers, receiver_duplicates])

    sources = build_sources(source_coords)

    if HAS_SKFMM:
        logger.info("Running FMM-based Frechet checks (scikit-fmm detected).")
        start = time.perf_counter()
        validate_slowness_derivatives(model, sources, receiver_coords)
        validate_velocity_derivatives(model, sources, receiver_coords)
        _ = model.compute_frechet(
            sources=sources,
            receivers=receiver_coords,
            method="fmm",
            cell_slowness=True,
            tt_cal=False,
            progress=True,
            pairwise=True,
        )
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
        pairwise=True,
    )
    ttcrpy_elapsed = time.perf_counter() - start

    n_pairs = len(source_coords)
    n_model_cells = model.data.size

    assert frechet_ttcrpy.shape[0] == n_pairs
    assert frechet_ttcrpy.shape[1] == n_model_cells
    if sparse.issparse(frechet_ttcrpy):
        assert np.all(np.isfinite(frechet_ttcrpy.data))
    else:
        assert np.all(np.isfinite(frechet_ttcrpy))

    # Order consistency check using a single raytrace over the unique sets
    unique_src_coords, src_inverse, _ = _deduplicate_points(source_coords)
    unique_rcv_coords, rcv_inverse, _ = _deduplicate_points(receiver_coords)

    unique_src_seeds = build_sources(unique_src_coords)
    frechet_unique = model.compute_frechet(
        sources=unique_src_seeds,
        receivers=unique_rcv_coords,
        method="ttcrpy",
        cell_slowness=True,
        tt_cal=False,
        progress=False,
        pairwise=False,
    )

    recomposed = frechet_unique[src_inverse, rcv_inverse, :]
    frechet_dense = frechet_ttcrpy.toarray() if sparse.issparse(frechet_ttcrpy) else frechet_ttcrpy
    if not np.allclose(frechet_dense, recomposed):
        raise AssertionError("Frechet matrix reordering check failed.")

    logger.info(f"ttcrpy backend validation completed in {ttcrpy_elapsed:.2f}s.")

    if HAS_SKFMM:
        logger.info("Frechet validation passed for FMM and ttcrpy backends.")
    else:
        logger.info("Frechet validation passed for the ttcrpy backend (scikit-fmm not installed).")


if __name__ == "__main__":
    main()
