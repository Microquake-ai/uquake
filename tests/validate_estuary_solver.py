"""Validation script for the Estuary eikonal/ray-tracing pipeline.

The script mirrors ``validate_frechet_solver.py`` but exercises the revived
``eikonal`` package found under ``libraries/estuary``.  It validates that the
fast marching solver and its sensitivity (Frechet) kernel produce consistent
travel times and path lengths for a handful of source/receiver pairs (with
intentional duplicates to exercise the bookkeeping).

Run from the repository root:

    python tests/validate_estuary_solver.py

The Estuary extensions must be built beforehand, e.g.:

    cd libraries/estuary/src/eikonal-ng
    python setup.py build_ext --inplace

"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import sparse

try:
    from eikonal import raytrace as eray
    from eikonal import solver as esolver
except ImportError:  # pragma: no cover - handled at runtime
    print(
        "Estuary eikonal extensions are not importable. Build them first with\n"
        "    cd libraries/estuary/src/eikonal-ng\n"
        "    python setup.py build_ext --inplace\n"
    )
    raise SystemExit(0)


@dataclass(frozen=True)
class Pair:
    source: Tuple[float, float]
    receiver: Tuple[float, float]


def _deduplicate_rows(rows: Iterable[sparse.csr_matrix]) -> Dict[str, sparse.csr_matrix]:
    """Utility to ensure duplicate pairs produce identical rows."""

    cache: Dict[str, sparse.csr_matrix] = {}
    for row in rows:
        key = hash(tuple(row.indices)), hash(tuple(np.round(row.data, 8)))
        if key not in cache:
            cache[key] = row
        else:
            # Verify that duplicate rows are identical within tolerance
            if not np.allclose(row.toarray(), cache[key].toarray()):
                raise AssertionError("Duplicate pair produced mismatched Frechet rows.")
    return cache


def _compute_arrival(
    velocity: np.ndarray,
    source: Tuple[float, float],
    spacing: float,
) -> np.ndarray:
    """Compute the travel-time grid for a single source using the Estuary solver."""

    shape = np.array(velocity.shape)
    padded_shape = tuple(shape + 4)

    viscosity = np.zeros(padded_shape, dtype=np.float64)
    inner_slice = (slice(2, -2),) * velocity.ndim
    viscosity[inner_slice] = 1.0 / velocity

    tag = np.full(padded_shape, 2, dtype=np.int32)
    tag[inner_slice] = 0

    arrival = np.empty(padded_shape, dtype=np.float64)
    arrival.fill(np.inf)

    seeds = np.asarray(source, dtype=np.float64).reshape(1, -1) + 2.0
    start_times = np.zeros(seeds.shape[0], dtype=np.float64)

    esolver.SFMM(
        np.ascontiguousarray(seeds),
        start_times,
        tag,
        viscosity,
        arrival,
        spacing,
        second_order=True,
    )

    return np.ascontiguousarray(arrival[[slice(2, -2)] * velocity.ndim])


def _pair_row(
    arrival: np.ndarray,
    velocity: np.ndarray,
    source: Tuple[float, float],
    receiver: Tuple[float, float],
    spacing: float,
) -> Tuple[sparse.csr_matrix, float]:
    """Return a CSR row containing the Frechet sensitivities for one pair."""

    n_cells = arrival.size
    max_entries = n_cells

    indices_buffer = np.zeros(max_entries, dtype=np.uintp)
    sens_buffer = np.zeros(max_entries, dtype=np.float64)

    travel_time, used_indices, used_sens = eray.sensivity2(
        np.ascontiguousarray(arrival),
        np.ascontiguousarray(velocity),
        tuple(float(c) for c in source),
        tuple(float(c) for c in receiver),
        indices_buffer,
        sens_buffer,
        spacing,
        1.0,
    )

    used_len = used_indices.shape[0]
    if used_len == 0:
        row = sparse.csr_matrix((1, n_cells), dtype=np.float64)
    else:
        row = sparse.csr_matrix(
            (
                used_sens[:used_len],
                (np.zeros(used_len, dtype=np.int32), used_indices[:used_len]),
            ),
            shape=(1, n_cells),
            dtype=np.float64,
        )

    return row, float(travel_time)


def validate_estuary():
    velocity_value = 3000.0  # m/s
    spacing = 1.0
    shape = (41, 41)

    velocity_grid = np.full(shape, velocity_value, dtype=np.float64)

    rng = np.random.default_rng(2025)
    base_sources = rng.integers(low=5, high=35, size=(20, 2)).astype(np.float64)
    base_receivers = rng.integers(low=5, high=35, size=(20, 2)).astype(np.float64)

    sources = np.vstack([base_sources, base_sources[:5]])
    receivers = np.vstack([base_receivers, base_receivers[:5]])

    pairs = [Pair(tuple(src), tuple(rcv)) for src, rcv in zip(sources, receivers)]

    unique_sources, src_inverse, src_rep = _deduplicate_coords(sources)
    arrivals = {
        idx: _compute_arrival(velocity_grid, tuple(src), spacing)
        for idx, src in enumerate(unique_sources)
    }

    rows = []
    travel_times = []

    for pair_idx, pair in enumerate(pairs):
        arrival = arrivals[src_inverse[pair_idx]]
        row, tt = _pair_row(arrival, velocity_grid, pair.source, pair.receiver, spacing)
        rows.append(row)
        travel_times.append(tt)

        path_length = row.data.sum()
        expected_distance = math.dist(pair.source, pair.receiver) * spacing
        expected_tt = expected_distance / velocity_value

        if not math.isclose(path_length, expected_distance, rel_tol=1e-3, abs_tol=1e-3):
            raise AssertionError(
                f"Path length mismatch for pair {pair_idx}: expected {expected_distance:.6f} "
                f"got {path_length:.6f}"
            )
        if not math.isclose(tt, expected_tt, rel_tol=1e-3, abs_tol=1e-3):
            raise AssertionError(
                f"Travel time mismatch for pair {pair_idx}: expected {expected_tt:.6f} "
                f"got {tt:.6f}"
            )

    frechet_matrix = sparse.vstack(rows, format="csr")
    travel_times = np.asarray(travel_times, dtype=np.float64)

    if frechet_matrix.shape != (len(pairs), velocity_grid.size):
        raise AssertionError(
            f"Unexpected Frechet shape {frechet_matrix.shape}, expected {(len(pairs), velocity_grid.size)}"
        )

    _deduplicate_rows(rows)

    print(
        "Estuary validation passed for "
        f"{len(pairs)} pairs (matrix shape {frechet_matrix.shape})."
    )


def _deduplicate_coords(coords: np.ndarray, *, decimals: int = 8):
    coords = np.asarray(coords, dtype=float)
    unique_list = []
    inverse = np.empty(coords.shape[0], dtype=int)
    mapping = {}
    rep = []
    for idx, row in enumerate(coords):
        key = tuple(np.round(row, decimals=decimals))
        if key not in mapping:
            mapping[key] = len(unique_list)
            unique_list.append(row)
            rep.append(idx)
        inverse[idx] = mapping[key]
    return np.array(unique_list, dtype=float), inverse, rep


if __name__ == "__main__":
    validate_estuary()
