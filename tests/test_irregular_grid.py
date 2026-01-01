import numpy as np
import xarray as xr
from uquake.core.coordinates import CoordinateSystem
from uquake.core.event import ResourceIdentifier
from uuid import uuid4
from uquake.grid.base import IrregularGrid


# Assuming the IrregularGrid class code is already included here or imported.

# Test function definitions
def test_grid_initialization():
    print("Testing grid initialization...")

    # Case 1: Regular grid with predefined value initialization
    dims = (10, 10)
    value = 5
    grid1 = IrregularGrid(data_or_dims=dims, axes=None, value=value)
    assert np.all(
        grid1.data.values == value), ("Grid values not initialized correctly "
                                      "with constant value")

    # Case 2: Irregular grid with specified axes
    x = np.linspace(0, 9, 10)
    y = np.linspace(0, 9, 10)
    axes = [x, y]
    grid2 = IrregularGrid(data_or_dims=dims, axes=axes, value=value)
    assert grid2.data.shape == (10, 10), "Grid dimensions are incorrect"
    assert np.allclose(grid2.data['dim_0'], x), "X-axis not assigned correctly"
    assert np.allclose(grid2.data['dim_1'], y), "Y-axis not assigned correctly"

    print("Initialization tests passed.")


def test_metadata():
    print("Testing metadata storage...")

    resource_id = str(uuid4())
    coordinate_system = CoordinateSystem.NED
    label = "Test Grid"
    grid = IrregularGrid(data_or_dims=(5, 5), axes=[np.arange(5), np.arange(5)],
                         resource_id=resource_id, coordinate_system=coordinate_system,
                         label=label)

    assert grid.resource_id == resource_id, "Resource ID not stored correctly"
    assert grid.coordinate_system == coordinate_system, "Coordinate system not stored correctly"
    assert grid.label == label, "Label not stored correctly"

    print("Metadata tests passed.")


def test_data_access():
    print("Testing data access through indexing...")

    value = 7
    grid = IrregularGrid(data_or_dims=(3, 3), axes=[np.arange(3), np.arange(3)],
                         value=value)
    assert grid[0, 0] == value, "Data access through indexing failed"
    assert grid.data[1, 2] == value, "Data access through xarray indexing failed"

    print("Data access tests passed.")


# Run tests
if __name__ == "__main__":
    test_grid_initialization()
    test_metadata()
    test_data_access()
