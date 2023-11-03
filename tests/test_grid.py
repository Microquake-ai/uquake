from uquake.grid.base import Grid
from uquake.grid import extended
from importlib import reload
reload(extended)

grid_dimensions = [10, 10, 10]
grid_spacing = [1, 1, 1]
grid_origin = [0, 0, 0]
grid = Grid(grid_dimensions, grid_spacing, grid_origin, value=1)
seeds = extended.SeedEnsemble.generate_random_seeds_in_grid(grid, n_seeds=10)

