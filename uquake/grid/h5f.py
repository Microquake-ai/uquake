import numpy as np
import h5py


class H5TTable(object):
    """docstring for H5TTable"""

    def __init__(self, path, dset_key=None):
        self.path = path
        self.hf = h5py.File(path, 'r')
        self.keys = list(self.hf.keys())

        self.dset = None

        if dset_key is not None:
            self.set_dataset(dset_key)

        self.stations = self.hf['stations'][:].astype('U4')
        self._stadict = dict(zip(self.stations, np.arange(len(self.stations))))

        self.locations = self.hf['locations'][:]
        self.coords = self.hf['grid_locs'][:]

    def set_dataset(self, key):
        if key in self.keys:
            self.dset = self.hf[key]
        else:
            raise KeyError('dataset %s does not exist' % key)

    @property
    def shape(self):
        return self.hf.attrs['shape']

    @property
    def origin(self):
        return self.hf.attrs['origin']

    @property
    def spacing(self):
        return self.hf.attrs['spacing']

    def index_sta(self, stations):
        if isinstance(stations, (list, np.ndarray)):
            return np.array([self._stadict[sta] for sta in stations])
        else:
            return self._stadict[stations]

    def icol_to_xyz(self, index):
        nx, ny, nz = self.shape
        iz = index % nz
        iy = ((index - iz) // nz) % ny
        ix = index // (nz * ny)
        loc = np.array([ix, iy, iz], dtype=float) * self.spacing + self.origin

        return loc

    def xyz_to_icol(self, loc):
        x, y, z = loc
        ix, iy, iz = ((loc - self.origin) / self.spacing).astype(int)
        nx, ny, nz = self.shape
        # return (iz * nx * ny) + (iy * nx) + ix;

        return int((ix * ny * nz) + (iy * nz) + iz)

    def close(self):
        self.hf.close()


def gdef_to_points(shape, origin, spacing):
    maxes = origin + shape * spacing
    x = np.arange(origin[0], maxes[0], spacing).astype(np.float32)
    y = np.arange(origin[1], maxes[1], spacing).astype(np.float32)
    z = np.arange(origin[2], maxes[2], spacing).astype(np.float32)
    points = np.zeros((np.product(shape), 3), dtype=np.float32)
    # points = np.stack(np.meshgrid(x, y, z), 3).reshape(3, -1).astype(np.float32)
    ix = 0

    for xv in x:
        for yv in y:
            for zv in z:
                points[ix] = [xv, yv, zv]
                ix += 1

    return points
