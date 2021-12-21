import os
from glob import glob
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

        self.sites = self.hf['sites'][:].astype('U6')
        self.stations = self.hf['stations'][:].astype('U4')
        self.station_locations = self.hf['station_locations'].astype('U2')
        self._sitedict = dict(zip(self.sites, np.arange(len(self.sites))))

        self.locations = self.hf['locations'][:]
        self.coords = self.hf['grid_locs'][:]

    def __delete__(self):
        sefl.hf.close()

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

    def index_sites(self, sites):
        if isinstance(sites, (list, np.ndarray)):
            return np.array([self._sitedict[site] for site in sites])
        else:
            return self._sitedict[site]

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
    x = np.arange(origin[0], maxes[0], spacing[0]).astype(np.float32)
    y = np.arange(origin[1], maxes[1], spacing[1]).astype(np.float32)
    z = np.arange(origin[2], maxes[2], spacing[2]).astype(np.float32)
    points = np.zeros((np.product(shape), 3), dtype=np.float32)
    ix = 0

    for xv in x:
        for yv in y:
            for zv in z:
                points[ix] = [xv, yv, zv]
                ix += 1

    return points

def array_from_travel_time_ensemble(tt_grids):

    data = {'P': [],
            'S': []}

    sites = []
    slocs = []
    shape = tt_grids[0].shape
    origin = tt_grids[0].origin
    spacing = tt_grids[0].spacing
    for tt_grid in tt_grids:
        sites.append(tt_grid.seed_label)
        slocs.append(tt_grid.seed)

    sites = np.array(sites)
    slocs = np.array(slocs)

    nsites = len(sites)
    ngrid = np.product(shape)

    tts = np.zeros((nsites, ngrid), dtype=np.float32)

    for i in range(nsites):
        tts[i] = tt_grids[i].data.reshape(ngrid).astype(np.float32)

    data[phase] = dict(ttable=tts, locations=slocs, shape=shape,
                       origin=origin, spacing=spacing, sites=sites)

    return data


def write_hdf5(fname, tt_grids):

    hf = h5py.File(fname, 'w')

    shape = tt_grids.shape
    spacing = tt_grids.spacing
    origin = tt_grids.origin

    hf.attrs['shape'] = shape
    hf.attrs['spacing'] = spacing
    hf.attrs['origin'] = origin

    sites = tt_grids.seed_labels
    locations = tt_grids.seeds

    hf.create_dataset('locations', data=locations.astype(np.float32))
    gridlocs = gdef_to_points(shape, origin, spacing)
    hf.create_dataset('grid_locs', data=gridlocs.astype(np.float32))
    gdef = np.concatenate((shape, origin, spacing)).astype(np.int32)
    hf.create_dataset('grid_def', data=gdef)
    hf.create_dataset('sites', data=sites.astype('S6'))
    stations = np.array([site[0:4] for site in sites])
    station_locations = np.array([site[4:] for site in sites])
    hf.create_dataset('stations', data=stations.astype('S4'))
    hf.create_dataset('station_locations', data=station_locations.astype('S2'))

    nsites = len(sites)
    ngrid = np.product(shape)

    tts = {'P': np.zeros((nsites, ngrid), dtype=np.float32),
           'S': np.zeros((nsites, ngrid), dtype=np.float32)}

    for i, site in enumerate(sites):
        for phase in ['P', 'S']:
            tt_grid = tt_grids.select(phase=phase, seed_labels=site)[0]
            tts[phase][i] = tt_grid.data.reshape(ngrid).astype(np.float32)

    hf.create_dataset('ttp', data=tts['P'])
    hf.create_dataset('tts', data=tts['S'])
    hf.close()
