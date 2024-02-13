import pandas as pd
import numpy as np
import vtk
from dxfwrite import DXFEngine as dxf
from io import BytesIO


class Borehole:
    def __init__(self, depth=None, x=None, y=None, z=None, collar=None,
                 toe=None, magnetic_declination=0):
        """

        :param depth: depth vector
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        :param collar: collar location vector (x, y, z)
        :param toe: toe location vector (x, y, z)
        :param magnetic_declination: magnetic declination in degree from
        true north
        """
        self.depth = depth
        self.x = x
        self.y = y
        self.z = z
        self.collar = collar
        self.toe = toe
        self.magnetic_declination = magnetic_declination
        pass

    @classmethod
    def from_gyro_file(cls, gyro_file, collar,
                       magnetic_declination=0, resolution=1, **kwargs):
        gyro_df = read_gyro_file(gyro_file, **kwargs)
        collar = np.array(collar)
        magnetic_declination = magnetic_declination
        trace_df = gyro_to_borehole_trace(gyro_df, collar,
                                          magnetic_declination, dl=resolution)

        x = trace_df['x'].values
        y = trace_df['y'].values
        z = trace_df['z'].values
        toe = np.array([x[-1], y[-1], z[-1]])
        depth = np.array(trace_df.index)
        return cls(depth=depth, x=x, y=y, z=z, collar=collar, toe=toe,
                   magnetic_declination=magnetic_declination)

    @property
    def trace(self):
        """
        return a dictionary containing the trace of the borehole
        :return: borehole trace
        """
        dict_out = {'depth': self.depth,
                    'x': self.x,
                    'y': self.y,
                    'z': self.z}

        return dict_out

    @property
    def length(self):
        return np.max(self.depth)

    @property
    def dip_azimuth(self):
        h = np.sqrt(x ** 2 + y ** 2)
        dip = np.arctan2(h, z)
        azimuth = np.arctan2(y, x)

        return {'depth': self.depth, 'dip': dip, 'azimuth': azimuth}

    def resample(self, resolution=1):
        dict_data = {'depth': self.depth,
                     'x': self.x,
                     'y': self.y,
                     'z': self.z}

        df = pd.DataFrame(dict_data)

        df.set_index('depth')

        df = df.reindex(np.arange(np.min(np.array(df.index)),
                                  np.max(np.array(df.index)) + resolution,
                                  resolution))
        df = df.apply(pd.Series.interpolate)

        x = df['x'].values
        y = df['y'].values
        z = df['z'].values
        depth = np.array(df.index)

        return {'depth': depth, 'x': x, 'y': y, 'z': z}

    def interpolate(self, depth):
        """
        :param depth: depth along the borehole
        :return: x, y, z at a specific depth along the borehole
        """
        x_i = np.interp(depth, self.depth, self.x)
        y_i = np.interp(depth, self.depth, self.y)
        z_i = np.interp(depth, self.depth, self.z)

        return x_i, y_i, z_i

    def orientation(self, depth, collar_to_toe=True):
        """
        returns the unit vector representing the orientation of a sensor
        aligned along the borehole axis.
        :param depth: depth or distance along the borhole
        :param collar_to_toe: True if pointing towards the collar.
        False if pointing towards the toe.
        :return: a unit vector representing the orientation
        """

        l1 = self.interpolate(depth + 1)  # towards toe
        l2 = self.interpolate(depth - 1)  # towards collar

        if collar_to_toe:
            orientation = l1 - l2
        else:
            orientation = l2 - l1

        orientation /= np.linalg.norm(orientation)

        return orientation

    def to_vtk_poly_data(self):
        """
        :return: a vtk polydata object
        """
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        line_vtk = vtk.vtkLine()
        for k in range(0, len(self.x) - 1):
            x0 = self.x[k]
            y0 = self.y[k]
            z0 = self.z[k]

            x1 = self.x[k+1]
            y1 = self.y[k+1]
            z1 = self.z[k+1]

            id1 = points.InsertNextPoint((x0, y0, z0))
            line_vtk.GetPointIds().SetId(0, id1)

            id2 = points.InsertNextPoint((x1, y1, z1))
            line_vtk.GetPointIds().SetId(1, id2)

            lines.InsertNextCell(line_vtk)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)
        return poly_data

    def write_to_vtp(self, vtp_file_path):
        """
        write the borehole trace to a VTP file
        :param vtp_file_path:
        :return:
        """

        vtk_poly_data = self.to_vtk_poly_data()
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(vtp_file_path)
        writer.SetInputData(vtk_poly_data)

        return writer.Write()

    def write_to_dxf(self, dxf_file_path):

        drawing = dxf.drawing(dxf_file_path)

        for k in range(0, len(self.x) - 1):
            x0 = self.x[k]
            y0 = self.y[k]
            z0 = self.z[k]

            x1 = self.x[k+1]
            y1 = self.y[k+1]
            z1 = self.z[k+1]

            drawing.add(dxf.line((x0, y0, z0), (x1, y1, z1), color=7))

        drawing.save()
        return


def read_gyro_file(gyro_file, header=7):
    """
    read a gyro survey file. The gyro survey file must contain three columns
    (DEPTH, DIP, AZI (MAG))
    :param gyro_file: full path to the gyro file
    :param header: the number of header lines
    :return: pandas dataframe contining two column
    """

    df = pd.read_excel(gyro_file, 'Sheet1', header=7)
    gyro_df = pd.DataFrame()
    gyro_df['depth'] = df['DEPTH']
    gyro_df['azimuth (mag)'] = df['AZI (MAG)']
    gyro_df['dip'] = df['DIP']

    gyro_df = gyro_df.set_index('depth')

    return gyro_df


def gyro_to_borehole_trace(gyro_df, collar, magnetic_declination, dl=1):
    """
    convert gyro survey data expressed in dip and azimuth to the x, y and z
    trace
    :param gyro_df: a dataframe containing the depth, dip and
    azimuth
    :param collar: list or array containing the borehole collar coordinates
    :param magnetic_declination: magnetic declination in degrees from north
    :param dl: resolution between the point forming the trace
    :return: a dataframe sampled at a resolution of dl containing the depth,
    azimuth, dip, x, y an z along the borehole
    """

    gyro = gyro_df[gyro_df['dip'].notnull()]
    gyro['azimuth'] = gyro['azimuth (mag)'] - magnetic_declination
    gyro['azimuth'] = gyro['azimuth'] / 180 * np.pi
    gyro['dip'] = gyro['dip'] / 180 * np.pi
    gyro['azimuth (mag)'] = gyro['azimuth (mag)'] / 180 * np.pi

    gyro = gyro.reindex(np.arange(np.min(np.array(gyro.index)),
                                  np.max(np.array(gyro.index)) + dl, dl))
    gyro = gyro.apply(pd.Series.interpolate)

    x = np.zeros(len(gyro))
    x[0] = collar[0]
    y = np.zeros(len(gyro))
    y[0] = collar[1]
    z = np.zeros(len(gyro))
    z[0] = collar[2]
    dip = gyro['dip']
    azimuth = gyro['azimuth']
    for k in range(0, len(gyro.index) - 1):
        dy_x = dl * np.cos(gyro.iloc[k]['dip'])
        dy = dy_x * np.cos(gyro.iloc[k]['azimuth'])
        dx = dy_x * np.sin(gyro.iloc[k]['azimuth'])
        dz = dl * np.sin(gyro.iloc[k]['dip'])

        x[k + 1] = x[k] + dx
        y[k + 1] = y[k] + dy
        z[k + 1] = z[k] - dz

    gyro['x'] = x
    gyro['y'] = y
    gyro['z'] = z

    return gyro


def borehole_collar_toe_to_trace(collar, toe, dl=1):
    """
    Return the borehole trace assuming a straight line between the borehole
    collar and toe
    :param collar: borehole collar coordinates
    :param toe: borehole toe coordinates
    :param dl: resolution of the output trace
    :return:
    """
    df = pd.DataFrame()

    max_depth = np.linalg.norm(toe - collar)
    depth = np.arange(0, max_depth, dl)

    x_borehole = np.linspace(collar[0], toe[0], len(depth))
    y_borehole = np.linspace(collar[1], toe[1], len(depth))
    z_borehole = np.linspace(collar[2], toe[2], len(depth))

    h_borehole = np.sqrt(x_borehole ** 2 + y_borehole ** 2)

    dip = np.arctan2(h_borehole, z_borehole)
    azimuth = np.arctan2(y_borehole, x_borehole)

    df['x'] = x_borehole
    df['y'] = y_borehole
    df['z'] = z_borehole

    df['dip'] = dip
    df['azimuth'] = azimuth
    df['depth'] = depth

    df = df.set_index('depth')

    return df
