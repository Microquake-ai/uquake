# Copyright (C) 2023, Jean-Philippe Mercier
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from enum import Enum
import json
from obspy.core.util import AttribDict
import utm
import pyproj
import numpy as np
from uquake.core.logging import logger


class CoordinateSystem(Enum):
    """
    Enum class to specify the coordinate system used in microseismic monitoring.
    This coordinate system enforces a right-hand coordinate system.

    Attributes:
    NED -- North, East, Down coordinate system.
    ENU -- East, North, Up coordinate system.
    """

    NED = 'NED'
    END = 'END'
    ENU = 'ENU'
    NEU = 'NEU'

    @classmethod
    def get_transformation_map(cls):
        """Returns the transformation mapping between coordinate systems."""
        return {
            (cls.NED, cls.ENU): lambda x, y, z: (y, x, -z),
            (cls.NED, cls.NEU): lambda x, y, z: (x, y, -z),
            (cls.NED, cls.END): lambda x, y, z: (y, x, z),

            (cls.ENU, cls.NED): lambda x, y, z: (y, x, -z),
            (cls.ENU, cls.NEU): lambda x, y, z: (y, x, z),
            (cls.ENU, cls.END): lambda x, y, z: (x, y, -z),

            (cls.NEU, cls.NED): lambda x, y, z: (x, y, -z),
            (cls.NEU, cls.ENU): lambda x, y, z: (y, x, z),
            (cls.NEU, cls.END): lambda x, y, z: (y, x, -z),

            (cls.END, cls.NED): lambda x, y, z: (y, x, z),
            (cls.END, cls.ENU): lambda x, y, z: (x, y, -z),
            (cls.END, cls.NEU): lambda x, y, z: (y, x, z),
        }

    @classmethod
    def transform_coordinates(cls, source: 'CoordinateSystem',
                              target: 'CoordinateSystem',
                              x: float, y: float, z: float):
        """
        Transform coordinates from one system to another using predefined mappings.

        :param source: Source coordinate system.
        :param target: Target coordinate system.
        :param x: X-coordinate.
        :param y: Y-coordinate.
        :param z: Z-coordinate.
        :return: Transformed coordinates as (x, y, z).
        """
        if source == target:
            return x, y, z  # No transformation needed

        transform_func = cls.get_transformation_map().get((source, target))

        if transform_func is None:
            raise ValueError(
                f"Unsupported coordinate transformation from {source} to {target}")

        return transform_func(x, y, z)

    def __repr__(self):
        if self.name == "NED":
            return "North, East, Down (NED)"
        elif self.name == "ENU":
            return "East, North, Up (ENU)"
        elif self.name == "NEU":
            return "North, East, Up (NEU) [WARNING - LEFT-HANDED COORDINATE SYSTEM]"
        elif self.name == "END":
            return "East, North, Down (END) [WARNING - LEFT-HANDED COORDINATE SYSTEM]"

    def __str__(self):
        return str(self.name)

    def map(self, from_coordinate_system, to_coordinate_system):
        coordinate_system_map = {
            (CoordinateSystem.NED, CoordinateSystem.ENU): (self.y, self.x, -self.z),
            (CoordinateSystem.NED, CoordinateSystem.NEU): (self.x, self.y, -self.z),
            (CoordinateSystem.NED, CoordinateSystem.END): (self.y, self.x, self.z),

            (CoordinateSystem.ENU, CoordinateSystem.NED): (self.y, self.x, -self.z),
            (CoordinateSystem.ENU, CoordinateSystem.NEU): (self.y, self.x, self.z),
            (CoordinateSystem.ENU, CoordinateSystem.END): (self.x, self.y, -self.z),

            (CoordinateSystem.NEU, CoordinateSystem.NED): (self.x, self.y, -self.z),
            (CoordinateSystem.NEU, CoordinateSystem.ENU): (self.y, self.x, self.z),
            (CoordinateSystem.NEU, CoordinateSystem.END): (self.y, self.x, -self.z),

            (CoordinateSystem.END, CoordinateSystem.NED): (self.y, self.x, self.z),
            (CoordinateSystem.END, CoordinateSystem.ENU): (self.x, self.y, -self.z),
            (CoordinateSystem.END, CoordinateSystem.NEU): (self.y, self.x, self.z),
        }

        raise ValueError(f"Cannot map {from_coordinate_system} to {to_coordinate_system}")



class CoordinateTransformation:
    """
    Class to handle transformations between a custom coordinate systems and latitude
    longitude.

    :param translation: Translation vector as (dx, dy, dz).
    :type translation: tuple[float, float, float]
    :param rotation: Rotation matrix or Euler angles.
    :type rotation: list[list[float]] or tuple[float, float, float]
    :param epsg_code: EPSG code for the target coordinate system.
    :type epsg_code: int
    :param scaling: Optional scaling factor or factors.
    :type scaling: float or tuple[float, float, float]
    :param reference_elevation: Reference elevation for depth conversions.
    :type reference_elevation: float
    """

    def __init__(
        self, translation=(0, 0, 0), rotation=None, epsg_code=None, scaling=None,
            reference_elevation=0.0
    ):
        self.translation = translation
        self.rotation = rotation
        self.epsg_code = epsg_code
        self.scaling = scaling if scaling is not None else 1.0
        self.reference_elevation = reference_elevation

    def __eq__(self, other):
        """
        Overrides the default equality method to compare CoordinateTransformation instances.

        :param other: Another CoordinateTransformation object.
        :type other: CoordinateTransformation
        :return: True if instances are equal, False otherwise.
        :rtype: bool
        """
        if not isinstance(other, CoordinateTransformation):
            return False

        if self.translation != other.translation:
            return False

        if self.rotation != other.rotation:
            return False

        if self.epsg_code != other.epsg_code:
            return False

        if self.scaling != other.scaling:
            return False

        if self.reference_elevation != other.reference_elevation:
            return False

        return True

    def __repr__(self):
        out_str = f"""
                translation: {self.translation}
                   rotation: {self.rotation}
                  epsg_code: {self.epsg_code}
                    scaling: {self.scaling}
        reference elevation: {self.reference_elevation}
        """

        return out_str

    def convert_elevation_depth(self, z, coordinate_system, to_depth=True):
        """
        Convert between elevation and depth using a reference elevation or depth and
        scaling.

        :param z: Input elevation or depth.
        :type z: float
        :param coordinate_system: Coordinate system.
        :type coordinate_system: CoordinateSystem
        :param to_depth: If True, convert elevation to depth. If False, convert depth to
        elevation.
        :type to_depth: bool
        :return: Converted depth or elevation.
        :rtype: float
        """

        if (coordinate_system in [CoordinateSystem.NED, CoordinateSystem.END]
                and not to_depth):
            z = -z
        elif (coordinate_system in [CoordinateSystem.NEU, CoordinateSystem.ENU]
                and to_depth):
            z = -z

        if to_depth:
            return (self.reference_elevation - z) * self.scaling
        else:
            return z / self.scaling - self.reference_elevation

    def apply_transformation(self, x, y, z):
        """
        Apply the coordinate transformation to a point.

        :param x: X-coordinate.
        :type x: float
        :param y: Y-coordinate.
        :type y: float
        :param z: Z-coordinate.
        :type z: float
        :return: Transformed (x, y, z).
        :rtype: tuple[float, float, float]
        """
        # Implement your transformation logic here
        pass

    def invert_transformation(self, lat, lon, z):
        """
        Invert the coordinate transformation for a point.

        :param lat: latitude.
        :type lat: float
        :param lon: longitude.
        :type lon: float
        :param z: Z-coordinate.
        :type z: float
        :return: Inverted (x, y, z).
        :rtype: tuple[float, float, float]
        """
        # Implement your inversion logic here
        pass

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, string):
        return cls(**json.loads(string))

    @property
    def utm_crs(self):
        return pyproj.CRS(f'epsg:{self.epsg_code}')

    @property
    def sph_crs(self):
        return pyproj.CRS(f'epsg:4326')

    def to_latlon(self, northing, easting):
        transformation = pyproj.Transformer.from_crs(
            self.utm_crs, self.sph_crs, always_xy=True)
        lon, lat = transformation.transform(easting, northing)
        return lat, lon

    def from_latlon(self, lat, lon):
        easting, northing, zone, hemisphere = utm.from_latlon(lat, lon)
        transformation = pyproj.Transformer.from_crs(
            self.sph_crs, self.utm_crs, always_xy=True)
        return easting, northing


class Coordinates:
    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        coordinate_system: CoordinateSystem = CoordinateSystem.NED,
        transformation: CoordinateTransformation = CoordinateTransformation(),
    ):
        """
        :param x: X-coordinate.
        :type x: float
        :param y: Y-coordinate.
        :type y: float
        :param z: Z-coordinate.
        :type z: float
        :param coordinate_system: Coordinate system, either "NED" or "ENU".
        :type coordinate_system: CoordinateSystem
        :param transformation: Coordinate transformation object.
        :type transformation: CoordinateTransformation
        """
        self.x = x
        self.y = y
        self.z = z
        self.coordinate_system = coordinate_system
        self.transformation = transformation

    @property
    def northing(self):
        """
        Get the northing coordinate.

        :return: Northing coordinate based on system.
        :rtype: float
        """
        return self.x if (self.coordinate_system == CoordinateSystem.NED) or \
                         (self.coordinate_system == CoordinateSystem.NEU) else self.y

    @property
    def easting(self):
        """
        Get the easting coordinate.

        :return: Easting coordinate based on system.
        :rtype: float
        """
        return self.x if (self.coordinate_system == CoordinateSystem.ENU) or \
                         (self.coordinate_system == CoordinateSystem.END) else self.y

    @property
    def up(self):
        """
        Get the up coordinate.

        :return: Up coordinate based on system.
        :rtype: float
        """
        if self.transformation:
            return self.transformation.convert_elevation_depth(
                self.z, self.coordinate_system, to_depth=False
            )
        return -self.z if self.coordinate_system in [
            CoordinateSystem.NED, CoordinateSystem.END] else self.z

    @property
    def down(self):
        """
        Get the down coordinate.

        :return: Down coordinate based on system.
        :rtype: float
        """
        if self.transformation:
            return self.transformation.convert_elevation_depth(
                self.z, self.coordinate_system, to_depth=True
            )
        return self.z if (self.coordinate_system == CoordinateSystem.NED) or \
                         (self.coordinate_system == CoordinateSystem.END) else -self.z

    @property
    def loc(self):
        return np.array([self.x, self.y, self.z])

    def __repr__(self):
        out_str = f"""
                        x: {self.x: 0.2f}
                        y: {self.y: 0.2f}
                        z: {self.z: 0.2f}
        coordinate system: {self.coordinate_system}
           transformation: {self.transformation}
        """
        return out_str

    def __eq__(self, other):
        if not isinstance(other, Coordinates):
            return False
        return (
            True
            if (
                (self.x == other.x)
                & (self.y == other.y)
                & (self.z == other.z)
                & (self.coordinate_system == other.coordinate_system)
                & (self.transformation == other.transformation)
            )
            else False
        )

    def to_json(self):
        out_dict = {}
        for key in self.__dict__.keys():
            # from ipdb import set_trace
            # set_trace()
            if key == "coordinate_system":
                out_dict[key] = str(self.coordinate_system)
            elif key == "transformation":
                if self.transformation is not None:
                    out_dict[key] = self.transformation.to_json()
                else:
                    out_dict[key] = None
            else:
                out_dict[key] = self.__dict__[key]
        return json.dumps(out_dict)

    def convert_coordinate_system(
            self, target_system: CoordinateSystem
    ) -> 'Coordinates':
        """
        Convert coordinates to a different coordinate system and return a new
        Coordinates object.

        :param target_system: Target coordinate system.
        :return: New Coordinates object in the target coordinate system.
        """
        new_x, new_y, new_z = CoordinateSystem.transform_coordinates(
            self.coordinate_system, target_system, self.x, self.y, self.z
        )

        return Coordinates(new_x, new_y, new_z, target_system, self.transformation)

    def rotate_azimuth(self, azimuth) -> 'Coordinates':
        """
        Rotate the coordinates around the z-axis by the given azimuth angle,
        taking the coordinate system into account.

        :param azimuth: Azimuth angle in degrees.
        :return: Rotated Coordinates object.
        """
        new_x, new_y = rotate_azimuth(self.x, self.y, self.coordinate_system, azimuth)

        return Coordinates(new_x, new_y, self.z, self.coordinate_system, self.transformation)

    @classmethod
    def from_json(cls, json_string):
        in_dict = json.loads(json_string)
        coordinate_system = getattr(CoordinateSystem, in_dict["coordinate_system"])

        if in_dict["transformation"] is not None:
            transformation = CoordinateTransformation.from_json(
                in_dict["transformation"])
        else:
            transformation = None

        return cls(
            in_dict["x"],
            in_dict["y"],
            in_dict["z"],
            coordinate_system=coordinate_system,
            transformation=transformation,
        )

    def to_extra_key(self, namespace="mq"):
        return AttribDict({"value": self.to_json(), "namespace": namespace})

    @classmethod
    def from_extra_key(cls, extra_key):
        return cls.from_json(extra_key["value"])

    @classmethod
    def from_lat_lon(cls, latitude, longitude, z,
                     coordinate_system: CoordinateSystem = CoordinateSystem.NED):
        easting, northing, zone_number, lat_zone = utm.from_latlon(latitude, longitude)
        if latitude >= 0:
            epsg_code = 32600 + zone_number  # Northern hemisphere
        else:
            epsg_code = 32700 + zone_number  # Southern hemisphere

        coordinate_transformation = CoordinateTransformation(epsg_code=epsg_code)

        # northing, easting = coordinate_transformation.from_latlon(latitude, longitude)

        if str(coordinate_system)[:2] == 'NE':
            x = northing
            y = easting
        else:
            x = easting
            y = northing

        return cls(x, y, z, coordinate_system=coordinate_system, transformation=coordinate_transformation)

    def to_lat_lon(self):
        if self.transformation.rotation is not None:
            logger.warning('the rotation is not implemented')
        if (self.coordinate_system == CoordinateSystem.NED or
                self.coordinate_system == CoordinateSystem.NEU):

            northing = self.x - self.transformation.translation[0]
            easting = self.y - self.transformation.translation[1]
        else:
            northing = self.y - self.transformation.translation[0]
            easting = self.x - self.transformation.translation[1]

        return self.transformation.to_latlon(northing, easting)

    @property
    def latitude(self):
        return self.to_lat_lon()[0]

    @property
    def longitude(self):
        return self.to_lat_lon()[1]

    @property
    def depth(self):
        return self.down

    @property
    def elevation(self):
        return self.up


def rotate_azimuth(
        x: float, y: float, coordinate_system: CoordinateSystem, azimuth: float
):
    """
    Coordinate system aware rotation of coordinates around the z-axis by the given azimuth
    Args:
        x: x-coordinate
        y: y-coordinate
        coordinate_system: coordinate system of the input coordinates
        azimuth: azimuth angle in degrees

    Returns:

    """
    azimuth_rad = np.radians(azimuth)
    cos_azimuth = np.cos(azimuth_rad)
    sin_azimuth = np.sin(azimuth_rad)

    if coordinate_system in [CoordinateSystem.NED, CoordinateSystem.NEU]:
        new_x = x * cos_azimuth - y * sin_azimuth
        new_y = x * sin_azimuth + y * cos_azimuth
    else:  # CoordinateSystem.ENU or CoordinateSystem.END
        new_x = y * cos_azimuth - x * sin_azimuth
        new_y = y * sin_azimuth + x * cos_azimuth
    return new_x, new_y

