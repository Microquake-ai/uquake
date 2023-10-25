from enum import Enum
import json
from obspy.core.util import AttribDict


class CoordinateSystem(Enum):
    """
    Enum class to specify the coordinate system used in microseismic monitoring.
    This coordinate system enforces a right-hand coordinate system.

    Attributes:
    NED -- North, East, Down coordinate system.
    ENU -- East, North, Up coordinate system.
    """

    NED = "NED"
    ENU = "ENU"

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return 'North, East, Down (NED)' if self.value == 'NED' \
            else 'East, North, Up (ENU)'


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
    def __init__(self, translation=0, rotation=0, epsg_code=None, scaling=None,
                 reference_elevation=0.0):
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

    def convert_elevation_depth(self, z, to_depth=True):
        """
        Convert between elevation and depth using a reference elevation or depth and
        scaling.

        :param z: Input elevation or depth.
        :type z: float
        :param to_depth: If True, convert elevation to depth. If False, convert depth to
        elevation.
        :type to_depth: bool
        :return: Converted depth or elevation.
        :rtype: float
        """
        if to_depth:
            return (self.reference_elevation - z) * self.scaling
        else:
            return self.reference_elevation - (z / self.scaling)

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


class Coordinates:
    def __init__(self, x: float, y: float, z: float,
                 coordinate_system: CoordinateSystem = CoordinateSystem('NED'),
                 transformation: CoordinateTransformation = CoordinateTransformation()):
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
        return self.y if self.coordinate_system == CoordinateSystem.NED else self.x

    @property
    def easting(self):
        """
        Get the easting coordinate.

        :return: Easting coordinate based on system.
        :rtype: float
        """
        return self.x if self.coordinate_system == CoordinateSystem.NED else self.y

    @property
    def up(self):
        """
        Get the up coordinate.

        :return: Up coordinate based on system.
        :rtype: float
        """
        if self.transformation:
            return self.transformation.convert_elevation_depth(self.z, to_depth=False)
        return -self.z if self.coordinate_system == CoordinateSystem.NED else self.z

    @property
    def down(self):
        """
        Get the down coordinate.

        :return: Down coordinate based on system.
        :rtype: float
        """
        if self.transformation:
            return self.transformation.convert_elevation_depth(self.z, to_depth=True)
        return self.z if self.coordinate_system == CoordinateSystem.NED else -self.z

    def __repr__(self):
        out_str = f"""
                        x: {self.x}
                        y: {self.y}
                        z: {self.z}
        coordinate system: {self.coordinate_system}
           transformation: {self.transformation}
        """
        return out_str

    def __eq__(self, other):
        if not isinstance(other, Coordinates):
            return False
        return True if ((self.x == other.x) & (self.y == other.y) & (self.z == other.z) &
                        (self.coordinate_system == other.coordinate_system) &
                        (self.transformation == other.transformation)) else False

    def to_json(self):
        dict = {}
        for key in self.__dict__.keys():
            # from ipdb import set_trace
            # set_trace()
            if key == 'coordinate_system':
                dict[key] = str(self.coordinate_system)
            elif key == 'transformation':
                if self.transformation is not None:
                    dict[key] = self.transformation.to_json()
                else:
                    dict[key] = None
            else:
                dict[key] = self.__dict__[key]
        return json.dumps(dict)

    @classmethod
    def from_json(cls, json_string):
        dict = json.loads(json_string)
        coordinate_system = CoordinateSystem(dict['coordinate_system'])

        if dict['transformation'] is not None:
            transformation = CoordinateTransformation.from_json(dict['transformation'])
        else:
            transformation = None

        return cls(dict['x'], dict['y'], dict['z'], coordinate_system=coordinate_system,
                   transformation=transformation)

    def to_extra_key(self, namespace='mq'):
        return AttribDict({'value': self.to_json(), 'namespace': namespace})

    @classmethod
    def from_extra_key(cls, extra_key):
        return cls.from_json(extra_key['value'])
