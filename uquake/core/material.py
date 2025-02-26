from .inventory import (Channel, Station, InstrumentSensitivity, Equipment, Response,
                        ResponseStage, PolesZerosResponseStage, CoefficientsTypeResponseStage)
from obspy.signal.invsim import corn_freq_2_paz
from .coordinates import Coordinates, CoordinateSystem, rotate_azimuth
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal
from datetime import datetime
from uquake.core import UTCDateTime
import numpy as np
import pandas as pd



units_descriptions = {
    'V': 'Volts',
    'M/S': 'Velocity in meters per second',
    'M/S/S': 'Acceleration in meters per second squared',
    'COUNTS': 'ADC Counts'
}

class GenericSensor(BaseModel):
    """
    Represents a generic seismic sensor, such as a geophone or an accelerometer.

    This class defines the fundamental parameters of a seismic sensor, including
    sensitivity, frequency response, and gain. It also provides methods to compute
    instrument sensitivity and response stages using pole-zero representation.

    Attributes
    ----------
    sensor_type : str
        The type of sensor (e.g., 'geophone', 'accelerometer').
    model : str
        The name or model of the sensor.
    input_units : Literal['V', 'M/S', 'M/S/S', 'COUNTS']
        The input measurement unit of the sensor.
    output_units : Literal['V', 'M/S', 'M/S/S', 'COUNTS']
        The output measurement unit of the sensor.
    gain : float
        The overall system gain of the sensor.
    natural_frequency : float
        The natural (or corner) frequency of the sensor in Hz.
    sensitivity : float
        The sensitivity of the sensor, typically given in output units per input unit.
    damping : float
        The damping factor of the sensor.
    stage_sequence_number : int, optional
        The processing stage number in a signal chain, defaulting to 0.

    Properties
    ----------
    instrument_sensitivity : InstrumentSensitivity
        Returns an `InstrumentSensitivity` object representing the sensor’s
        sensitivity at its natural frequency.
    response_stage : PolesZerosResponseStage
        Returns a `PolesZerosResponseStage` object defining the sensor's
        response using a pole-zero representation.

    Notes
    -----
    - The `instrument_sensitivity` property encapsulates the sensor's sensitivity
      along with its input/output units and natural frequency.
    - The `response_stage` property computes the pole-zero representation
      based on the sensor’s damping and natural frequency.
    - This class ensures consistency in sensitivity and response calculations
      across different sensor types.
    """

    sensor_type: str
    model: str
    input_units: Literal['V', 'M/S', 'M/S/S', 'COUNTS']
    output_units: Literal['V', 'M/S', 'M/S/S', 'COUNTS']
    natural_frequency: float
    sensitivity: float
    damping: float
    stage_sequence_number: int = 0
    gain: float = 1

    @property
    def instrument_sensitivity(self) -> InstrumentSensitivity:
        """
        Computes the sensor's instrument sensitivity.

        Returns
        -------
        InstrumentSensitivity
            The sensitivity object containing value, frequency, and units.
        """
        return InstrumentSensitivity(
            value=self.sensitivity,
            frequency=self.natural_frequency,
            input_units=self.input_units,
            output_units=self.output_units
        )

    @property
    def response_stage(self) -> PolesZerosResponseStage:
        """
        Computes the sensor's response stage using pole-zero representation.

        Returns
        -------
        PolesZerosResponseStage
            The response stage containing poles and zeros based on damping
            and natural frequency.
        """
        paz = corn_freq_2_paz(self.natural_frequency, damp=self.damping)

        return PolesZerosResponseStage(
            self.stage_sequence_number,
            self.gain,
            self.natural_frequency,
            self.input_units,
            self.output_units,
            'LAPLACE (RADIANT/SECOND)',
            self.natural_frequency,
            paz['zeros'],
            paz['poles'],
            name=self.model,
            input_units_description=units_descriptions[self.input_units],
            output_units_description=units_descriptions[self.output_units]
        )


class Geophone(GenericSensor):
    """
    Represents the configuration parameters for a geophone sensor.

    This class extends `GenericSensor`, predefining values specific to geophones,
    such as input and output units.
    """

    sensor_type: Literal['geophone'] = 'geophone'  # ✅ Correctly annotated
    input_units: Literal['V'] = 'V'  # ✅ Correctly annotated
    output_units: Literal['M/S'] = 'M/S'  # ✅ Correctly annotated
    gain: float = 1
    stage_sequence_number: int = 0

    def __init__(self, model: str, sensitivity: float, damping: float, natural_frequency: float):
        super().__init__(
            sensor_type=self.sensor_type,
            model=model,
            input_units=self.input_units,
            output_units=self.output_units,
            natural_frequency=natural_frequency,
            sensitivity=sensitivity,
            damping=damping,
            stage_sequence_number=self.stage_sequence_number,
            gain=self.gain
        )


class Accelerometer(GenericSensor):
    """
    Represents the configuration parameters for an accelerometer sensor.

    This class extends `GenericSensor`, predefining values specific to accelerometers,
    such as input and output units.
    """

    sensor_type: Literal['accelerometer'] = 'accelerometer'  # ✅ Correctly annotated
    input_units: Literal['V'] = 'V'  # ✅ Correctly annotated
    output_units: Literal['M/S/S'] = 'M/S/S'  # ✅ Correctly annotated
    gain: float = 1
    stage_sequence_number: int = 0

    def __init__(self, model: str, sensitivity: float, natural_frequency: float, damping=0.707):
        super().__init__(
            sensor_type=self.sensor_type,
            model=model,
            input_units=self.input_units,
            output_units=self.output_units,
            natural_frequency=natural_frequency,
            sensitivity=sensitivity,
            damping=damping,  # Default damping for accelerometers
            stage_sequence_number=self.stage_sequence_number,
            gain=self.gain
        )


class Digitizer(BaseModel):
    """
    Represents the configuration and response characteristics of a seismic digitizer.

    The digitizer converts an analog signal (voltage) from a sensor (e.g., geophone,
    accelerometer) into digital ADC counts. This class defines its essential
    parameters, including gain and sampling rate.

    Attributes
    ----------
    gain : float
        The gain of the digitizer, representing the scaling factor between
        input voltage (V) and ADC counts.
    sampling_rate : float
        The sampling rate of the digitizer in Hz.
    model : str
        The name or model of the digitizer.
    input_units : Literal['V']
        Fixed as 'V' (Volts), representing the input unit from analog sensors.
    output_units : Literal['COUNTS']
        Fixed as 'COUNTS' (ADC Counts), representing the digitized output.
    stage_sequence_number : int
        Fixed at 1, representing the digitizer as the next processing stage after
        the sensor.

    Properties
    ----------
    stage_response : CoefficientsTypeResponseStage
        Returns a `CoefficientsTypeResponseStage` object representing the digitizer's
        gain response.

    Notes
    -----
    - The digitizer’s response is modeled using a simple gain factor.
    - Some digitizers may include anti-aliasing filters, which are not explicitly
      modeled here.
    - This class ensures consistency with ObsPy’s response structure, allowing
      integration with seismic metadata.
    """
    model: str = "generic digitizer"
    gain: float
    sampling_rate: float
    input_units: Literal["V"] = "V"
    output_units: Literal["COUNTS"] = "COUNTS"
    stage_sequence_number: int = 1

    @property
    def stage_response(self):
        """
        Returns the digitizer's response stage using a simple gain factor.

        This response is modeled as a coefficient response stage, as digitizers
        generally apply a linear transformation from voltage to ADC counts.

        Returns
        -------
        CoefficientsTypeResponseStage
            A `CoefficientsTypeResponseStage` object defining the digitizer's gain
            and input/output unit mappings.
        """
        return CoefficientsTypeResponseStage(
            stage_sequence_number=self.stage_sequence_number,
            stage_gain=self.gain,
            stage_gain_frequency=1.0,  # Assuming gain is defined at 1 Hz
            input_units=self.input_units,
            output_units=self.output_units,
            cf_transfer_function_type="DIGITAL",
            numerator=[self.gain],
            denominator=[1.0],
            name=self.model,
            description="Digitizer gain stage"
        )



class Cable(BaseModel):
    """
    Represents the frequency response characteristics of a sensor cable.

    This class models the effect of cable capacitance and resistance on the
    signal transmission. The cable can introduce a low-pass filtering effect,
    represented by a single pole in the response.

    Parameters
    ----------
    output_resistance : float, optional
        The output resistance of the connected sensor (Ohms). Defaults to infinity.
    cable_length : float, optional
        The length of the cable (meters). Defaults to 0 (no cable effect).
    cable_capacitance : float, optional
        The capacitance of the cable per unit length (Farads/m). Defaults to infinity.

    Attributes
    ----------
    stage_sequence_number : int
        Defines the sequence number of this response stage in signal processing.
        Defaults to 1 (typically after the sensor).
    input_units : str
        Fixed as 'V' (Volts), representing the voltage signal input.
    input_units_description : str
        Fixed as 'Volts' for clarity.
    output_units : str
        Fixed as 'V' (Volts), as cables do not change unit types.
    output_units_description : str
        Fixed as 'Volts'.

    Properties
    ----------
    poles : list
        Returns a list of poles representing the cable's frequency response.
    response_stage : PolesZerosResponseStage
        Returns a `PolesZerosResponseStage` object modeling the cable’s response.

    Notes
    -----
    - If `output_resistance * cable_length * cable_capacitance` is finite and nonzero,
      the cable introduces a **low-pass filter effect**, modeled as a single pole.
    - If any parameter is **infinite or zero**, the cable is treated as **ideal** (no filtering).
    """

    output_resistance: float = np.inf
    cable_length: float = 0.0
    cable_capacitance: float = np.inf
    stage_sequence_number: int = 1

    input_units: str = "V"
    input_units_description: str = "Volts"
    output_units: str = "V"
    output_units_description: str = "Volts"

    @property
    def poles(self):
        """
        Computes the cable's pole based on resistance, capacitance, and length.

        Returns
        -------
        list
            A list containing the computed pole, or an empty list if no pole is needed.
        """
        if (self.output_resistance * self.cable_length * self.cable_capacitance) not in [0, np.inf]:
            pole_cable = -1 / (self.output_resistance * self.cable_length * self.cable_capacitance)
            return [pole_cable]
        return []

    @property
    def response_stage(self):
        """
        Returns the `PolesZerosResponseStage` defining the cable's response.

        Returns
        -------
        PolesZerosResponseStage
            A response stage modeling the cable's effect as a low-pass filter.
        """
        return PolesZerosResponseStage(
            stage_sequence_number=self.stage_sequence_number,
            stage_gain=1,  # Cables typically do not amplify signals
            stage_gain_frequency=0,  # No reference frequency
            input_units=self.input_units,
            output_units=self.output_units,
            pz_transfer_function_type="LAPLACE (RADIANT/SECOND)",
            normalization_frequency=0,
            zeros=[],  # Cables do not introduce zeros
            poles=self.poles,
            input_units_description=self.input_units_description,
            output_units_description=self.output_units_description
        )


class ComponentType(BaseModel):
    """
    Represents a component, including a sensor, an
    optional cable, and an optional digitizer.

    This class models the overall system response by combining the responses
    of the individual components into a multi-stage response.

    Parameters
    ----------
    sensor : Union[GenericSensor, Geophone, Accelerometer]
        The primary sensor in the system.
    cable : Optional[Cable], optional
        An optional cable connecting the sensor to the digitizer.
    digitizer : Optional[Digitizer], optional
        An optional digitizer converting analog signals to digital counts.
    description : Optional[str], optional
        Additional descriptive details about the system.
    manufacturer : Optional[str], optional
        The company that manufactured the system.
    vendor : Optional[str], optional
        The vendor or distributor of the system.
    model : Optional[str], optional
        The specific model identifier of the system.

    Properties
    ----------
    response : Response
        Returns an uQuake `Response` object representing the full system response.
    equipment : Equipment
        Returns an uQuake `Equipment` object containing system metadata.

    Notes
    -----
    - The response is dynamically built based on available components.
    - If a cable is included, its response is added before the digitizer.
    - The `response` property ensures seamless integration with ObsPy’s response handling.
    """

    sensor: Union[GenericSensor, Geophone, Accelerometer]
    cable: Optional[Cable] = None
    type: str
    digitizer: Optional[Digitizer] = None
    description: Optional[str] = None
    manufacturer: Optional[str] = None
    vendor: Optional[str] = None
    model: Optional[str] = None

    @property
    def response_stages(self) -> List[ResponseStage]:
        """
        Builds and returns the list of response stages for the system.

        The response chain follows this sequence:
        1. Sensor (Required)
        2. Cable (Optional)
        3. Digitizer (Optional)

        Returns
        -------
        List[ResponseStage]
            A list of `ResponseStage` objects representing the system response.
        """
        self.sensor.stage_sequence_number = 0
        stages = [self.sensor.response_stage]  # Sensor is always required
        if self.cable:
            self.cable.stage_sequence_number = 1
            stages.append(self.cable.response_stage)  # Add cable if present

        if self.digitizer:
            if self.cable:
                self.digitizer.stage_sequence_number = 2
            else:
                self.digitizer.stage_sequence_number = 1
            stages.append(self.digitizer.stage_response)  # Add digitizer if present

        return stages

    @property
    def system_sensitivity(self) -> InstrumentSensitivity:
        """
        Computes the overall system sensitivity.

        If a digitizer is present, the system sensitivity is the combination
        of the sensor and digitizer gains. Otherwise, it defaults to the sensor's sensitivity.

        Returns
        -------
        InstrumentSensitivity
            The overall system sensitivity.
        """
        sensor_sensitivity = self.sensor.instrument_sensitivity

        if self.digitizer:
            combined_sensitivity = sensor_sensitivity.value * self.digitizer.gain
            return InstrumentSensitivity(
                value=combined_sensitivity,
                frequency=sensor_sensitivity.frequency,
                input_units=self.sensor.output_units,
                output_units=self.digitizer.output_units
            )

        return sensor_sensitivity  # If no digitizer, sensor sensitivity is used

    @property
    def response(self) -> Response:
        """
        Returns the full system response, including all response stages.

        Returns
        -------
        Response
            An ObsPy `Response` object containing the system sensitivity
            and response stages.
        """
        return Response(
            instrument_sensitivity=self.system_sensitivity,
            response_stages=self.response_stages
        )

    def to_channel(self, channel_code, location_code, orientation_vector, sample_rate, coordinates,
                   start_date: Union[datetime, UTCDateTime, str] = None,
                   end_date: Union[datetime, UTCDateTime, str] = None,
                   equipment: Equipment = None):

        if isinstance(start_date, datetime) or isinstance(start_date, str):
            start_date = UTCDateTime(start_date)

        if isinstance(end_date, datetime) or isinstance(end_date, str):
            end_date = UTCDateTime(end_date)

        return Channel(
            code=channel_code,
            location_code=location_code,
            oriented=True,
            orientation_vector=orientation_vector,
            response=self.response,
            start_date=start_date,
            end_date=end_date,
            sample_rate=sample_rate,
            calibration_units=self.sensor.output_units,
            sensor=equipment,
            coordinates=coordinates
        )


class Component(BaseModel):
    """
    Represents an individual seismic component within a system.

    Attributes:
        orientation_vector (List[float]): A 3D vector representing the orientation of the component in space.
        coordinate_system (CoordinateSystem): The coordinate system of the component.
        channel_code (str): The SEED channel code associated with this component.
        name (str): Optional name for the component.
        component_type (ComponentType): Type of the component.
    """

    orientation_vector: List[float]
    coordinate_system: CoordinateSystem
    channel_code: str
    name: str = None
    component_type: ComponentType

    def rotate_azimuth(self, azimuth: float) -> 'Component':
        """
        Rotates the component's orientation vector around the vertical axis by a given azimuth angle.

        Args:
            azimuth (float): The azimuth angle in degrees.

        Returns:
            Component: A new rotated component.
        """
        x, y, z = self.orientation_vector
        new_x, new_y = rotate_azimuth(x, y, self.coordinate_system, azimuth)

        return Component(
            orientation_vector=[new_x, new_y, z],
            coordinate_system=self.coordinate_system,
            channel_code=self.channel_code,
            name=self.name,
            component_type=self.component_type
        )

    def change_coordinate_system(self,
                                 new_coordinate_system: CoordinateSystem) -> 'Component':
        """
        Converts the component's orientation vector to a new coordinate system.

        Args:
            new_coordinate_system (CoordinateSystem): The target coordinate system.

        Returns:
            Component: A new component in the target coordinate system.
        """
        new_orientation_vector = CoordinateSystem.transform_coordinates(
            self.coordinate_system, new_coordinate_system,
            self.orientation_vector[0], self.orientation_vector[1],
            self.orientation_vector[2]
        )

        return Component(
            orientation_vector=new_orientation_vector,
            coordinate_system=new_coordinate_system,
            channel_code=self.channel_code,
            name=self.name,
            component_type=self.component_type
        )


class Components(BaseModel):
    """
    Represents a collection of seismic components with an associated coordinate system.

    Attributes:
        components (List[Component]): A list of component objects.
        coordinate_system (CoordinateSystem): The coordinate system in which the components are defined.
    """

    components: List[Component] = Field(default_factory=list)

    def __iter__(self):
        return iter(self.components)

    def __getitem__(self, item):
        return self.components[item]

    def __len__(self):
        """Returns the number of components."""
        return len(self.components)

    def __contains__(self, component: Component):
        """Checks if a component exists in the components list."""
        return component in self.components

    def __repr__(self):
        """Provides a string representation of the object."""
        return (f"Components({len(self.components)} components")

    def append(self, component: Component):
        """Appends a new component to the list."""
        self.components.append(component)

    def change_coordinate_system(self, new_coordinate_system: CoordinateSystem) -> None:
        """
        Converts all component orientation vectors to a new coordinate system.

        Args:
            new_coordinate_system (CoordinateSystem): The target coordinate system.

        Returns:
            None
        """
        self.components = [component.change_coordinate_system(new_coordinate_system) for
                           component in self.components]

    def rotate_azimuth(self, azimuth: float) -> None:
        """
        Rotates all component orientation vectors around the vertical axis by a given azimuth angle.

        Args:
            azimuth (float): The azimuth angle (in degrees) to rotate the components.

        Returns:
            None
        """
        self.components = [component.rotate_azimuth(azimuth) for component in
                           self.components]


class DeviceType(BaseModel):
    """
    Represents the general characteristics of a seismic device.

    A `DeviceType` defines the blueprint for a seismic device, including the
    list of components and the coordinate system used for positioning.

    Attributes
    ----------
    name : str
        The name of the seismic device.
    type : str
        The type of device (e.g., 'nodal geophone', 'accelerometer').
    components : List[Component]
        A list of components included in the device.
    coordinate_system : CoordinateSystem
        The coordinate system used for defining locations. Defaults to NED (North-East-Down).
    """

    name: str
    type: str
    manufacturer: str = None,
    vendor: str = None,
    model: str = None,
    components: Components
    coordinate_system: CoordinateSystem = CoordinateSystem.NED

    def __init__(self, **data):
        super().__init__(**data)
        self.components.change_coordinate_system(self.coordinate_system)

    def rotate_components_azimuth(self, azimuth: float) -> None:
        """
        Rotates all component orientation vectors around the vertical axis by a given azimuth angle.

        Args:
            azimuth (float): The azimuth angle (in degrees) to rotate the components.

        Returns:
            None
        """
        self.components.rotate_azimuth(azimuth)

    def change_components_coordinate_system(self, new_coordinate_system: CoordinateSystem) -> None:
        """
        Converts all component orientation vectors to a new coordinate system.

        Args:
            new_coordinate_system (CoordinateSystem): The target coordinate system.

        Returns:
            None
        """
        self.components.change_coordinate_system(new_coordinate_system)


class Device(BaseModel):
    """
    Represents a physical seismic device.

    A `Device` extends `DeviceType` by adding unique identifiers such as
    serial number and manufacturing/calibration dates. It also provides
    functionality to convert itself into an ObsPy `Station`.

    Attributes
    ----------
    device_type : DeviceType
        The type of device.
    serial_number : str
        The serial number of the device.
    calibration_dates : Optional[List[Union[str, datetime, UTCDateTime]]]
        The date when the device was last calibrated.
    manufactured_date : Union[str, datetime, UTCDateTime]
        The date when the device was manufactured.

    Methods
    -------
    to_station(station_code, location_code, coordinates, sampling_rate, start_date, end_date)
        Converts this device into an ObsPy `Station` object.
    to_station_lat_long(station_code, location_code, latitude, longitude, elevation, sampling_rate, start_date, end_date)
        Converts this device into an ObsPy `Station` using latitude/longitude coordinates.
    """

    device_type: DeviceType
    serial_number: str
    calibration_dates: List[Union[str, datetime, UTCDateTime]] = None
    manufactured_date: Union[str, datetime, UTCDateTime] = None

    class Config:
        arbitrary_types_allowed = True

    if manufactured_date is not None:
        if isinstance(manufactured_date, datetime) or isinstance(manufactured_date, str):
            end_date = UTCDateTime(manufactured_date)

    @property
    def device_id(self):
        return self.serial_number

    @property
    def components(self):
        return self.device_type.components

    @property
    def coordinate_system(self):
        return self.device_type.coordinate_system

    def to_station(self, station_code: str, location_code: str, coordinates: Coordinates,
                   sampling_rate: float, azimuth: float = 0, tilt: float = 0,
                   installation_date: Union[datetime, UTCDateTime, str] = None,
                   removal_date: Union[datetime, UTCDateTime, str] = None):
        """
        Converts the device into an ObsPy `Station` object.

        Parameters
        ----------
        station_code : str
            The SEED station code.
        location_code : str
            The SEED location code.
        coordinates : Coordinates
            The geographical or relative coordinates of the station.
        sampling_rate : float
            The sampling rate of the device in Hz.
        azimuth : float
            The azimuth of the device north axis in degrees (0 if it is oriented north).
        tilt : float
            The tilt of the device in degrees (0 if it is leveled). This should be 0
            for most devices except if the devices is equipped with omni-directional
            sensors.
        installation_date : Union[datetime, UTCDateTime, str], optional
            The start date of the station's validity period.
        removal_date : Union[datetime, UTCDateTime, str], optional
            The end date of the station's validity period.

        Returns
        -------
        Station
            An ObsPy `Station` object representing this seismic device.
        """

        if isinstance(installation_date, datetime) or isinstance(installation_date, str):
            installation_date = UTCDateTime(installation_date)
        if isinstance(removal_date, datetime) or isinstance(removal_date, str):
            removal_date = UTCDateTime(removal_date)

        # correct for the device azimuth
        azimuth = azimuth % 360
        if self.device.coordinate_system != coordinates.coordinate_system:
            self.change_components_coordinate_system(self.coordinates.coordinate_system)
        self.device_type.rotate_components_azimuth(azimuth)

        channels = []
        for component in self.device_type.components:

            channels.append(component.to_channel(
                location_code=location_code,
                sampling_rate=sampling_rate,
                coordinates=coordinates,
                start_date=installation_date,
                end_date=removal_date,
                equipment=self.equipment(installation_date, removal_date)
            ))

        return Station(
            code=station_code,
            start_date=installation_date,
            end_date=removal_date,
            coordinates=coordinates,
            alternate_code=self.serial_number,
            latitude=coordinates.latitude,
            longitude=coordinates.longitude,
            elevation=coordinates.elevation,
        )

    def equipment(self, installation_date=None, removal_date=None) -> Equipment:
        """
        Returns an `Equipment` object containing metadata about the system.

        Returns
        -------
        Equipment
            An ObsPy `Equipment` object with details such as manufacturer, model,
            and calibration date.
        """
        return Equipment(
            type=self.device_type.type,
            manufacturer=self.device_type.manufacturer,
            vendor=self.device_type.vendor,
            model=self.device_type.model,
            serial_number=self.serial_number,
            calibration_dates=self.calibration_dates,
            installation_date=installation_date,
            removal_date=removal_date
        )
