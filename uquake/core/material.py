from .inventory import (Channel, Station, InstrumentSensitivity, Equipment, Response,
                        ResponseStage, PolesZerosResponseStage, CoefficientsTypeResponseStage)
from obspy.signal.invsim import corn_freq_2_paz
from .coordinates import Coordinates, CoordinateSystem
from pydantic import BaseModel
from typing import List, Optional, Union, Literal
from datetime import datetime
from uquake.core import UTCDateTime
import numpy as np



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
                   serial_number=None, calibration_date=None,
                   start_date: Union[datetime, UTCDateTime, str] = None,
                   end_date: Union[datetime, UTCDateTime, str] = None):

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
            sensor=self.equipment(serial_number, calibration_date),
            coordinates=coordinates
        )

    def equipment(self, serial_number, calibration_date) -> Equipment:
        """
        Returns an `Equipment` object containing metadata about the system.

        Returns
        -------
        Equipment
            An ObsPy `Equipment` object with details such as manufacturer, model,
            and calibration date.
        """
        return Equipment(
            type=self.type,
            manufacturer=self.manufacturer,
            vendor=self.vendor,
            model=self.model,
            serial_number=serial_number,
            calibration_date=calibration_date,
        )

class Component(BaseModel):
    """
    Represents an individual seismic component within a system.

    A `Component` is a subclass of `ComponentType` that includes additional
    metadata such as orientation and channel code. It provides functionality
    for converting itself into an ObsPy `Channel` object.

    Attributes
    ----------
    orientation_vector : List[float]
        A 3D vector representing the orientation of the component in space.
    channel_code : str
        The SEED channel code associated with this component.

    Methods
    -------
    to_channel(location_code, sampling_rate, coordinates, serial_number, calibration_date, start_date, end_date)
        Converts this component into an ObsPy `Channel` object.
    """

    orientation_vector: List[float]
    channel_code: str
    name: str = None
    component_type: ComponentType

    def to_channel(self, location_code, sampling_rate, coordinates, serial_number=None,
                   calibration_date=None,
                   start_date: Union[datetime, UTCDateTime, str] = None,
                   end_date: Union[datetime, UTCDateTime, str] = None):
        """
        Converts the component into an ObsPy `Channel` object.

        Parameters
        ----------
        location_code : str
            The SEED location code for the component.
        sampling_rate : float
            The sampling rate of the component in Hz.
        coordinates : Coordinates
            The geographical or relative coordinates of the component.
        serial_number : str, optional
            The serial number of the component.
        calibration_date : str, optional
            The last calibration date of the component.
        start_date : Union[datetime, UTCDateTime, str], optional
            The start date of the channel's validity period.
        end_date : Union[datetime, UTCDateTime, str], optional
            The end date of the channel's validity period.

        Returns
        -------
        Channel
            An ObsPy `Channel` object representing this seismic component.
        """
        return self.component_type.to_channel(
            channel_code=self.channel_code,
            location_code=location_code,
            orientation_vector=self.orientation_vector,
            sample_rate=sampling_rate,
            coordinates=coordinates,
            serial_number=serial_number,
            calibration_date=calibration_date,
            start_date=start_date,
            end_date=end_date,
        )


class DeviceType(BaseModel):
    """
    Represents the general characteristics of a seismic device.

    A `DeviceType` defines the blueprint for a seismic device, including the
    list of components and the coordinate system used for positioning.

    Attributes
    ----------
    name : str
        The name of the seismic device.
    components : List[Component]
        A list of components included in the device.
    coordinate_system : CoordinateSystem
        The coordinate system used for defining locations. Defaults to NED (North-East-Down).
    """

    name: str
    components: List[Component]
    coordinate_system: CoordinateSystem = CoordinateSystem.NED


class Device(BaseModel):
    """
    Represents a physical seismic device.

    A `Device` extends `DeviceType` by adding unique identifiers such as
    serial number and manufacturing/calibration dates. It also provides
    functionality to convert itself into an ObsPy `Station`.

    Attributes
    ----------
    DeviceType : DeviceType
        The type of device.
    serial_number : str
        The serial number of the device.
    calibration_date : str
        The date when the device was last calibrated.
    manufactured_date : str
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
    calibration_date: str
    manufactured_date: str

    def to_station(self, station_code, location_code, coordinates: Coordinates, sampling_rate,
                   start_date: Union[datetime, UTCDateTime, str] = None,
                   end_date: Union[datetime, UTCDateTime, str] = None):
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
        start_date : Union[datetime, UTCDateTime, str], optional
            The start date of the station's validity period.
        end_date : Union[datetime, UTCDateTime, str], optional
            The end date of the station's validity period.

        Returns
        -------
        Station
            An ObsPy `Station` object representing this seismic device.
        """
        channels = []
        for component in self.component_list:
            channels.append(component.to_channel(
                location_code=location_code,
                sampling_rate=sampling_rate,
                coordinates=coordinates,
                serial_number=self.serial_number,
                calibration_date=self.calibration_date,
                start_date=start_date,
                end_date=end_date
            ))

        return Station(
            code=station_code,
            start_date=start_date,
            end_date=end_date,
            coordinates=coordinates,
        )

    def to_station_lat_long(self, station_code, location_code, latitude, longitude, elevation, sampling_rate,
                            start_date: Union[datetime, UTCDateTime, str] = None,
                            end_date: Union[datetime, UTCDateTime, str] = None):
        """
        Converts the device into an ObsPy `Station` using latitude and longitude.

        This method is useful when defining a station's location using GPS
        coordinates instead of a pre-defined `Coordinates` object.

        Parameters
        ----------
        station_code : str
            The SEED station code.
        location_code : str
            The SEED location code.
        latitude : float
            The latitude of the station in degrees.
        longitude : float
            The longitude of the station in degrees.
        elevation : float
            The elevation of the station in meters.
        sampling_rate : float
            The sampling rate of the device in Hz.
        start_date : Union[datetime, UTCDateTime, str], optional
            The start date of the station's validity period.
        end_date : Union[datetime, UTCDateTime, str], optional
            The end date of the station's validity period.

        Returns
        -------
        Station
            An ObsPy `Station` object representing this seismic device.
        """
        coordinates = Coordinates.from_lat_lon(latitude, longitude, elevation, self.coordinate_system)

        return self.to_station(
            station_code=station_code,
            location_code=location_code,
            coordinates=coordinates,
            sampling_rate=sampling_rate,
            start_date=start_date,
            end_date=end_date
        )
