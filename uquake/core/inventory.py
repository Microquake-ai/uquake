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

# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: inventory.py
#  Purpose: Expansion of the obspy.core.inventory.inventory module
#   Author: uquake development team
#    Email: devs@uquake.org
#
# Copyright (C) 2016 uquake development team
# --------------------------------------------------------------------
"""
Expansion of the obspy.core.event module

:copyright:
    uquake development team (devs@uquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import io
from obspy.core import inventory, AttribDict, UTCDateTime
from obspy.core.inventory import Network
from obspy.signal.invsim import corn_freq_2_paz
from obspy.core.inventory import (Response, InstrumentSensitivity,
                                  PolesZerosResponseStage, ResponseStage)
import numpy as np
from obspy.core.inventory.util import (Equipment, Operator, Person,
                                       PhoneNumber, Site, _textwrap,
                                       _unified_content_strings)
from uquake.core.util.decorators import expand_input_format_compatibility
from uquake.core.coordinates import Coordinates, CoordinateSystem
from pathlib import Path

from .logging import logger
from uquake import __package_name__ as ns

import pandas as pd
from io import BytesIO
from .util.tools import lon_lat_x_y

from .util import ENTRY_POINTS
from pkg_resources import load_entry_point
from tempfile import NamedTemporaryFile
import os
from .util.requests import download_file_from_url
from uquake.core.util.attribute_handler import set_extra, get_extra, namespace
import hashlib
from enum import Enum
from pydantic import BaseModel
from typing import Union, Litteral, Optional


class GenericSensor(BaseModel):
    """
    Represents the configuration parameters for a generic seismic sensor,
    including geophones and accelerometers.

    Attributes
    ----------
    sensor_type : str
        The type of sensor (e.g., 'geophone', 'accelerometer').
    sensor_name : str
        The name or model of the sensor.
    input_units : Literal['V', 'M/S', 'M/S/S', 'COUNTS']
        The input measurement unit of the sensor.
    input_units_description : Literal[
        'Volts', 'Velocity in meters per second',
        'Acceleration in meters per second squared', 'ADC Counts'
    ]
        A descriptive label for the input units.
    output_units : Literal['V', 'M/S', 'M/S/S', 'COUNTS']
        The output measurement unit of the sensor.
    output_units_description : Literal[
        'Volts', 'Velocity in meters per second',
        'Acceleration in meters per second squared', 'ADC Counts'
    ]
        A descriptive label for the output units.
    gain : float
        The overall system gain of the sensor.
    natural_frequency : float
        The natural (or natural) frequency of the sensor in Hz.
    sensitivity : float
        The sensitivity of the sensor, typically given in the ratio of
        output units per input units.
    damping : float
        The damping factor of the sensor.
    stage_sequence_number : int, optional
        The processing stage number in a signal chain, defaulting to 0.

    Properties
    ----------
    sensitivity : InstrumentSensitivity
        Returns an `InstrumentSensitivity` object that represents the sensor’s
        sensitivity at its natural frequency.
    response_stage : PolesZerosResponseStage
        Returns a `PolesZerosResponseStage` object, which defines the sensor's
        response using pole-zero representation.

    Notes
    -----
    - The `sensitivity` property encapsulates the sensor's sensitivity along with
      its input/output units and natural frequency.
    - The `response_stage` property calculates the pole-zero representation based
      on the sensor’s damping and natural frequency.
    - This class provides a structured way to define various seismic sensor configurations
      while ensuring consistency in sensitivity and response calculations.
    """
    sensor_type: str
    sensor_name: str
    input_units: Litteral['V', 'M/S', 'M/S/S', 'COUNTS']
    input_units_description: Litteral[
        'Volts', 'Velocity in meters per second',
        'Acceleration in meters per second squared', 'ADC Counts'
    ]
    output_units: Litteral['V', 'M/S', 'M/S/S', 'COUNTS']
    output_units_description: Litteral[
        'Volts', 'Velocity in meters per second',
        'Acceleration in meters per second squared', 'ADC Counts'
    ]
    gain: float
    natural_frequency: float
    sensitivity: float
    damping: float
    stage_sequence_number: int = 0

    @property
    def sensitivity(self):
        return InstrumentSensitivity(
            value=self.sensitivity,
            frequency=self.natural_frequency,
            input_units=self.input_units,
            output_units=self.output_units
        )

    @property
    def response_stage(self):
        paz = corn_freq_2_paz(self.natural_frequency, damp=self.damping)
        pzr = PolesZerosResponseStage(
            self.stage_sequence_number,
            self.gain,
            self.natural_frequency,
            self.input_units,
            self.output_units,
            'LAPLACE (RADIANT/SECOND)',
            self.natural_frequency,
            paz['zeros'],
            paz['poles'],
            name=self.sensor_name,
            input_units_description=self.input_units_description,
            output_units_description=self.output_units_description
        )

        return pzr


class Geophone(GenericSensorConfig):
    """
    Represents the configuration parameters for a geophone sensor.

    This class extends `GenericSensorConfig`, predefining values specific
    to geophones, such as input and output units.

    Parameters
    ----------
    sensitivity : float
        The sensitivity of the geophone, typically expressed in output units
        per input units (e.g., m/s per V).
    damping : float
        The damping factor of the geophone.
    sensor_name : str, optional
        The name or model of the geophone.

    Attributes
    ----------
    sensor_type : str
        Fixed as 'geophone'.
    input_units : Literal['V']
        Fixed as 'V' (Volts) since geophones produce a voltage output.
    input_units_description : str
        Fixed as 'Volts' for clarity.
    output_units : Literal['M/S']
        Fixed as 'M/S' (meters per second), representing velocity.
    output_units_description : str
        Fixed as 'Velocity in meters per second'.
    gain : float
        Fixed at 1, assuming unitary gain in processing.
    sensitivity : float
        The specified sensitivity of the geophone.
    damping : float
        The damping factor of the geophone.
    stage_sequence_number : int
        Fixed at 0, representing the initial processing stage.

    Notes
    -----
    - This class ensures geophone-specific configurations are set automatically.
    - The `sensor_type` is hardcoded as 'geophone' to distinguish it from
      other sensor types.
    """

    def __init__(self, sensitivity: float, damping: float, sensor_name: str = None):
        super().__init__(
            sensor_type='geophone',
            sensor_name=sensor_name,
            input_units='V',
            input_units_description='Volts',
            output_units='M/S',
            output_units_description='Velocity in meters per second',
            gain=1,
            sensitivity=sensitivity,
            damping=damping,
            stage_sequence_number=0
        )


class Accelerometer(GenericSensorConfig):
    """
    Represents the configuration parameters for an accelerometer sensor.

    This class extends `GenericSensorConfig`, predefining values specific
    to accelerometers, such as input and output units.

    Parameters
    ----------
    sensitivity : float
        The sensitivity of the accelerometer, typically expressed in output units
        per input units (e.g., m/s² per V).
    natural_frequency : float
        The natural frequency of the accelerometer in Hz.
    sensor_name : str, optional
        The name or model of the accelerometer.

    Attributes
    ----------
    sensor_type : str
        Fixed as 'accelerometer'.
    input_units : Literal['V']
        Fixed as 'V' (Volts), indicating the accelerometer produces a voltage output.
    input_units_description : str
        Fixed as 'Volts' for clarity.
    output_units : Literal['M/S/S']
        Fixed as 'M/S/S' (meters per second squared), representing acceleration.
    output_units_description : str
        Fixed as 'Acceleration in meters per second squared'.
    gain : float
        Fixed at 1, assuming unitary gain in processing.
    sensitivity : float
        The specified sensitivity of the accelerometer.
    natural_frequency : float
        The natural frequency of the accelerometer in Hz.
    stage_sequence_number : int
        Fixed at 0, representing the initial processing stage.

    Notes
    -----
    - This class ensures accelerometer-specific configurations are set automatically.
    - The `sensor_type` is hardcoded as 'accelerometer' to distinguish it from
      other sensor types.
    """

    def __init__(self, sensitivity: float, natural_frequency: float, sensor_name: str = None):
        super().__init__(
            sensor_type='accelerometer',
            sensor_name=sensor_name,
            output_units='V',
            output_units_description='Volts',
            input_units='M/S/S',
            input_units_description='Acceleration in meters per second squared',
            gain=1,
            sensitivity=sensitivity,
            natural_frequency=natural_frequency,
            damping=0.707,  # Default damping factor for accelerometers
            stage_sequence_number=0
        )


class Digitizer(BaseModel):
    """
    Represents the configuration and response characteristics of a seismic digitizer.

    The digitizer converts an analog signal (voltage) from a sensor (e.g., geophone,
    accelerometer) into digital ADC counts. This class defines its essential
    parameters, including gain and sampling rate.

    Parameters
    ----------
    gain : float
        The gain of the digitizer, representing the scaling factor between
        input voltage (V) and ADC counts.
    sampling_rate : float
        The sampling rate of the digitizer in Hz.
    sensor_name : str, optional
        The name or model of the digitizer.

    Attributes
    ----------
    sensor_type : str
        Fixed as 'digitizer'.
    sensor_name : str
        The name or model of the digitizer.
    input_units : Literal['V']
        Fixed as 'V' (Volts), representing the input unit from analog sensors.
    input_units_description : str
        Fixed as 'Volts' for clarity.
    output_units : Literal['COUNTS']
        Fixed as 'COUNTS' (ADC Counts), representing the digitized output.
    output_units_description : str
        Fixed as 'ADC Counts'.
    gain : float
        The gain of the digitizer, which scales voltage to ADC counts.
    stage_sequence_number : int
        Fixed at 1, representing the digitizer as the next processing stage after
        the sensor.
    sampling_rate : float
        The sampling rate of the digitizer in Hz.

    Properties
    ----------
    stage_response : CoefficientsResponseStage
        Returns a `CoefficientsResponseStage` object representing the digitizer's
        gain response.

    Notes
    -----
    - The digitizer’s response is modeled using a simple gain factor.
    - Some digitizers may include anti-aliasing filters, which are not explicitly
      modeled here.
    - This class ensures consistency with ObsPy’s response structure, allowing
      integration with seismic metadata.
    """

    sensor_type: str = "digitizer"
    sensor_name: str = "generic digitizer"
    input_units: Literal["V"] = "V"
    input_units_description: str = "Volts"
    output_units: Literal["COUNTS"] = "COUNTS"
    output_units_description: str = "ADC Counts"
    gain: float
    stage_sequence_number: int = 1
    sampling_rate: float

    @property
    def stage_response(self):
        """
        Returns the digitizer's response stage using a simple gain factor.

        This response is modeled as a coefficient response stage, as digitizers
        generally apply a linear transformation from voltage to ADC counts.

        Returns
        -------
        CoefficientsResponseStage
            A `CoefficientsResponseStage` object defining the digitizer's gain
            and input/output unit mappings.
        """
        return CoefficientsResponseStage(
            stage_sequence_number=self.stage_sequence_number,
            gain=self.gain,
            input_units=self.input_units,
            output_units=self.output_units,
            name=self.sensor_name,
            input_units_description=self.input_units_description,
            output_units_description=self.output_units_description
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
            gain=1,  # Cables typically do not amplify signals
            frequency=0,  # No reference frequency
            input_units=self.input_units,
            output_units=self.output_units,
            response_type="LAPLACE (RADIANT/SECOND)",
            normalization_frequency=0,
            zeros=[],  # Cables do not introduce zeros
            poles=self.poles,
            input_units_description=self.input_units_description,
            output_units_description=self.output_units_description
        )


class System(BaseModel):
    """
    Represents a complete seismic acquisition system, including a sensor, an
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
    name : str
        The name of the seismic system.
    type : str
        The type of system (e.g., "seismic station", "borehole array").
    description : Optional[str], optional
        Additional descriptive details about the system.
    manufacturer : Optional[str], optional
        The company that manufactured the system.
    vendor : Optional[str], optional
        The vendor or distributor of the system.
    model : Optional[str], optional
        The specific model identifier of the system.
    serial_number : Optional[str], optional
        The serial number of the system.
    calibration_date : Optional[str], optional
        The last calibration date of the system.

    Attributes
    ----------
    response_stages : list
        A list of `ResponseStage` objects forming the system response.
    system_sensitivity : InstrumentSensitivity
        The overall system sensitivity, computed based on the sensor and digitizer.

    Properties
    ----------
    response : Response
        Returns an ObsPy `Response` object representing the full system response.
    equipment : Equipment
        Returns an ObsPy `Equipment` object containing system metadata.

    Notes
    -----
    - The response is dynamically built based on available components.
    - If a cable is included, its response is added before the digitizer.
    - The `response` property ensures seamless integration with ObsPy’s response handling.
    """

    sensor: Union[GenericSensor, Geophone, Accelerometer]
    cable: Optional[Cable] = None
    digitizer: Optional[Digitizer] = None
    name: str
    type: str
    description: Optional[str] = None
    manufacturer: Optional[str] = None
    vendor: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    calibration_date: Optional[str] = None

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
        self.sensor.stage_number_sequence = 0
        stages = [self.sensor.response_stage]  # Sensor is always required

        if self.cable:
            cable.stage_sequence_number = 1
            stages.append(self.cable.response_stage)  # Add cable if present

        if self.digitizer:
            if cable:
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
        sensor_sensitivity = self.sensor.sensitivity

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

    @property
    def equipment(self) -> Equipment:
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
            serial_number=self.serial_number,
            calibration_date=self.calibration_date,
        )




class SystemResponse:
    def __init__(
            self,
            sensor_params: Union[
                GeophoneConfig, AccelerometerConfig, GenericSensorConfig
            ]
    ):

        """
        Initializes a SystemResponse object to represent the response characteristics
        of a seismic sensor, either a geophone or an accelerometer.

        Parameters
        ----------
        sensor_params : GeophoneConfig or AccelerometerConfig
            An instance of either `GeophoneConfig` or `AccelerometerConfig`,
            containing the relevant sensor parameters.

            - For `GeophoneConfig`:
                - corner_frequency (float): The natural frequency of the geophone (Hz).
                - sensitivity (float): The sensor sensitivity (units depend on application).
                - damping (float): The damping factor of the geophone.
                - stage_sequence_number (int, optional): The sequence number of the
                  processing stage. Default is 0.

            - For `AccelerometerConfig`:
                - natural_frequency (float): The natural frequency of the accelerometer (Hz).
                - sensitivity (float): The sensor sensitivity (units depend on application).
                - stage_sequence_number (int, optional): The sequence number of the
                  processing stage. Default is 0.

        Notes
        -----
        - The `sensor_type` is determined implicitly by the provided configuration class
          (`GeophoneConfig` or `AccelerometerConfig`).
        - The `stage_sequence_number` helps define the processing order when multiple
          transformation stages are involved in the signal processing chain.
        """

        self.components_info = sensor_params.dict()
        self.sensor_type = sensor_params.sensor_type

    def add_cable(self, **cable_params):
        if 'cable' in self.components_info:
            raise ValueError("Cable response already added.")
        self.components_info['cable'] = {'params': cable_params}

    def add_digitizer(self, **digitizer_params):
        if 'digitizer' in self.components_info:
            raise ValueError("Digitizer already added.")
        self.compronents_info['digitize'] = {'params': digitizer_params}

    def validate(self):
        if 'sensor' not in self.components_info:
            raise ValueError("System must have at least a sensor.")

    def generate_stage(self, component_key, stage_sequence_number):
        component_info = self.components_info.get(component_key, {})
        if component_key == 'sensor':
            if component_info['type'] == 'geophone':
                stage = geophone_sensor_response(**component_info['params'])
            elif component_info['type'] == 'accelerometer':
                stage = accelerometer_sensor_response(**component_info['params'])
            else:
                raise ValueError("Invalid sensor type.")
        elif component_key == 'cable':
            stage = sensor_cable_response(**component_info['params'])
        # Add more conditions for other components
        stage.stage_sequence_number = stage_sequence_number
        return stage

    @property
    def response(self):
        self.validate()
        response_stages = []
        sequence_number = 1
        for key in ['sensor', 'cable', 'digitizer']:
            if key in self.components_info:
                stage = self.generate_stage(key, sequence_number)
                response_stages.append(stage)
                sequence_number += 1

        # Create and return a Response object using the gathered response stages
        system_response = Response(instrument_sensitivity=self.instrument_sensitivity,
                                   response_stages=response_stages)
        return system_response

    @property
    def instrument_sensitivity(self):
        if self.components_info['sensor']['type'] == 'gephone':
            units = 'M/S'
        else:
            units = 'M/S/S'
        natural_frequency = self.components_info['sensor'][
            'params']['natural_frequency']
        return InstrumentSensitivity(value=1, frequency=natural_frequency,
                                     input_units=units, output_units=units)


def geophone_sensor_response(natural_frequency, sensitivity, gain=1, damping=0.707,
                             stage_sequence_number=1):
    """
    Generate a Poles and Zeros response stage for a geophone.

    Parameters:
    - natural_frequency (float): natural frequency of the geophone in Hz.
    - sensitivity (float): Sensitivity of the sensor.
    - gain (float): Gain factor.
    - damping (float, optional): Damping ratio. Defaults to 0.707 (critical damping).
    - stage_sequence_number (int, optional): Sequence number for the response stage.

    Returns:
    - PolesZerosResponseStage: Response stage object with configured poles and zeros.

    Notes:
    - The function utilizes the Laplace transform in the frequency domain.
    """
    paz = corn_freq_2_paz(natural_frequency, damp=damping)
    pzr = PolesZerosResponseStage(stage_sequence_number, gain,
                                  natural_frequency, 'M/S', 'V',
                                  'LAPLACE (RADIANT/SECOND)',
                                  natural_frequency,
                                  paz['zeros'],
                                  paz['poles'])
    return pzr


def accelerometer_sensor_response(natural_frequency, gain, sensitivity=1,
                                  damping=0.707, stage_sequence_number=1):
    """
    Generate a Poles and Zeros response stage for an accelerometer.

    Parameters:
    - natural_frequency (float): natural frequency of the accelerometer in Hz.
    - gain (float): Gain factor.
    - sensitivity (float, optional): Sensitivity of the sensor. Defaults to 1.
    - damping (float, optional): Damping ratio. Defaults to 0.707 (critical damping).
    - stage_sequence_number (int, optional): Sequence number for the response stage.

    Returns:
    - PolesZerosResponseStage: Response stage object with configured poles and zeros.

    Notes:
    - The function utilizes the Laplace transform in the frequency domain.
    """
    paz = corn_freq_2_paz(natural_frequency, damp=damping)
    paz['zeros'] = []
    pzr = PolesZerosResponseStage(stage_sequence_number, gain,
                                  natural_frequency, 'M/S/S', 'V',
                                  'LAPLACE (RADIANT/SECOND)',
                                  natural_frequency, paz['zeros'],
                                  paz['poles'])
    return pzr


def sensor_cable_response(output_resistance=np.inf, cable_length=np.inf,
                          cable_capacitance=np.inf, stage_sequence_number=2):
    """
    Generate a Poles and Zeros response stage for sensor-cable interaction.

    Parameters:
    - output_resistance (float, optional): Output resistance of the sensor in ohms. D
    efaults to infinity.
    - cable_length (float, optional): Length of the cable in meters. Defaults to
    infinity.
    - cable_capacitance (float, optional): Cable capacitance in farads per meter.
    Defaults to infinity.
    - stage_sequence_number (int, optional): Sequence number for the response stage.

    Returns:
    - PolesZerosResponseStage: Response stage object with configured poles.

    Notes:
    - Assumes a Laplace transform in the frequency domain for the response.
    """
    R = output_resistance
    l = cable_length
    C = cable_capacitance
    poles = []

    if ((R * l * C) != np.inf) and ((R * l * C) != 0):
        pole_cable = -1 / (R * l * C)
        poles.append(pole_cable)

    pzr = PolesZerosResponseStage(stage_sequence_number, 1,
                                  0, 'V', 'V',
                                  'LAPLACE (RADIANT/SECOND)',
                                  0, [], poles)
    return pzr


def get_response_from_nrl(datalogger_keys, sensor_keys):
    pass


class Inventory(inventory.Inventory):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(self, *args, **kwargs)

    @classmethod
    def from_obspy_inventory_object(cls, obspy_inventory,
                                    xy_from_lat_lon=False):

        source = ns  # Network ID of the institution sending
        # the message.

        inv = cls([], ns)
        inv.networks = []
        for network in obspy_inventory.networks:
            inv.networks.append(Network.from_obspy_network(network,
                                                           xy_from_lat_lon=
                                                           xy_from_lat_lon))

        return inv

    @staticmethod
    def from_url(url):
        """
        Load an ObsPy inventory object from a URL.

        :param url: The URL to download the inventory file from.
        :type url: str
        :return: The loaded ObsPy inventory object.
        :rtype: obspy.core.inventory.Inventory
        """
        # Download the inventory file from the URL

        inventory_data = download_file_from_url(url)

        # Save the inventory data to a temporary file
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(inventory_data.read())
            temp_file.flush()

            # Load the inventory from the temporary file
            file_format = 'STATIONXML'  # Replace with the correct format
            inventory = read_inventory(temp_file.name, format=file_format)

        # Remove the temporary file after reading the inventory
        os.remove(temp_file.name)

        return inventory

    def write(self, path_or_file_obj, *args, format='stationxml', **kwargs):
        return super().write(path_or_file_obj, *args, format=format,
                             nsmap={'mq': namespace}, **kwargs)

    def get_station(self, sta):
        return self.select(sta)

    def get_channel(self, station=None, location=None, channel=None):
        return self.select(station=station, location=location, channel=channel)

    def select_instrument(self, instruments=None):
        if isinstance(instruments, list):
            for location in instruments:
                for obj_site in self.instruments:
                    if location.code == obj_site.code:
                        yield obj_site

        elif isinstance(instruments, str):
            location = instruments
            for obj_site in self.instruments:
                if location.code == obj_site.code:
                    return obj_site

    def to_bytes(self):

        file_out = BytesIO()
        self.write(file_out, format='stationxml')
        file_out.seek(0)
        return file_out.getvalue()

    @staticmethod
    def from_bytes(byte_string):
        file_in = io.BytesIO(byte_string)
        file_in.read()
        return read_inventory(file_in, format='stationxml')

    @staticmethod
    def read(path_or_file_obj, format='stationxml', **kwargs):
        return read_inventory(path_or_file_obj, format=format, **kwargs)

    def __eq__(self, other):
        return np.all(self.instruments == other.instruments)

    # def write(self, filename):
    #     super().write(self, filename, format='stationxml', nsmap={ns: ns})

    @property
    def instruments(self):
        instruments = []
        for network in self.networks:
            for station in network.stations:
                for instrument in station.instruments:
                    instruments.append(instrument)

        return np.sort(instruments)

    @property
    def short_ids(self):
        unique_ids = set()
        short_ids = []

        for network in self.networks:
            for station in network.stations:
                for instrument in station.instruments:
                    if len(instrument.code) > 6:
                        hash = hashlib.md5(instrument.code.encode()).hexdigest()[:5]
                        if hash + '0' not in unique_ids:
                            unique_ids.add(hash + '0')
                            short_ids.append(hash + '0')
                        else:
                            # First try appending numbers
                            found_unique = False
                            for i in range(1, 10):
                                potential_id = hash + str(i)
                                if potential_id not in unique_ids:
                                    unique_ids.add(potential_id)
                                    short_ids.append(potential_id)
                                    found_unique = True
                                    break

                            # If all numbers are used, start appending letters
                            if not found_unique:
                                for letter in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                                    potential_id = hash + letter
                                    if potential_id not in unique_ids:
                                        unique_ids.add(potential_id)
                                        short_ids.append(potential_id)
                                        break
        return short_ids

    def instrument_code_from_shortids(self, short_id):
        """
        return the instrument code from the short id
        :param short_id: a string representing the short id
        :return:
        """

        short_ids = self.short_ids

        for i, sid in enumerate(short_ids):
            if sid == short_id:
                return self.instruments[i].code


class Network(inventory.Network):
    __doc__ = inventory.Network.__doc__.replace('obspy', ns)

    extra_keys = ['vp', 'vs']

    def __init__(self, *args, **kwargs):

        if 'extra' not in self.__dict__.keys():  # hack for deepcopy to work
            self['extra'] = {}

        self.extra = AttribDict()

        # for extra_key in self.extra_keys:
        #     if extra_key in kwargs.keys():
        #         self.extra[extra_key] = kwargs.pop(extra_key)

        if 'vp' in kwargs.keys():
            self.vp = kwargs.pop('vp')
        if 'vs' in kwargs.keys():
            self.vs = kwargs.pop('vs')
        if 'units' in kwargs.keys():
            self.units = kwargs.pop('units')

        super().__init__(*args, **kwargs)

    def __setattr__(self, name, value):
        name = name.lower()
        if name in self.extra_keys:
            self.extra[name] = AttribDict({"value": f"{value}", "namespace": namespace})
        else:
            super().__setattr__(name, value)

    def __getattr__(self, item):
        item = item.lower()
        if item in self.extra_keys:
            try:
                return float(self.extra[item].value)
            except:
                return self.extra[item].value
        else:
            super().__getattr__(item)

    def __setitem__(self, key, value):
        self.__dict__['key'] = value

    @classmethod
    def from_obspy_network(cls, obspy_network, xy_from_lat_lon=False,
                           input_projection=4326, output_projection=None):

        net = cls(obspy_network.code)

        for key in obspy_network.__dict__.keys():
            if 'stations' in key:
                net.__dict__[key] = []
            else:
                try:
                    net.__dict__[key] = obspy_network.__dict__[key]
                except Exception as e:
                    logger.error(e)

        for i, station in enumerate(obspy_network.stations):
            net.stations.append(Station.from_obspy_station(station,
                                                           xy_from_lat_lon))

        return net

    def get_grid_extent(self, padding_fraction=0.1, ignore_stations=[],
                        ignore_sites=[]):
        """
        return the extents of a grid encompassing all sensors comprised in the
        network
        :param padding_fraction: buffer to add around the stations
        :param ignore_stations: list of stations to exclude from the
        calculation of the grid extents
        :param ignore_sites: list of location to exclude from the calculation of
        the grid extents
        :type ignore_stations: list
        :return: return the lower and upper corners
        :rtype: dict
        """

        xs = []
        ys = []
        zs = []

        coordinates = []
        for station in self.stations:
            if station.code in ignore_stations:
                continue
            for location in station.instruments:
                if location.code in ignore_sites:
                    continue
                coordinates.append(location.loc)

        min = np.min(coordinates, axis=0)
        max = np.max(coordinates, axis=0)
        # center = min + (max - min) / 2
        center = (min + max) / 2
        d = (max - min) * (1 + padding_fraction)

        c1 = center - d / 2
        c2 = center + d / 2

        return c1, c2

    @property
    def site_coordinates(self):
        coordinates = []
        for station in self.stations:
            coordinates.append(station.site_coordinates)

        return np.array(coordinates)

    @property
    def station_coordinates(self):
        return np.array([station.loc for station in self.stations])

    @property
    def instruments(self):
        instruments = []
        for station in self.stations:
            for instrument in station.instruments:
                instruments.append(instrument)
        return instruments

    @property
    def sensors(self):
        return self.instruments

    @property
    def instrument_coordinates(self):
        coordinates = []
        for instrument in self.instruments:
            coordinates.append(instrument.loc)
        return np.array(coordinates)

    @property
    def sensor_coordinates(self):
        return self.instrument_coordinates


class Station(inventory.Station):
    __doc__ = inventory.Station.__doc__.replace('obspy', ns)

    def __init__(self, *args, coordinates: Coordinates = Coordinates(0, 0, 0), **kwargs):

        if 'extra' not in self.__dict__.keys():  # hack for deepcopy to work
            self['extra'] = {}

        self.extra = AttribDict()

        if 'latitude' not in kwargs.keys():
            kwargs['latitude'] = 0
        if 'longitude' not in kwargs.keys():
            kwargs['longitude'] = 0
        if 'elevation' not in kwargs.keys():
            kwargs['elevation'] = 0

        # initialize the extra key

        if not hasattr(self, 'extra'):
            self.extra = AttribDict()

        super().__init__(*args, **kwargs)

        self.extra['coordinates'] = coordinates.to_extra_key(namespace=namespace)

    def __setattr__(self, name, value):
        if name == 'coordinates':
            self.extra[name] = value.to_extra_key(namespace=namespace)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, item):
        if item == 'coordinates':
            return Coordinates.from_extra_key(self.extra[item])
        else:
            super().__getattr__(item)

    def __setitem__(self, key, value):
        self.__dict__['key'] = value

    def __eq__(self, other):
        if not super().__eq__(other):
            return False

        return self.coordinates == other.coordinates

    @classmethod
    def from_obspy_station(cls, obspy_station, xy_from_lat_lon=False):

        #     cls(*params) is same as calling Station(*params):

        stn = cls(obspy_station.code, latitude=obspy_station.latitude,
                  longitude=obspy_station.longitude,
                  elevation=obspy_station.elevation)
        for key in obspy_station.__dict__.keys():
            try:
                stn.__dict__[key] = obspy_station.__dict__[key]
            except Exception as e:
                logger.error(e)

        if xy_from_lat_lon:
            if (stn.latitude is not None) and (stn.longitude is not None):

                x, y = lon_lat_x_y(
                    longitude=stn.longitude, latitude=stn.latitude)

                z = obspy_station.elevation

                stn.coordinates = Coordinates(
                    x, y, z, coordinate_system=CoordinateSystem.NEU)

            else:
                logger.warning(f'Latitude or Longitude are not'
                               f'defined for station {obspy_station.code}.')

                output_projection = 32725

        stn.channels = []

        for channel in obspy_station.channels:
            stn.channels.append(Channel.from_obspy_channel(channel,
                                                           xy_from_lat_lon=
                                                           xy_from_lat_lon))

        return stn

    @property
    def x(self):
        return self.coordinates.x

    @property
    def y(self):
        return self.coordinates.y

    @property
    def z(self):
        return self.coordinates.z

    @property
    def loc(self):
        return np.array(self.coordinates.loc)

    @property
    def instruments(self):
        location_codes = []
        channel_dict = {}
        instruments = []
        for channel in self.channels:
            location_codes.append(channel.location_code)
            channel_dict[channel.location_code] = []

        for channel in self.channels:
            channel_dict[channel.location_code].append(channel)

        for key in channel_dict.keys():
            instruments.append(Instrument(self, channel_dict[key]))

        return instruments

    @property
    def sensors(self):
        return self.instruments

    @property
    def instrument_coordinates(self):
        coordinates = []
        for instrument in self.instruments:
            coordinates.append(instrument.loc)
        return np.array(coordinates)

    @property
    def sensor_coordinates(self):
        return self.instrument_coordinates

    def __str__(self):
        contents = self.get_contents()

        x = self.x
        y = self.y
        z = self.z

        location_count = len(self.instruments)
        channel_count = len(self.channels)

        format_dict = {
            'code': self.code or 'N/A',
            'location_count': location_count or 0,
            'channel_count': channel_count or 0,
            'start_date': self.start_date or 'N/A',
            'end_date': self.end_date or 'N/A',
            'x': f"{x:.0f}" if x is not None else 0,
            'y': f"{y:.0f}" if y is not None else 0,
            'z': f"{z:.0f}" if z is not None else 0
        }

        ret = ("\tStation Code: {code}\n"
               "\tLocation Count: {location_count}\n"
               "\tChannel Count: {channel_count}\n"
               "\tDate range: {start_date} - {end_date}\n"
               "\tx: {x}, y: {y}, z: {z} m\n").format(**format_dict)

        if getattr(self, 'extra', None):
            if getattr(self.extra, 'x', None) and getattr(self.extra, 'y',
                                                          None):
                x = self.x
                y = self.y
                z = self.z
                ret = ("Station {station_name}\n"
                       "\tStation Code: {station_code}\n"
                       "\tLocation Count: {site_count}\n"
                       "\tChannel Count: {selected}/{total}"
                       " (Selected/Total)\n"
                       "\tDate range: {start_date} - {end_date}\n"
                       "\tEasting [x]: {x:.0f} m, Northing [y] m: {y:.0f}, "
                       "Elevation [z]: {z:.0f} m\n")

        ret = ret.format(
            station_name=contents["stations"][0],
            station_code=self.code,
            site_count=len(self.instruments),
            selected=self.selected_number_of_channels,
            total=self.total_number_of_channels,
            start_date=str(self.start_date),
            end_date=str(self.end_date) if self.end_date else "",
            restricted=self.restricted_status,
            alternate_code="Alternate Code: %s " % self.alternate_code if
            self.alternate_code else "",
            historical_code="Historical Code: %s " % self.historical_code if
            self.historical_code else "",
            x=x, y=y, z=z)
        ret += "\tAvailable Channels:\n"
        ret += "\n".join(_textwrap(
            ", ".join(_unified_content_strings(contents["channels"])),
            initial_indent="\t\t", subsequent_indent="\t\t",
            expand_tabs=False))

        return ret

    def __repr__(self):
        return self.__str__()


class Instrument:
    """
    This class is a container for grouping the channels into coherent entity
    that are Instruments. From the uquake package perspective a station is
    the physical location where data acquisition instruments are grouped.
    One or multiple instruments can be connected to a station.
    """

    def __init__(self, station, channels):

        location_codes = []
        for channel in channels:
            location_codes.append(channel.location_code)
            if len(np.unique(location_codes)) > 1:
                raise ValueError('the channels in the channel list should have a unique '
                                 'location code')

        self.location_code = location_codes[0]

        if len(np.unique(location_codes)) > 1:
            logger.error('the channels in the channel list should have a'
                         'unique location code')
            raise KeyError

        self.station = station
        self.channels = channels

    def __repr__(self):
        ret = f'\tInstrument {self.instrument_code}\n' \
              f'\tx: {self.x:.0f} m, y: {self.y:.0f} m z: {self.z:0.0f} m\n' \
              f'\tChannel Count: {len(self.channels)}'

        return ret

    def __str__(self):
        return self.instrument_code

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __getitem__(self, item):
        return self.channels[item]

    @property
    def loc(self):
        return np.array([self.x, self.y, self.z])

    @property
    def alternate_code(self):
        return self.channels[0].alternative_code

    @property
    def x(self):
        return self.channels[0].x

    @property
    def y(self):
        return self.channels[0].y

    @property
    def z(self):
        return self.channels[0].z

    @property
    def coordinates(self):
        return self.channels[0].coordinates

    @property
    def station_code(self):
        return self.station.code

    @property
    def instrument_code(self):
        return self.code

    @property
    def simplified_code(self):
        return f'{self.station_code}{self.location_code}'

    @property
    def code(self):
        return self.make_instrument_code(self.station_code, self.location_code)

    @property
    def label(self):
        return self.simplified_code

    @property
    def sensor_type_code(self):
        return self.channels[0].code[0:-1]

    @staticmethod
    def make_instrument_code(station_code, location_code):
        return f'{station_code}.{location_code}'

    @property
    def coordinate_system(self):
        return self.coordinates.coordinate_system


class Channel(inventory.Channel):
    defaults = {}

    __doc__ = inventory.Channel.__doc__.replace('obspy', ns)

    def __init__(self, code, location_code, active: bool = True, oriented: bool = False,
                 coordinates: Coordinates = Coordinates(0, 0, 0),
                 orientation_vector=None, **kwargs):

        latitude = kwargs.pop('latitude') if 'latitude' in kwargs.keys() else 0
        longitude = kwargs.pop('longitude') if 'longitude' in kwargs.keys() else 0
        elevation = kwargs.pop('elevation') if 'elevation' in kwargs.keys() else 0
        depth = kwargs.pop('depth') if 'depth' in kwargs.keys() else 0

        super().__init__(code, location_code, latitude, longitude,
                         elevation, depth, **kwargs)

        if 'extra' not in self.__dict__.keys():  # hack for deepcopy to work
            self.__dict__['extra'] = {}

        self.extra['coordinates'] = coordinates.to_extra_key(namespace=namespace)
        set_extra(self, 'active', active, namespace=namespace)
        set_extra(self, 'oriented', oriented, namespace=namespace)

        if orientation_vector is not None:
            # making the orientation vector (cosine vector) unitary
            orientation_vector = orientation_vector / np.linalg.norm(orientation_vector)
            self.set_orientation(orientation_vector)

    @classmethod
    def from_obspy_channel(cls, obspy_channel, xy_from_lat_lon=False,
                           output_projection=None, input_projection=4326):

        cha = cls(obspy_channel.code, obspy_channel.location_code,
                  latitude=obspy_channel.latitude,
                  longitude=obspy_channel.longitude,
                  elevation=obspy_channel.elevation,
                  depth=obspy_channel.depth)

        if hasattr(obspy_channel, 'extra'):
            for key in cha.extra.keys():
                if key not in obspy_channel.__dict__['extra'].keys():
                    cha.__dict__['extra'][key] = 0
                else:
                    cha.__dict__['extra'][key] = \
                        obspy_channel.__dict__['extra'][key]

        for key in obspy_channel.__dict__.keys():
            cha.__dict__[key] = obspy_channel.__dict__[key]

        if xy_from_lat_lon:
            if (cha.latitude is not None) and (cha.longitude is not None):

                x, y = lon_lat_x_y(longitude=cha.longitude, latitude=cha.latitude)

                z = cha.elevation

                coordinates = Coordinates(
                    x, y, z, coordinate_system=CoordinateSystem.NEU)
                cha.coordinates = coordinates

        return cha

    def __getattr__(self, item):
        if item == 'coordinates':
            return Coordinates.from_extra_key(self.extra['coordinates'])
        elif item in ('active', 'oriented'):
            return get_extra(item)
        else:
            if hasattr(super(), item):
                return getattr(super(), item)
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        if key == 'coordinates':
            self.extra[key] = value.to_extra_key(namespace=namespace)
        elif key in ('active', 'oriented'):
            set_extra(self, key, value, namespace=namespace)
        else:
            super().__setattr__(key, value)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__['key'] = value

    def __repr__(self):
        time_range = f"{self.start_date} - {self.end_date}" if self.start_date and \
                                                               self.end_date else 'N/A'

        if self.coordinates.coordinate_system == CoordinateSystem.ENU:

            attributes = {
                'Channel': self.code,
                'Location': self.location_code,
                'Time range': time_range,
                'Easting [x]': f"{self.x:0.0f} m" if self.x is not None else 'N/A',
                'Northing [y]': f"{self.y:0.0f} m" if self.y is not None else 'N/A',
                'Elevation (UP) [z]':
                    f"{self.z:0.0f} m" if self.z is not None else 'N/A',
                'Dip (degrees)': f"{self.dip:0.0f}" if self.dip is not None else 'N/A',
                'Azimuth (degrees)': f"{self.azimuth:0.0f}" if self.azimuth
                                                               is not None else 'N/A',
                'Response information': 'available' if self.response else 'not available'
            }

        elif self.coordinates.coordinate_system == CoordinateSystem.NED:
            attributes = {
                'Channel': self.code,
                'Location': self.location_code,
                'Time range': time_range,
                'Northing [x]': f"{self.x:0.0f} m" if self.x is not None else 'N/A',
                'Easting [y]': f"{self.y:0.0f} m" if self.y is not None else 'N/A',
                'Depth (Down) [z]': f"{self.z:0.0f} m" if self.z is not None else 'N/A',
                'Dip (degrees)': f"{self.dip:0.0f}" if self.dip is not None else 'N/A',
                'Azimuth (degrees)': f"{self.azimuth:0.0f}" if self.azimuth
                                                               is not None else 'N/A',
                'Response information': 'available' if self.response else 'not available'
            }

        elif self.coordinates.coordinate_system == CoordinateSystem.NEU:
            attributes = {
                'Channel': self.code,
                'Location': self.location_code,
                'Time range': time_range,
                'Northing [x]': f"{self.x:0.0f} m" if self.x is not None else 'N/A',
                'Easting [y]': f"{self.y:0.0f} m" if self.y is not None else 'N/A',
                'Elevation (up) [z]':
                    f"{self.z:0.0f} m" if self.z is not None else 'N/A',
                'Dip (degrees)': f"{self.dip:0.0f}" if self.dip is not None else 'N/A',
                'Azimuth (degrees)': f"{self.azimuth:0.0f}" if self.azimuth
                                                               is not None else 'N/A',
                'Response information': 'available' if self.response else 'not available'
            }

        elif self.coordinates.coordinate_system == CoordinateSystem.END:

            attributes = {
                'Channel': self.code,
                'Location': self.location_code,
                'Time range': time_range,
                'Easting [x]': f"{self.x:0.0f} m" if self.x is not None else 'N/A',
                'Northing [y]': f"{self.y:0.0f} m" if self.y is not None else 'N/A',
                'Depth (Down) [z]': f"{self.z:0.0f} m" if self.z is not None else 'N/A',
                'Dip (degrees)': f"{self.dip:0.0f}" if self.dip is not None else 'N/A',
                'Azimuth (degrees)': f"{self.azimuth:0.0f}" if self.azimuth
                                                               is not None else 'N/A',
                'Response information': 'available' if self.response else 'not available'
            }

        ret = "\n".join([f"{key}: {value}" for key, value in attributes.items()])
        return ret

    def __eq__(self, other):
        if not super().__eq__(other):
            return False

        return (self.coordinates == other.coordinates and
                self.active == other.active and
                self.oriented == other.oriented)

    def set_orientation(self, orientation_vector):
        """
        set the Azimuth and Dip from an orientation vector assuming the
        orientation vector provided is east, north, up.
        :param self:
        :param orientation_vector:
        :return:

        Azimuth is defined from the north direction and positive clockwise
        Dip is defined as the angle from the horizontal plane and positive down
        """

        azimuth, dip = self.calculate_azimuth_and_dip(orientation_vector)
        self.azimuth = azimuth
        self.dip = dip

    def calculate_azimuth_and_dip(self, orientation_vector):
        """
        calculate the Azimuth and Dip from an orientation vector assuming the
        orientation vector provided is east, north, up.
        :param orientation_vector:
        :return:

        Azimuth is defined from the north direction and positive clockwise
        Dip is defined as the angle from the horizontal plane and positive down
        """

        if self.coordinates.coordinate_system == CoordinateSystem.ENU:

            east = orientation_vector[0]
            north = orientation_vector[1]
            up = orientation_vector[2]

        elif self.coordinates.coordinate_system == CoordinateSystem.NED:

            north = orientation_vector[0]
            east = orientation_vector[1]
            up = - orientation_vector[2]

        elif self.coordinates.coordinate_system == CoordinateSystem.NEU:
            north = orientation_vector[0]
            east = orientation_vector[1]
            up = orientation_vector[2]

        elif self.coordinates.coordinate_system == CoordinateSystem.END:
            east = orientation_vector[0]
            north = orientation_vector[1]
            up = - orientation_vector[2]

        horizontal_length = np.linalg.norm([east, north])

        azimuth = np.arctan2(east, north) * 180 / np.pi
        if azimuth < 0:
            azimuth = 360 + azimuth

        dip = np.arctan2(-up, horizontal_length) * 180 / np.pi

        return azimuth, dip

    @property
    def orientation_vector(self):
        """
        Computes the orientation vector based on the current azimuth and dip values.

        The method first converts the azimuth and dip from degrees to radians.
        It then calculates
        the components of the vector (up, east, north) based on trigonometric
        relationships. The
        final orientation vector is dependent on the coordinate system of the instance
        (either ENU or NED).

        For ENU (East-North-Up), the vector is returned as [east, north, up].
        For NED (North-East-Down), it is returned as [north, east, -up].

        Returns:
            numpy.ndarray: A 3-element array representing the orientation vector in the
            specified coordinate system.
        """

        up = -np.sin(self.dip * np.pi / 180)
        east = np.sin(self.azimuth * np.pi / 180) * \
               np.cos(self.dip * np.pi / 180)
        north = np.cos(self.azimuth * np.pi / 180) * \
                np.cos(self.dip * np.pi / 180)

        if self.coordinates.coordinate_system == CoordinateSystem.ENU:
            ov = np.array([east, north, up])

        elif self.coordinates.coordinate_system == CoordinateSystem.NED:
            ov = np.array([north, east, -up])

        elif self.coordinates.coordinate_system == CoordinateSystem.NEU:
            ov = np.array([north, east, up])
        elif self.coordinates.coordinate_system == CoordinateSystem.END:
            ov = np.array([east, north, -up])
        # else:
        #     raise ValueError('coordinate system not supported')
        return ov

    @property
    def coordinate_system(self):
        return self.coordinates.coordinate_system

    @property
    def x(self):
        return self.coordinates.x

    @property
    def y(self):
        return self.coordinates.y

    @property
    def z(self):
        return self.coordinates.z

    @property
    def loc(self):
        return np.array([self.x, self.y, self.z])


def load_from_excel(file_name) -> Inventory:
    """
    Read in a multi-sheet excel file with network metadata sheets:
        Locations, Networks, Hubs, Stations, Components, Locations, Cables,
        Boreholes
    Organize these into a uquake Inventory object

    :param xls_file: path to excel file
    :type: xls_file: str
    :return: inventory
    :rtype: uquake.core.data.inventory.Inventory

    """

    df_dict = pd.read_excel(file_name, sheet_name=None)

    source = df_dict['Locations'].iloc[0]['code']
    # sender (str, optional) Name of the institution sending this message.
    sender = df_dict['Locations'].iloc[0]['operator']
    net_code = df_dict['Networks'].iloc[0]['code']
    net_descriptions = df_dict['Networks'].iloc[0]['name']

    contact_name = df_dict['Networks'].iloc[0]['contact_name']
    contact_email = df_dict['Networks'].iloc[0]['contact_email']
    contact_phone = df_dict['Networks'].iloc[0]['contact_phone']
    site_operator = df_dict['Locations'].iloc[0]['operator']
    site_country = df_dict['Locations'].iloc[0]['country']
    site_name = df_dict['Locations'].iloc[0]['name']
    location_code = df_dict['Locations'].iloc[0]['code']

    print("source=%s" % source)
    print("sender=%s" % sender)
    print("net_code=%s" % net_code)

    network = Network(net_code)
    inventory = Inventory([network], source)

    # obspy requirements for PhoneNumber are super specific:
    # So likely this will raise an error if/when someone changes the value in
    # Networks.contact_phone
    """
    PhoneNumber(self, area_code, phone_number, country_code=None,
    description=None):
        :type area_code: int
        :param area_code: The area code.
        :type phone_number: str
        :param phone_number: The phone number minus the country and
        area code. Must be in the form "[0-9]+-[0-9]+", e.g. 1234-5678.
        :type country_code: int, optional
        :param country_code: The country code.
    """

    import re
    phone = re.findall(r"[\d']+", contact_phone)
    area_code = int(phone[0])
    number = "%s-%s" % (phone[1], phone[2])
    phone_number = PhoneNumber(area_code=area_code, phone_number=number)

    person = Person(names=[contact_name], agencies=[site_operator],
                    emails=[contact_email], phones=[phone_number])
    operator = Operator(site_operator, contacts=[person])
    location = Instrument(name=site_name, description=site_name,
                          country=site_country)

    # Merge Stations+Components+Locations+Cables info into sorted stations +
    # channels dicts:

    df_dict['Stations']['station_code'] = df_dict['Stations']['code']
    df_dict['Locations']['sensor_code'] = df_dict['Locations']['code']
    df_dict['Components']['code_channel'] = df_dict['Components']['code']
    df_dict['Components']['sensor'] = df_dict['Components']['sensor__code']
    df_merge = pd.merge(df_dict['Stations'], df_dict['Locations'],
                        left_on='code', right_on='station__code',
                        how='inner', suffixes=('', '_channel'))

    df_merge2 = pd.merge(df_merge, df_dict['Components'],
                         left_on='sensor_code', right_on='sensor__code',
                         how='inner', suffixes=('', '_sensor'))

    df_merge3 = pd.merge(df_merge2, df_dict['Cable types'],
                         left_on='cable__code', right_on='code',
                         how='inner', suffixes=('', '_cable'))

    df_merge4 = pd.merge(df_merge3, df_dict['Location types'],
                         left_on='sensor_type__model', right_on='model',
                         how='inner', suffixes=('', '_sensor_type'))

    df = df_merge4.sort_values(['sensor_code', 'location_code']).fillna(0)

    # Need to sort by unique station codes, then look through 1-3 channels
    # to add
    stn_codes = set(df['sensor_code'])
    stations = []

    for code in stn_codes:
        chan_rows = df.loc[df['sensor_code'] == code]
        row = chan_rows.iloc[0]
        station = {}
        # Set some keys explicitly
        #     from ipdb import set_trace; set_trace()
        station['code'] = '{}'.format(row['sensor_code'])
        station['x'] = row['location_x_channel']
        station['y'] = row['location_y_channel']
        station['z'] = row['location_z_channel']
        station['loc'] = np.array(
            [station['x'], station['y'], station['z']])
        station['long_name'] = "{}.{}.{:02d}".format(row['network__code'],
                                                     row['station_code'],
                                                     row['location_code'])

        # MTH: 2019/07 Seem to have moved from pF to F on Cables sheet:
        station['cable_capacitance_pF_per_meter'] = row['c'] * 1e12

        # Set the rest (minus empty fields) directly from spreadsheet names:
        renamed_keys = {'sensor_code', 'location_x', 'location_y',
                        'location_z', 'name'}

        # These keys are either redundant or specific to channel, not station:
        remove_keys = {'code', 'id_channel', 'orientation_x',
                       'orientation_y', 'orientation_z', 'id_sensor',
                       'enabled_channel', 'station_id', 'id_cable'}
        keys = row.keys()
        empty_keys = keys[pd.isna(row)]
        keys = set(keys) - set(empty_keys) - renamed_keys - remove_keys

        for key in keys:
            station[key] = row[key]

        # Added keys:
        station['motion'] = 'VELOCITY'

        if row['sensor_type'].upper() == 'ACCELEROMETER':
            station['motion'] = 'ACCELERATION'

        # Attach channels:
        station['channels'] = []

        for index, rr in chan_rows.iterrows():
            chan = {}
            chan['cmp'] = rr['code_channel_sensor'].upper()
            chan['orientation'] = np.array([rr['orientation_x'],
                                            rr['orientation_y'],
                                            rr['orientation_z']])
            chan['x'] = row['location_x_channel']
            chan['y'] = row['location_y_channel']
            chan['z'] = row['location_z_channel']
            chan['enabled'] = rr['enabled']
            station['channels'].append(chan)

        stations.append(station)

    # from ipdb import set_trace; set_trace()

    # Convert these station dicts to inventory.Station objects and attach to
    # inventory.network:
    station_list = []

    for station in stations:
        # This is where namespace is first employed:
        station = Station.from_station_dict(station, site_name)
        station.location = location
        station.operators = [operator]
        station_list.append(station)

    network.stations = station_list

    return inventory


@expand_input_format_compatibility
def read_inventory(path_or_file_object, format='STATIONXML',
                   xy_from_lat_lon=False, *args, **kwargs) -> Inventory:
    """
    Read inventory file
    :param path_or_file_object: the path to the inventory file or a file object
    :param format: the format
    :param xy_from_lat_lon: if True convert populate the XY field by converting
    the latitude and longitude to UTM
    :param args: see obspy.core.inventory.read_inventory for more information
    :param kwargs: see obspy.core.inventory.read_inventory for more information
    :return: an inventory object
    :rtype: ~uquake.core.inventory.Inventory
    """


    if type(path_or_file_object) is Path:
        path_or_file_object = str(path_or_file_object)

    # del kwargs['xy_from_lat_lon']

    if (format not in ENTRY_POINTS['inventory'].keys()) or \
            (format.upper() == 'STATIONXML'):

        obspy_inv = inventory.read_inventory(path_or_file_object, *args, format=format,
                                             **kwargs)

        return Inventory.from_obspy_inventory_object(
            obspy_inv, xy_from_lat_lon=xy_from_lat_lon)

    else:
        format_ep = ENTRY_POINTS['inventory'][format]

        read_format = load_entry_point(format_ep.dist.key,
                                       'obspy.io.%s' %
                                       format_ep.name, 'readFormat')

        return expand_input_format_compatibility(
            read_format(path_or_file_object, **kwargs))

    # else:


# def read_inventory(filename, format='STATIONXML', **kwargs):
#     if isinstance(filename, Path):
#         filename = str(filename)
#
#     if format in ENTRY_POINTS['inventory'].keys():
#         format_ep = ENTRY_POINTS['inventory'][format]
#         read_format = load_entry_point(format_ep.dist.key,
#                                        'obspy.plugin.inventory.%s' %
#                                        format_ep.name, 'readFormat')
#
#         return Inventory(inventory=read_format(filename, **kwargs))
#
#     else:
#         return inventory.read_inventory(filename, format=format,
#                                                **kwargs)


read_inventory.__doc__ = inventory.read_inventory.__doc__.replace(
    'obspy', ns)
