from ...core.inventory import (Inventory, Network, Station, Site, Channel,
                                   geophone_response, SystemResponse)
from ...core import UTCDateTime
from datetime import datetime
from typing import TextIO, Union, List, Dict
from pathlib import Path
from pandas import DataFrame
import string
import random


# The naming convention in the sensor file is not consistent, making it hard
# to translate the .sen file into a more
# standard representation

# The below table is not great
sensor_type_lookup_table = {
    'G3': 'geophone',
    'G1': 'geophone',
    'S3': 'geophone',
    'S1': 'geophone'
}

channel_code_lookup_table = {
    'geophone': 'GP'
}


def create_sensor_dictionary(column_name_line: List, sensor_lines: List) \
        -> DataFrame:
    columns = column_name_line.split()
    sensor_dict = {}

    # initializing the dictionary with the header
    columns[0] = columns[0][2:]
    new_columns = []
    for column in columns:
        if column == 'Orientation':
            new_columns.append('Orientation_N')
            new_columns.append('Orientation_E')
            new_columns.append('Orientation_U')
        else:
            new_columns.append(column)
    columns = new_columns
    for column in columns:
        sensor_dict[column] = []

    sensor_dict['location_code'] = []
    sensor_dict['station_code'] = []

    # filling the dictionary list
    for sensor_line in sensor_lines:
        line_elements = sensor_line.split()
        for i, column in enumerate(columns):
            if column == 'Description':
                try:
                    location_code = int(line_elements[i][-3:] - 1)
                    if location_code < 0:
                        location_code = 0
                    sensor_dict['location_code'].append(location_code)
                    sensor_dict['station_code'].append(line_elements[i][:-4])
                except Exception as e:
                    sensor_dict['location_code'].append(0)
                    sensor_dict['station_code'].append(line_elements[i])

            try:
                line_element = float(line_elements[i])
            except Exception as e:
                line_element = line_elements[i]
            sensor_dict[column].append(line_element)

    return DataFrame(sensor_dict)


def create_channels(sensor_df: DataFrame) -> Dict:
    dict_out = {}
    for row in sensor_df.iterrows():
        row = row[1]
        channel_code = f'{channel_code_lookup_table[sensor_type_lookup_table[row["Type"]]]}{row["Cmp"]:0.0f}'
        location_code = f'{row["location_code"]:02d}'

        # Important Notes:
        # Sampling rate is assumed and given a value of 6000, this information
        # should be available
        # start_date is not defined in the sensor file but is a value that is
        # important
        # end_date is not defined in the sensor file but is a value that is
        # important if the sensor has been decommissioned.

        channel = Channel(code=channel_code, location_code=location_code,
                          x=row['Easting'], y=row['Northing'], z=row['Elev'],
                          sample_rate=6000,
                          start_date=UTCDateTime(datetime.fromtimestamp(0)))
        channel.set_orientation([row['Orientation_E'], row['Orientation_N'],
                                 row['Orientation_U']])
        if sensor_type_lookup_table[row['Type']] == 'geophone':
            # assuming for now that the sensor is critically damped

            system_response = SystemResponse()

            ### Need to handle the gain row['Gain']

            system_response.add_geophone(row['Frequency'], row['Sens'])
            system_response.add_digitizer(max_voltage=row['Vmax'],
                                          digitizer_name='Paladin')

            sensitivity = 1
            for stage in system_response.response_stages:
                sensitivity *= stage.stage_gain

            system_response.sensitivity = sensitivity

            channel.response = system_response.response

            # channel.response = geophone_response(row['Frequency'],
            # row['Gain'], sensitivity=row['Sens'])
            # There might be some change required to convert the data in the
            # waveform files that are expressed in ADC counts. The previous
            # and following line may not be accurate.
            channel.calibration_units = 'COUNT'

            # To be rigorous, the sensor, pre-amplifier, digitizer should be
            # added
            # channel.sensor, channel.pre_amplifier, channel.data_logger.

        if row['station_code'] not in dict_out.keys():
            dict_out[row['station_code']] = []

        dict_out[row['station_code']].append(channel)

    return dict_out


def create_station(channel_dict: Dict):
    stations = []
    for key in channel_dict.keys():
        letters = string.ascii_uppercase
        station_code = ''.join(random.choice(letters) for i in range(4))
        stations.append(Station(code=station_code, channels=channel_dict[key],
                                total_number_of_channels=
                                len(channel_dict[key]),
                                selected_number_of_channels=
                                len(channel_dict[key]),
                                start_date=
                                UTCDateTime(datetime.fromtimestamp(0)),
                                alternate_code=key))
    return stations


def read_esg_sensor_file(sen_file: Union[TextIO, str, Path], network: str = "",
                         header_length: int = 38) -> Inventory:
    if not isinstance(sen_file, TextIO):
        sen_file = open(sen_file)

    # Reading the file and removing the header
    previous_site = None

    sensor_file_lines = sen_file.readlines()
    column_name_line = sensor_file_lines[header_length - 2]
    sensor_lines = sensor_file_lines[header_length:]
    sensor_df = create_sensor_dictionary(column_name_line, sensor_lines)

    dict_channel = create_channels(sensor_df)
    stations = create_station(dict_channel)
    network = Network(code=network, stations=stations)
    inventory = Inventory(networks=[network], source='ESG Solutions',
                          sender='ESG Solutions', module='uQuake',
                          module_uri="")
    return inventory


# if __name__ == '__main__':
#     inventory = read_esg_sensor_file("esg_test.sen", network='GB')
#     inventory.write('ESG_test.xml', format='stationxml')
