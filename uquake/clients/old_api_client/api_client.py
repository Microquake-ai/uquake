import json
import urllib.parse
from io import BytesIO

import requests
from dateutil import parser
from obspy import UTCDateTime
from microquake.core.event import Catalog
from obspy.core.util.attribdict import AttribDict

from ...core.logging import logger
from ...core import read
from ...core.event import Ray, read_events
from uuid import uuid4
# import seismic_client
from microquake.core.settings import settings

timeout = 200

from microquake.core.decorators import deprecated


class RequestRay(AttribDict):
    def __init__(self, json_data):
        super(RequestRay, self).__init__(json_data)
        self.ray = Ray(self.nodes)
        del self.nodes


class RequestEvent:
    def __init__(self, ev_dict):
        for key in ev_dict.keys():
            if 'time' in key:
                if key == 'timezone':
                    continue

                if type(ev_dict[key]) is not str:
                    setattr(self, key, ev_dict[key])

                    continue
                setattr(self, key, UTCDateTime(parser.parse(ev_dict[key])))
            else:
                setattr(self, key, ev_dict[key])

    def get_event(self):
        event_file = requests.request('GET', self.event_file, timeout=timeout)

        return read_events(event_file.content, format='QUAKEML')

    def get_waveforms(self):
        if self.waveform_file is None:
            logger.warning(f'No waveform object associated with event '
                           f'{self.event_resource_id}')
            return None
        waveform_file = requests.request('GET', self.waveform_file, 
                                         timeout=timeout)
        byte_stream = BytesIO(waveform_file.content)

        return read(byte_stream, format='MSEED')

    def get_context_waveforms(self):
        if self.waveform_context_file is None:
            logger.warning(f'No context trace associated with event '
                           f'{self.event_resource_id}')
            return None
        waveform_context_file = requests.request('GET',
                                                 self.waveform_context_file,
                                                 timeout=timeout)
        byte_stream = BytesIO(waveform_context_file.content)

        return read(byte_stream)

    def get_variable_length_waveforms(self):
        if self.variable_size_waveform_file is None:
            logger.warning(f'No variable length waveform associated with '
                           f'event {self.event_resource_id}')
            return None

        variable_length_waveform_file = requests.request('GET',
                                                         self.variable_size_waveform_file,
                                                         timeout=timeout)

        byte_stream = BytesIO(variable_length_waveform_file.content)

        return read(byte_stream)

    def select(self):
        # select by different attributes
        # TODO write this function :-)
        pass

    def keys(self):
        return self.__dict__.keys()

    def __repr__(self):
        outstr = "\n"

        for key in self.keys():
            outstr += '%s: %s\n' % (key, self.__dict__[key])
        outstr += "\n"

        return outstr


# class SeismicClient:
#     def __init__(self, api_base_url, username, password,
#                  microquake_event_type=True):
#         configuration = seismic_client.Configuration()
#         configuration.username = username
#         configuration.password = password
#         self.api_base_url = api_base_url
#         self.username = username
#         self.password = password
#         self.api_instance = seismic_client.ApiApi(
#              seismic_client.ApiClient(configuration))
#
#     @staticmethod
#     def event_detail():
#         return seismic_client.EventDetail()
#
#     def events_list(self, start_time=None,
#                     end_time=None,
#                     microquake_event_type=True,
#                     **kwargs):
#         """
#         get the list of event from start_time to end_time
#         :param start_time: start_time in UTC expressed as
#         "YYYY-MM-DDTHH:MM:SS.MS" or a UTCDateTime, or a datetime
#         explicitely providing this paramter will override the time_utc_after
#         :param end_time: end_time in UTC expressed as "YYYY-MM-DDTHH:MM:SS.MS"
#         or a UTCDateTime, or a datetime
#         explicitely providing this paramter will override the time_utc_before
#         :param microquake_event_type: if true use microquake event types else
#         use Quakeml event types
#         :param kwargs:
#         :return:
#         """
#
#         if 'event_type' in kwargs.keys():
#             if microquake_event_type:
#                 event_types_lut = get_event_types(self.api_base_url,
#                                                   username=self.username,
#                                                   password=self.password)
#                 kwargs['event_type'] = event_types_lut[kwargs['event_type']]
#
#         if start_time:
#             kwargs['time_utc_after'] = str(start_time)
#         if end_time:
#             kwargs['time_utc_before'] = str(end_time)
#
#         events = []
#
#         try:
#             response = self.api_instance.api_v1_events_list(**kwargs)
#         except Exception as e:
#             logger.error(e)
#             return {}, events
#
#         for event in response.results:
#             events.append(RequestEvent(event.to_dict()))
#         query = response.next
#
#         while query:
#             re = requests.get(query, timeout=timeout)
#             if not re:
#                 break
#             response = re.json()
#             logger.info(f"page {response['current_page']} of "
#                         f"{response['total_pages']}")
#
#             query = response['next']
#
#             for event in response['results']:
#                 events.append(RequestEvent(event))
#
#         return response, events
#
#     def events_read(self, event_resource_id):
#         encoded_id = encode(event_resource_id)
#         if self.api_base_url[-1] != '/':
#             self.api_base_url + '/'
#         encoded_id = event_resource_id
#         url = self.api_base_url + 'events/' + encoded_id
#
#         try:
#             # api_response = self.api_instance.api_v1_events_read(encoded_id)
#             api_response = requests.get(url,
#                                         auth=(self.username, self.password))
#         except Exception as e:
#             logger.error(e)
#             return None, None
#
#         if not api_response:
#             return api_response, None
#
#         event = None
#         if api_response.json():
#             event = RequestEvent(api_response.json())
#
#         return api_response, event
#
#     def events_partial_update(self, event_resource_id, **kwargs):
#
#         data = kwargs
#         encoded_id = urllib.parse.quote(event_resource_id, safe='')
#         url = self.api_base_url + 'events/' + encoded_id
#         resp = requests.patch(url, json=data, auth=(self.username,
#                                                     self.password))
#         logger.info(resp)
#         return(resp)
#
#     def post_data_from_objects(self, network, event_id=None, cat=None,
#                                 stream=None, context=None, variable_length=None,
#                                 tolerance=0.5, send_to_bus=False):
#         response = post_data_from_objects(self.api_base_url, network,
#                                           event_id=event_id,
#                                           cat=cat, stream=stream,
#                                           context=context,
#                                           variable_length=variable_length,
#                                           tolerance=tolerance,
#                                           send_to_bus=send_to_bus,
#                                           username=self.username,
#                                           password=self.password)
#         return response
#
#         # try:
#         #     response = self.api_instance.api_v1_events_partial_update_0(body,
#         #                                  event_resource_id)
#         # except Exception as e:
#         #     logger.error(e)
#         # return response


def encode(resource_id):
    return urllib.parse.quote(resource_id, safe='')


def post_data_from_files(api_base_url, network, event_id=None, event_file=None,
                         mseed_file=None, mseed_context_file=None,
                         variable_length_stream_file=None,
                         tolerance=0.5):
    """
    Build request directly from objects
    :param api_base_url: base url of the API
    :param network: network code
    :param event_id: event_id (not required if an event is provided)
    :param event_file: path to a QuakeML file
    :param stream_file: path to a mseed seismogram file
    :param context_stream_file: path to a context seismogram file
    :param tolerance: Minimum time between an event already in the database
    and this event for this event to be inserted into the database. This is to
    avoid duplicates. event with a different
    <event_id> within the <tolerance> seconds of the current object will
    not be inserted. To disable this check set <tolerance> to None.
    :return: same as build_request_data_from_bytes
    """

    event = None,
    stream = None
    context_stream = None

    # read event

    if event_file is not None:
        event = read_events(event_file)

    # read waveform

    if mseed_file is not None:
        stream = read(mseed_file, format='MSEED')

    # read context waveform

    if mseed_context_file is not None:
        context_stream = read(mseed_context_file, format='MSEED')

    # read variable length waveform

    if variable_length_stream_file is not None:
        variable_length_stream = read(variable_length_stream_file,
                                      format='MSEED')

    return post_data_from_objects(api_base_url, network, event_id=event_id,
                                  cat=event, stream=stream,
                                  context=context_stream,
                                  variable_length=variable_length_stream,
                                  tolerance=tolerance)


def post_data_from_objects(api_base_url, network, event_id=None, cat=None,
                           stream=None, context=None, variable_length=None,
                           tolerance=0.5, send_to_bus=False, username=None,
                           password=None):
    """
    Build request directly from objects
    :param api_base_url: base url of the API
    :param event_id: event_id (not required if an event is provided)
    :param cat: microquake.core.event.Event or a
    microquake.core.event.Catalog containing a single event. If the catalog
    contains more than one event, only the first event, <catalog[0]> will be
    considered. Use catalog with caution.
    :param stream: event seismogram (microquake.core.Stream.stream)
    :param context: context seismogram trace (
    microquake.core.Stream.stream)
    :param variable_length: variable length seismogram trace (
    microquake.core.Stream.stream)
    :param tolerance: Minimum time between an event already in the database
    and this event for this event to be inserted into the database. This is to
    avoid duplicates. event with a different
    <event_id> within the <tolerance> seconds of the current object will
    not be inserted. To disable this check set <tolerance> to None.
    :param logger: a logging.Logger object
    :return: same as build_request_data_from_bytes
    """

    api_url = api_base_url + "events"

    if type(cat) is Catalog:
        cat = cat[0]
        logger.warning('a <microquake.core.event.Catalog> object was '
                       'provided, only the first element of the catalog will '
                       'be used, this may lead to an unwanted behavior')

    if cat is not None:
        event_time = cat.preferred_origin().time
        event_resource_id = str(cat.resource_id)
    else:
        if event_id is None:
            logger.warning('A valid event_id must be provided when no '
                           '<event> is not provided.')
            logger.info('exiting')

            return

        re = get_event_by_id(api_base_url, event_id)

        if re is None:
            logger.warning('request did not return any event with the '
                           'specified event_id. A valid event_id or an event '
                           'object must be provided to insert a stream or a '
                           'context stream into the database.')
            logger.info('exiting')

            return

        event_time = re.time_utc
        event_resource_id = str(re.event_resource_id)

    # data['event_resource_id'] = event_resource_id

    logger.info('processing event with resource_id: %s' % event_resource_id)

    base_event_file_name = str(event_time)
    files = {}
    # Test if there is an event within <tolerance> seconds in the database
    # with a different <event_id>. If so abort the insert

    if tolerance is not None:
        start_time = event_time - tolerance
        end_time = event_time + tolerance
        re_list = get_events_catalog(api_base_url, start_time, end_time)

        if re_list:
            logger.warning('Event found within % seconds of current event'
                           % tolerance)
            logger.warning('The current event will not be inserted into the'
                           ' data base')

            return

    files = prepare_data(cat=cat, stream=stream, context=context,
                         variable_length=variable_length)

    return post_event_data(api_url, network, event_resource_id, files,
                           send_to_bus=send_to_bus, username=username,
                           password=password)


def prepare_data(cat=None, stream=None, context=None, variable_length=None):

    files = {}
    base_event_file_name = str(uuid4())

    if cat is not None:
        logger.info('preparing event data')
        event_bytes = BytesIO()
        cat.write(event_bytes, format='QUAKEML')
        event_file_name = base_event_file_name + '.xml'
        event_bytes.name = event_file_name
        event_bytes.seek(0)
        files['event_file'] = event_bytes
        logger.info('done preparing event data')

    if stream is not None:
        logger.info('preparing waveform data')
        mseed_bytes = BytesIO()
        stream.write(mseed_bytes, format='MSEED')
        mseed_file_name = base_event_file_name + '.mseed'
        mseed_bytes.name = mseed_file_name
        mseed_bytes.seek(0)
        files['waveform_file'] = mseed_bytes
        logger.info('done preparing waveform data')

    if context is not None:
        logger.info('preparing context waveform data')
        mseed_context_bytes = BytesIO()
        context.write(mseed_context_bytes, format='MSEED')
        mseed_context_file_name = base_event_file_name + '.context_mseed'
        mseed_context_bytes.name = mseed_context_file_name
        mseed_context_bytes.seek(0)
        files['waveform_context_file'] = mseed_context_bytes
        logger.info('done preparing context waveform data')

    if variable_length is not None:
        logger.info('preparing variable length waveform data')
        mseed_variable_bytes = BytesIO()
        variable_length.write(mseed_variable_bytes, format='MSEED')
        mseed_variable_file_name = base_event_file_name + '.variable_mseed'
        mseed_variable_bytes.name = mseed_variable_file_name
        mseed_variable_bytes.seek(0)
        files['variable_size_waveform_file'] = mseed_variable_bytes
        logger.info('done preparing variable length waveform data')

    return files


def get_event_types(api_base_url, username=None, password=None):

    if api_base_url[-1] != '/':
        api_base_url += '/'

    url = api_base_url + 'inventory/microquake_event_types'
    if username:
        response = requests.get(url, auth=(username, password))
    else:
        response = requests.get(url, timeout=timeout)

    if not response:
        raise ConnectionError('API Connection Error')

    data = json.loads(response.content)
    dict_out = {}
    for d in data:
        dict_out[d['microquake_type']] = d['quakeml_type']

    return dict_out


def post_event_data(api_base_url, network, event_resource_id, request_files,
                    send_to_bus=False, username=None, password=None):
    # removing id from URL as no longer used
    # url = api_base_url + "%s/" % event_resource_id
    url = api_base_url
    logger.info('posting data on %s' % url)

    event_resource_id = encode(event_resource_id)

    if username is None:
        result = requests.post(url, data={"send_to_bus": send_to_bus},
                               files=request_files, timeout=timeout)
    else:
        result = requests.post(url, data={"send_to_bus": send_to_bus},
                               files=request_files, timeout=timeout,
                               auth=(username, password))
    logger.info(result)

    return result


def put_data_from_objects(api_base_url, network, cat=None, stream=None,
                          context=None, variable_length=None, username=None,
                          password=None):

    event_id = cat[0].resource_id.id

    base_url = api_base_url
    if base_url[-1] == '/':
        base_url = base_url[:-1]

    # check if the event type and event status has changed on the API

    # re = get_event_by_id(api_base_url, event_id)
    # try:
    #     cat[0].event_type = re.event_type
    #     cat[0].preferred_origin().evaluation_status = re.status
    # except AttributeError as e:
    #     logger.error(e)

    url = f'{base_url}/events/{event_id}'

    files = prepare_data(cat=cat, stream=stream, context=context,
                         variable_length=variable_length)

    logger.info(f'attempting to PUT catalog for event {event_id}')

    if username is None:
        response = requests.patch(url, files=files, timeout=timeout)
    else:
        response = requests.patch(url, files=files, timeout=timeout,
                                  auth=(username, password))

    logger.info(f'API responded with {response.status_code} code')
    return response


@deprecated
def get_events_catalog(api_base_url, start_time, end_time,
                       status='accepted', event_type=''):
    """
    return a list of events
    :param api_base_url:
    :param start_time:
    :param end_time:
    :param event_type:  example seismic_event,blast,drilling noise,open pit
    blast,quarry blast... etc
    :param status: Event status, accepted, rejected, accepted,rejected
    :return:
    """
    url = api_base_url + "catalog"

    # request work in UTC, time will need to be converted from whatever
    # timezone to UTC before the request is built.

    querystring = {"start_time": start_time, "end_time": end_time, "status":
                   status, "type": event_type}

    response = requests.request("GET", url, params=querystring,
                                timeout=timeout).json()

    events = []

    for event in response:
        events.append(RequestEvent(event))

    return events


def get_event_by_id(api_base_url, event_resource_id):
    # smi:local/e7021615-e7f0-40d0-ad39-8ff8dc0edb73
    url = api_base_url + "events/"
    # querystring = {"event_resource_id": event_resource_id}

    event_resource_id = encode(event_resource_id)
    response = requests.request("GET", url + event_resource_id, timeout=timeout)

    if response.status_code != 200:
        return None

    return RequestEvent(json.loads(response.content))


def post_signal_quality(
        api_base_url, request_data, username=None, password=None
):
    session = requests.Session()
    session.trust_env = False

    if username is None:
        return session.post("{}signal_quality".format(api_base_url),
                            json=request_data)
    else:
        return session.post("{}signal_quality".format(api_base_url),
                            json=request_data, auth=(username, password))


def reject_event(api_base_url, event_id):
    session = requests.Session()
    session.trust_env = False

    event_id = encode(event_id)

    return session.post("{}{}/interactive/reject".format(api_base_url,
                                                         event_id))


def get_catalog(api_base_url, start_time, end_time, event_type=None,
                status=None, **kwargs):
    """
    get the event catalogue from the API
    :param api_base_url: API base url
    :param start_time: start time as datetime if not time aware, UTC is
    assumed
    :param end_time: end time as datetime if not time aware, UTC is assumed
    :param time_zone:
    :param event_type: microquake event type
    :param status: event status acceptable values are ('preliminary',
    'reviewed', 'confirmed', 'final', 'rejected', accepted'). 'accepted'
    encompasses all status with the exception of 'rejected'
    :return: a RequestEvent object
    """

    if api_base_url[-1] != '/':
        api_base_url += '/'

    api_base_url += 'events'

    event_types = get_event_types(api_base_url)
    obs_event_type = None

    request_dict = {'time_utc_after': str(start_time),
                    'time_utc_before': str(end_time)}

    if event_type is not None:
        try:
            request_dict['event_type'] = event_types[event_type]
        except KeyError:
            logger.error(f'event type: {event_type} does not appear to be a '
                         f'valid event type for your system')

            raise KeyError
    else:
        event_type_str = ''
        for key in event_types.keys():
            event_type_str += f'{event_types[key]},'
        event_type_str = event_type_str[:-1]
        request_dict['event_type'] = event_type_str

    if status is not None:
        request_dict['status'] = status

    events = []

    tmp = urllib.parse.urlencode(request_dict)
    query = f'{api_base_url}?{tmp}'

    while query:
        re = requests.get(query, timeout=timeout)
        # from ipdb import set_trace; set_trace()()
        if not re:
            # logger.info('The API catalogue does not contain any events that'
            #             'corresponds to the request')
            break
        response = re.json()
        logger.info(f"page {response['current_page']} of "
                    f"{response['total_pages']}")

        query = response['next']

        for event in response.results:
            events.append(RequestEvent(event))

    return events


def get_catalog_modified(api_base_url, modification_start_time,
                         modification_end_time, event_type=None,
                         status=None):
    """
    get the event catalogue from the API
    :param api_base_url: API base url
    :param modification_start_time: start time as datetime if not time aware, UTC is
    assumed
    :param modification_end_time: end time as datetime if not time aware, UTC is assumed
    :param event_type: microquake event type
    :param status: event status acceptable values are ('preliminary',
    'reviewed', 'confirmed', 'final', 'rejected', accepted'). 'accepted'
    encompasses all status with the exception of 'rejected'
    :return: a RequestEvent object
    """

    if api_base_url[-1] != '/':
        api_base_url += '/'

    api_base_url += 'events'

    event_types = get_event_types(api_base_url)
    obs_event_type = None

    request_dict = {'modification_timestamp_after': str(
                     modification_start_time),
                    'modification_timestamp_before': str(
                     modification_end_time)}

    if event_type is not None:
        try:
            request_dict['event_type'] = event_types[event_type]
        except KeyError:
            logger.error(f'event type: {event_type} does not appear to be a '
                         f'valid event type for your system')

            raise KeyError
    else:
        event_type_str = ''
        for key in event_types.keys():
            event_type_str += f'{event_types[key]},'
        event_type_str = event_type_str[:-1]
        request_dict['event_type'] = event_type_str

    if status is not None:
        request_dict['status'] = status

    events = []

    tmp = urllib.parse.urlencode(request_dict)
    query = f'{api_base_url}?{tmp}'

    while query:
        re = requests.get(query, timeout=timeout)
        # from ipdb import set_trace; set_trace()()
        if not re:
            # logger.info('The API catalogue does not contain any events that'
            #             'corresponds to the request')
            break
        response = re.json()
        logger.info(f"page {response['current_page']} of "
                    f"{response['total_pages']}")

        query = response['next']

        for event in response['results']:
            events.append(RequestEvent(event))

    return events


def get_magnitude(api_base_url, magnitude_id):

    if api_base_url[-1] != '/':
        api_base_url += '/'

    request_url = f'{api_base_url}magnitudes/{magnitude_id}'

    # requests_dict = {'event_id': event_id,
    #                  'origin_id': origin_id}

    # tmp = urllib.parse.urlencode(requests_dict)
    # query = f'{_url}'

    return requests.get(request_url)
