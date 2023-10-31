import unittest
from uquake.core import event
from uquake.core import coordinates
from importlib import reload
import numpy as np
import datetime
import string
import random
from typing import List
import os
reload(event)


class TestEventMethods(unittest.TestCase):

    def test_uncertainty_point_cloud_serialization(self):
        reload(event)

        uncertainty_point_cloud = generate_uncertainty_point_cloud()

        upc_json = uncertainty_point_cloud.to_json()
        uncertainty_point_cloud2 = event.UncertaintyPointCloud.from_json(upc_json)

        self.assertEqual(uncertainty_point_cloud, uncertainty_point_cloud2)

    def test_ray_serialization(self):
        reload(event)

        ray = generate_ray()

        ray2 = event.Ray.from_json(ray.to_json())

        self.assertEqual(ray, ray2)

    def test_read_write(self):
        cat = generate_catalog()
        cat.write('test.xml')

    # def tearDown(self):
    #     # Remove the test.xml file after the test
    #     if os.path.exists('test.xml'):
    #         os.remove('test.xml')


def generate_uncertainty_point_cloud():
    locations = np.random.randn(3, 100)
    probabilities = np.random.rand(3, 100)
    coordinate_system = coordinates.CoordinateSystem('ENU')

    uncertainty_point_cloud = event.UncertaintyPointCloud(locations, probabilities,
                                                          coordinate_system)

    return uncertainty_point_cloud


def generate_ray():

    waveform_id = generate_waveform_id()

    nodes = np.random.randn(1000, 3)

    ray = event.Ray(nodes, waveform_id, event.ResourceIdentifier(), 'P',
                    travel_time=10.0,
                    takeoff_angle=10, azimuth=2,
                    velocity_model_id=event.ResourceIdentifier(),
                    coordinate_system=coordinates.CoordinateSystem('ENU'))

    return ray


def generate_ray_collection(n_rays):
    ray_collection = event.RayCollection()
    for i in range(n_rays):
        ray_collection.append(generate_ray())

    return ray_collection


def generate_uncertainty_point_cloud(n_points=100):
    location = (np.random.rand(3, 100) * 100).tolist()
    probabilities = [random.random() for i in range(n_points)]

    return event.UncertaintyPointCloud(location, probabilities)


def generate_pick(origin_time=datetime.datetime(2010, 1, 1, 0, 0, 0)):
    snr = np.random.rand() * 20
    azimuth = np.random.rand() * 360 - 180
    incidence_angle = np.random.rand() * 90
    planarity = np.random.rand()
    linearity = np.random.rand()

    time = origin_time + datetime.timedelta(seconds=np.random.rand() / 10)
    time_error = np.random.rand() / 100
    waveform_id = generate_waveform_id()
    filter_id = event.ResourceIdentifier()
    method_id = event.ResourceIdentifier()
    onset = 'Emergent'
    phase_hint = 'P'
    polarity = 'Positive'
    evaluation_mode = 'manual'
    evaluation_status = 'reviewed'
    comments = [event.base.Comment('test')]
    creation_info = event.base.CreationInfo('creation info test')

    return event.Pick(snr=snr, azimuth=azimuth, incidence_angle=incidence_angle,
                      planarity=planarity, linearity=linearity, time=time,
                      time_error=time_error, waveform_id=waveform_id,
                      filter_id=filter_id, method_id=method_id, onset=onset,
                      phase_hint=phase_hint, polarity=polarity,
                      evaluation_mode=evaluation_mode,
                      evaluation_status=evaluation_status,
                      comments=comments, creation_info=creation_info)


def generate_arrival(pick):

    kwargs = {'pick_id': pick.resource_id,
              'phase': pick.phase_hint,
              'time_correction': np.random.rand() / 100,
              'azimuth': np.random.randn() * 180,
              'takeoff_angle': np.random.randn() * 90,
              'time_residual': np.random.randn() / 100,
              'velocity_mode_id': event.ResourceIdentifier(),
              'comments': [event.base.Comment('test')],
              'creation_info': event.base.CreationInfo('creation info test')
              }
    return event.Arrival(**kwargs)


def generate_waveform_id():
    station_code = 'STA' + ''.join(random.choices(string.digits, k=2))
    location_code = ''.join(random.choices(string.ascii_uppercase, k=2))
    channel_code = ''.join(random.choices(string.ascii_uppercase, k=3))
    resource_uri = event.ResourceIdentifier()

    return event.WaveformStreamID(
        network_code='XX',
        station_code=station_code,
        location_code=location_code,
        channel_code=channel_code,
        resource_uri=resource_uri
    )


# def generate_event(n_origin=1, n_picks_per_origin=20):
#
#     picks = []
#     for i in range(0, n_origin):
#         arrivals = []
#         for j in range(0, n_picks_per_origin):
#             pick = generate_pick()
#             picks.append(pick)
#             arrivals.append(generate_arrival(pick))
#
#         event.Origin()
#
#     pass


def generate_origin(origin_time: datetime = datetime.datetime(2010, 1, 1, 0, 0, 0),
                    n_picks=20) -> (event.Origin, List[event.Pick]):
    arrivals = []
    picks = []
    for i in range(0, n_picks):
        pick = generate_pick(origin_time)
        picks.append(pick)
        arrivals.append(generate_arrival(pick))

    x = np.random.rand() * 1000
    y = np.random.rand() * 1000
    z = np.random.rand() * 1000
    coords = coordinates.Coordinates(
        x, y, z, coordinate_system=coordinates.CoordinateSystem.ENU)

    origin_uncertainty = event.OriginUncertainty(associated_phase_count=len(arrivals),
                                                 used_phase_count=len(arrivals))

    evaluation_mode = random.choice(list(event.header.EvaluationMode))
    evaluation_status = random.choice(list(event.header.EvaluationStatus))

    origin = event.Origin(time=origin_time,
                          time_error=event.QuantityError(uncertainty=0),
                          coordinates=coords,
                          rays=generate_ray_collection(n_picks),
                          uncertainty_point_cloud=generate_uncertainty_point_cloud(),
                          time_fixed=False,
                          velocity_model_id=event.ResourceIdentifier(),
                          arrivals=arrivals,
                          origin_uncertainty=origin_uncertainty,
                          origin_type=event.header.OriginType.hypocenter,
                          region='crusher',
                          evaluation_mode=evaluation_mode,
                          evaluation_status=evaluation_status,
                          comments=[event.Comment('a comment')],
                          creation_info=event.CreationInfo('creation info'))

    return origin, picks


def generate_magnitude(origin):

    evaluation_mode = random.choice(list(event.header.EvaluationMode))
    evaluation_status = random.choice(list(event.header.EvaluationStatus))

    magnitude = event.Magnitude(mag=random.randrange(-2, 3),
                                magnitude_type='Mw',
                                corner_frequency_p=random.randrange(1, 1000, 10),
                                corner_frequency_s=random.randrange(1, 1000, 10),
                                corner_frequency=random.randrange(1, 1000, 10),
                                corner_frequency_error=random.randrange(0, 100),
                                corner_frequency_p_error=random.randrange(0, 100),
                                corner_frequency_s_error=random.randrange(0, 100),
                                energy_p=random.random() * 100,
                                energy_s=random.random() * 100,
                                energy_p_error=random.random(),
                                energy_s_error=random.random(),
                                origin_id=origin.resource_id,
                                evaluation_mode=evaluation_mode,
                                evaluation_status=evaluation_status)

    return magnitude


def generate_event(n_origins=5):
    origins = []
    magnitudes = []
    picks = []
    for i in range(n_origins):
        origin, pks = generate_origin()
        magnitude = generate_magnitude(origin)

        picks += pks
        origins.append(origin)
        magnitudes.append(magnitude)

    evt = event.Event(origins=origins, magnitudes=magnitudes, picks=picks,
                      preferred_origin_id=origin.resource_id,
                      preferred_magnitude_id=magnitude.resource_id)

    return evt


def generate_catalog(n_events=1):
    events = []
    for i in range(n_events):
        events.append(generate_event())

    cat = event.Catalog(events=events)
    return cat


cat = generate_catalog()
cat.write('test2.xml')


if __name__ == '__main__':
    unittest.main()
