import unittest
from uquake.core import event
from uquake.core import coordinates
from importlib import reload
import numpy as np
from datetime import datetime, timedelta
import string
import random


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

    def test_origin_serialization(self):
        pass


def generate_uncertainty_point_cloud():
    locations = np.random.randn(3, 100)
    probabilities = np.random.rand(3, 100)
    coordinate_system = coordinates.CoordinateSystem('ENU')

    uncertainty_point_cloud = event.UncertaintyPointCloud(locations, probabilities,
                                                          coordinate_system)

    return uncertainty_point_cloud


def generate_ray():

    waveform_id = generate_waveform_id()

    nodes = np.random.randn(3, 1000)

    ray = event.Ray(nodes, waveform_id, event.ResourceIdentifier(), 'P',
                    travel_time=10.0,
                    takeoff_angle=10, azimuth=2,
                    velocity_model_id=event.ResourceIdentifier(),
                    coordinate_system=coordinates.CoordinateSystem('ENU'))

    return ray


def generate_pick(origin_time=datetime(2010, 1, 1, 0, 0, 0)):
    snr = np.random.rand() * 20
    azimuth = np.random.rand() * 360 - 180
    incidence_angle = np.random.rand() * 90
    planarity = np.random.rand()
    linearity = np.random.rand()

    time = origin_time + timedelta(seconds=np.random.rand() / 10)
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


def generate_event(n_origin=1, n_picks_per_origin=20):

    picks = []
    for i in range(0, n_origin):
        arrivals = []
        for j in range(0, n_picks_per_origin):
            pick = generate_pick()
            picks.append(pick)
            arrivals.append(generate_arrival(pick))

        event.Origin()

    pass


if __name__ == '__main__':
    unittest.main()
