import unittest
from uquake.core import event
from importlib import reload
from uquake.synthetic.event import (generate_catalog, generate_ray,
                                    generate_uncertainty_point_cloud)
reload(event)


class TestEventMethods(unittest.TestCase):

    # def tearDown(self):
    #     # Remove the test.xml file after the test
    #     if os.path.exists('test.xml'):
    #         os.remove('test.xml')

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
        cat2 = event.read_events('test.xml')

        assert cat[0].preferred_origin().arrivals[0].pick.time == \
               cat2[0].preferred_origin().arrivals[0].pick.time


if __name__ == '__main__':
    unittest.main()
