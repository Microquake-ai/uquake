import unittest
from uquake.core.util.requests import download_file_from_url
from uquake import read, read_events, read_inventory
import os
from uquake.core.mde import MicroseismicDataExchange


class MicroseismicDataExchangeTest(unittest.TestCase):

    def setUp(self):
        self.mseed_url = "https://www.dropbox.com/scl/fi/8gm7pt2b4drmifqg02f17/" \
                         "ffff9aa5fc9d5e83b630d35d83c8870c.mseed" \
                         "?rlkey=2l9y84kw61eynonw1rn8zpc9y&dl=1"
        self.mseed_bytes = download_file_from_url(self.mseed_url)
        self.st = read(self.mseed_bytes)

        self.quakeml_url = "https://www.dropbox.com/scl/fi/tm8hd943g5mnl1po19q7q/" \
                           "ffff9aa5fc9d5e83b630d35d83c8870c.xml" \
                           "?rlkey=3iyxt4734f3hm76oljtx57dvs&dl=1"
        self.catalog_bytes = download_file_from_url(self.quakeml_url)
        self.cat = read_events(self.catalog_bytes)

        self.stationxml_url = "https://www.dropbox.com/scl/fi/nw7j0j3o3vnf0s1mfb3ag/" \
                              "inventory.xml?rlkey=k3foc7x9uthpzw8lum92z284t&dl=1"
        self.inventory_bytes = download_file_from_url(self.stationxml_url)
        self.inv = read_inventory(self.inventory_bytes)

    def test_microseismic_data_exchange(self):


        exchange_object = MicroseismicDataExchange(stream=self.st, catalog=self.cat,
                                                   inventory=self.inv)
        test_file_path = "test.asdf"
        exchange_object.write(test_file_path, 'tag')

        # Uncomment the following lines as per your requirement
        # read_back_object = MicroseismicDataExchange.read(test_file_path)
        # self.assertEqual(self.st, read_back_object.stream)
        # self.assertEqual(self.cat, read_back_object.catalog)
        # self.assertEqual(self.inv, read_back_object.inventory)

        os.remove(test_file_path)


if __name__ == '__main__':
    unittest.main()

