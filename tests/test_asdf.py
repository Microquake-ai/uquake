import unittest
import os
from uquake.synthetic import event, inventory, stream
from uquake.core import data_exchange
from importlib import reload
from time import time


class TestDataExchange(unittest.TestCase):

    def tearDown(self):
        try:
            os.remove('test.asdf')
        except FileNotFoundError:
            pass

    def test_data_exchange(self):
        reload(stream)

        cat = event.generate_catalog()
        inv = inventory.generate_inventory()
        st = stream.generate_waveform(inv,
                                      start_time=cat[0].preferred_origin().time - 0.01)

        t0 = time()
        mde = data_exchange.MicroseismicDataExchange(stream=st, catalog=cat,
                                                     inventory=inv)
        t1 = time()
        self.assertLess(t1 - t0, 1, 'Creation time exceeds 1 second')

        t0 = time()
        mde.write('test.asdf', waveform_tag='test_waveform')
        t1 = time()
        self.assertLess(t1 - t0, 10, 'Write time exceeds 10 second')

        t0 = time()
        mde2 = data_exchange.MicroseismicDataExchange.read('test.asdf', 'test_waveform')
        t1 = time()
        self.assertLess(t1 - t0, 10, 'Read time exceeds 10 second')

        self.assertEqual(mde.inventory, mde2.inventory)
        self.assertListEqual(mde.stream[0].data.tolist(), mde2.stream[0].data.tolist())
        self.assertEqual(mde.catalog[0].preferred_origin().time,
                         mde2.catalog[0].preferred_origin().time)


if __name__ == '__main__':
    unittest.main()





