import unittest
from uquake.core import inventory
from uquake.synthetic.inventory import generate_inventory
import os


class TestInventoryMethods(unittest.TestCase):

    def tearDown(self):
        # Remove the test.xml file after the test
        if os.path.exists('test.xml'):
            os.remove('test.xml')

    def test_inventory_serialization(self):

        inv = generate_inventory()
        # Write and read back
        inv.write('test.xml')
        inv2 = inventory.read_inventory('test.xml')

        # Check equality
        self.assertEqual(inv, inv2)


if __name__ == '__main__':
    unittest.main()
