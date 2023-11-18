from uquake.synthetic import event, inventory, stream
from uquake.core import data_exchange
from importlib import reload
from time import time
from uquake.io.data_exchange.zarr import write_zarr, read_zarr
import shutil


def write_read_test_zarr():
    # Clean up any existing file
    try:
        shutil.rmtree('test.zarr')
    except FileNotFoundError:
        pass

    # Reload stream to ensure a clean state
    reload(stream)

    # Generate synthetic catalog, inventory, and stream
    cat = event.generate_catalog()
    inv = inventory.generate_inventory()
    st = stream.generate_waveform(inv, start_time=cat[0].preferred_origin().time - 0.01)

    # Create MicroseismicDataExchange object
    mde = data_exchange.MicroseismicDataExchange(stream=st, catalog=cat, inventory=inv)

    # Write to Zarr
    write_start_time = time()
    write_zarr('test.zarr', mde)
    write_end_time = time()
    print(f"Write Time: {write_end_time - write_start_time} seconds")

    # Read from Zarr
    read_start_time = time()
    mde2 = read_zarr('test.zarr')
    read_end_time = time()
    print(f"Read Time: {read_end_time - read_start_time} seconds")

    # Assertions and validations
    if mde.inventory == mde2.inventory:
        print("Inventory match: Passed")
    else:
        print("Inventory match: Failed")

    if mde.stream[0].data.tolist() == mde2.stream[0].data.tolist():
        print("Stream data match: Passed")
    else:
        print("Stream data match: Failed")

    if mde.catalog[0].preferred_origin().time == mde2.catalog[0].preferred_origin().time:
        print("Catalog preferred origin time match: Passed")
    else:
        print("Catalog preferred origin time match: Failed")

    # Clean up
    try:
        shutil.rmtree('test.zarr')
    except FileNotFoundError:
        pass


if __name__ == '__main__':
    write_read_test_zarr()
