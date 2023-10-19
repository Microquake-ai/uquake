import pytest
import urllib.request
import os
from uquake.core.util.requests import download_file_from_url
from uquake import read, read_events, read_inventory


@pytest.fixture(scope="session")
def st():
    mseed_url = "https://www.dropbox.com/scl/fi/8gm7pt2b4drmifqg02f17/" \
                "ffff9aa5fc9d5e83b630d35d83c8870c.mseed" \
                "?rlkey=2l9y84kw61eynonw1rn8zpc9y&dl=1"
    mseed_bytes = download_file_from_url(mseed_url)
    stream = read(mseed_bytes)
    return stream


@pytest.fixture(scope="session")
def cat():
    quakeml_url = "https://www.dropbox.com/scl/fi/tm8hd943g5mnl1po19q7q/" \
                  "ffff9aa5fc9d5e83b630d35d83c8870c.xml" \
                  "?rlkey=3iyxt4734f3hm76oljtx57dvs&dl=1"
    catalog_bytes = download_file_from_url(quakeml_url)

    cat = read_events(catalog_bytes)
    return cat


@pytest.fixture(scope="session")
def inv():
    stationxml_url = "https://www.dropbox.com/scl/fi/nw7j0j3o3vnf0s1mfb3ag/" \
                     "inventory.xml?rlkey=k3foc7x9uthpzw8lum92z284t&dl=1"
    inventory_bytes = download_file_from_url(stationxml_url)
    inv = read_inventory(inventory_bytes)

    return inv


# Fixture for mseed
@pytest.fixture(scope="session")
def mseed_file():
    mseed_url = "https://www.dropbox.com/scl/fi/8gm7pt2b4drmifqg02f17/" \
                "ffff9aa5fc9d5e83b630d35d83c8870c.mseed?r" \
                "lkey=2l9y84kw61eynonw1rn8zpc9y&dl=1"
    mseed_path = "tests_stream.mseed"

    # Download the file from the URL
    urllib.request.urlretrieve(mseed_url, mseed_path)

    # Yield the path to the downloaded file
    yield mseed_path

    # Clean up the file after all tests have finished
    os.remove(mseed_path)


# Fixture for stationxml
@pytest.fixture(scope="session")
def stationxml_file():
    stationxml_url = "https://www.dropbox.com/scl/fi/nw7j0j3o3vnf0s1mfb3ag/" \
                     "inventory.xml?rlkey=k3foc7x9uthpzw8lum92z284t&dl=1"
    stationxml_path = "tests_inventory.xml"

    # Download the file from the URL
    urllib.request.urlretrieve(stationxml_url, stationxml_path)

    # Yield the path to the downloaded file
    yield stationxml_path

    # Clean up the file after all tests have finished
    os.remove(stationxml_path)


# Fixture for quakeml
@pytest.fixture(scope="session")
def quakeml_file():
    quakeml_url = "https://www.dropbox.com/scl/fi/tm8hd943g5mnl1po19q7q/" \
                  "ffff9aa5fc9d5e83b630d35d83c8870c.xml" \
                  "?rlkey=3iyxt4734f3hm76oljtx57dvs&dl=1"
    quakeml_path = "tests_catalog.xml"

    # Download the file from the URL
    urllib.request.urlretrieve(quakeml_url, quakeml_path)

    # Yield the path to the downloaded file
    yield quakeml_path

    # Clean up the file after all tests have finished
    os.remove(quakeml_path)
