import pytest
from uquake.nlloc.nlloc import Srces, Site


sites = []
for i in range(0, 10):
    sites.append(Site(f'test_long_name{i: 02d}', 0, 0, 0))

srces = Srces(sites=sites)


def test_create_create_srces():
    print(srces.site_code_mapping)
    assert srces is not None


def test_add_site_srces():
    srces.add_site(f'shtest', 0, 0, 0)
    assert len(sites) == 11


if __name__ == "__main__":
    pytest.main()
