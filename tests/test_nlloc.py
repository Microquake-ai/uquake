# Copyright (C) 2023, Jean-Philippe Mercier
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
