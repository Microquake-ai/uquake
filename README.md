# README #

Microquake is an open source package licensed under the [GNU general Public License, version 3 (GPLv3)](http://www.gnu.org/licenses/gpl-3.0.html). Microquake is an extension of Obspy for the processing of microseismic data

### Development

```
pip install poetry
poetry config http-basic.microquake {user} {password}
poetry install
```

Running tests

```
poetry run pytest
```

### How to release a new version

```
poetry version
git add pyproject.toml
git commit -m "bump version"
git tag newversion
git push --tags
git push
```

### Automatic tagging and releasing

By adding the following command to your git config you can bump and release a new version with one command

```
git config --global alias.patch '!poetry version patch && version=$(poetry version | awk "{print \$NF}") && git add pyproject.toml && git commit -m "Bumping version to $version" && git push && git tag "$version" && git push --tags && poetry build && poetry publish'

git config --global alias.minor '!poetry version minor && version=$(poetry version | awk "{print \$NF}") && git add pyproject.toml && git commit -m "Bumping version to $version" && git push && git tag "$version" && git push --tags && poetry build && poetry publish'

git config --global alias.major '!poetry version major && version=$(poetry version | awk "{print \$NF}") && git add pyproject.toml && git commit -m "Bumping version to $version" && git push && git tag "$version" && git push --tags && poetry build && poetry publish'
```

After running the above command you may release a new version with:

```
git bump
```

### System prerequisites

To run the interloc module the FFTw library must be installed on the system. To install the library, simply run

```
sudo apt-get install libfftw3-3
```

