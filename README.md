# README #

&mu;Quake is an open source package licensed under the [GNU general Public License, version 3 (GPLv3)](http://www.gnu.org/licenses/gpl-3.0.html). &mu;Quake is an extension of Obspy for the processing of microseismic data

### Development

We recommend the use of Poetry for development purposes.

```
pip install poetry
poetry install
```

Running tests

```
poetry run pytest
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
git patch
git minor
git major
```

