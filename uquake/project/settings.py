import os
from dynaconf import LazySettings
from pathlib import Path


class Settings(LazySettings):
    def __init__(self, settings_location):
        """
        Init function currently just initializes the object allowing
        """

        config_dir = Path(settings_location)

        dconf = {}
        dconf.setdefault('ENVVAR_PREFIX_FOR_DYNACONF', 'UQUAKE')

        env_prefix = '{0}_ENV'.format(
            dconf['ENVVAR_PREFIX_FOR_DYNACONF']
        )  # SPP_ENV

        dconf.setdefault(
            'ENV_FOR_DYNACONF',
            os.environ.get(env_prefix, 'DEV').upper()
        )

        # This was an incredibly odd fix, the base settings_template.toml needs to be on
        # top of the list otherwise you will not be able to modify the settings
        # downstream
        default_paths = [str(config_dir / 'settings.toml')]
        #     "settings.toml, settings.py,.secrets.py,"
        #     "settings.toml, settings.tml,.secrets.toml,.secrets.tml,"
        #     "settings.yaml,settings.yml,.secrets.yaml,.secrets.yml,"
        #     "settings.ini,settings.conf,settings.properties,"
        #     "connectors.toml,connectors.tml,.connectors.toml,.connectors.tml,"
        #     "connectors.json,"
        #     ".secrets.ini,.secrets.conf,.secrets.properties,"
        #     "settings.json,.secrets.json"
        # )

        # default_paths = str(settings_location)

        dconf['SETTINGS_FILE_FOR_DYNACONF'] = os.path.join(settings_location,
                                                           'settings.toml')
        dconf['ROOT_PATH_FOR_DYNACONF'] = settings_location
        # dconf['INCLUDES_FOR_DYNACONF'] = os.path.join(config_dir,
        #                                               'settings.toml')

        print(dconf)

        super().__init__(**dconf)

