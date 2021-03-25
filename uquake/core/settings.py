import os
from dynaconf import LazySettings


class Settings(LazySettings):
    def __init__(self):
        """
        Init function currently just initializes the object allowing
        """
        if "SPP_CONFIG" in os.environ:
            # keep this as legacy behavior
            config_dir = os.environ['UQUAKE_CONFIG']
        else:
            config_dir = os.getcwd()

        dconf = {}
        dconf.setdefault('ENVVAR_PREFIX_FOR_DYNACONF', 'UQUAKE')

        env_prefix = '{0}_ENV'.format(
            dconf['ENVVAR_PREFIX_FOR_DYNACONF']
        )  # SPP_ENV

        dconf.setdefault(
            'ENV_FOR_DYNACONF',
            os.environ.get(env_prefix, 'DEV').upper()
        )

        # This was an incredibly odd fix, the base settings.toml needs to be on
        # top of the list otherwise you will not be able to modify the settings
        # downstream
        default_paths = (
            "settings.py,.secrets.py,"
            "settings.toml,settings.tml,.secrets.toml,.secrets.tml,"
            "settings.yaml,settings.yml,.secrets.yaml,.secrets.yml,"
            "settings.ini,settings.conf,settings.properties,"
            "connectors.toml,connectors.tml,.connectors.toml,.connectors.tml,"
            "connectors.json,"
            ".secrets.ini,.secrets.conf,.secrets.properties,"
            "settings.json,.secrets.json"
        )

        dconf['SETTINGS_FILE_FOR_DYNACONF'] = default_paths
        dconf['ROOT_PATH_FOR_DYNACONF'] = config_dir

        super().__init__(**dconf)


settings = Settings()
