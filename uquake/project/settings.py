import os
from dynaconf import LazySettings


class Settings(LazySettings):
    def __init__(self, settings_file):
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

        default_paths = settings_file

        dconf['SETTINGS_FILE_FOR_DYNACONF'] = default_paths
        dconf['ROOT_PATH_FOR_DYNACONF'] = config_dir

        super().__init__(**dconf)

