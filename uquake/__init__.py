__package_name__ = 'uquake'
__version__ = '0.3.1'
__import__('pkg_resources').declare_namespace(__name__)
from .core import read, read_inventory, read_events
