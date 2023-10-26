import warnings
from obspy.core.event import ResourceIdentifier
import numpy as np
import base64
import io
from obspy.core.util import AttribDict


namespace = 'https://microquake.ai/xml/station/1'


def set_extra(self, name, value, namespace='mq'):
    self['extra'][name] = AttribDict({'value': value, 'namespace': namespace})


def get_extra(self, name):
    return self['extra'][name].value







