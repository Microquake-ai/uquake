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


def array_to_b64(array):
    output = io.BytesIO()
    np.save(output, array)
    content = output.getvalue()
    encoded = base64.b64encode(content).decode('utf-8')
    return encoded


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def b64_to_array(b64str):
    arr = np.load(io.BytesIO(base64.b64decode(b64str)))
    return arr


def pop_keys_matching(dict_in, keys):
    # Move keys from dict_in to dict_out
    dict_out = {}
    for key in keys:
        if key in dict_in:
            dict_out[key] = dict_in.pop(key)
    return dict_out


def parse_string_val(val, arr_flag='npy64_'):
    """
    Parse extra args in quakeML which are all stored as string.
    """
    if val is None:  # hack for deepcopy ignoring isfloat try-except
        val = None
    elif type(val) == AttribDict:
        val = val
    elif isfloat(val):
        val = float(val)
    elif str(val) == 'None':
        val = None
    elif val[:len(arr_flag)] == 'npy64_':
        val = b64_to_array(val[len(arr_flag):])
    return val


def _set_attr_handler(self, name, value, namespace=namespace):
    """
    Generic handler to set attributes for uquake objects
    which inherit from ObsPy objects. If 'name' is not in
    default keys then it will be set in self['extra'] dict. If
    'name' is not in default keys but in the self.extra_keys
    then it will also be set as a class attribute. When loading
    extra keys from quakeml file, those in self.extra_keys will
    be set as attributes.
    """

    #  use obspy default setattr for default keys
    if name in self.defaults.keys():
        self.__dict__[name] = value
        if isinstance(self.__dict__[name], ResourceIdentifier):
            self.__dict__[name] = ResourceIdentifier(id=value.id)
        # super(type(self), self).__setattr__(name, value)
    elif name in self.extra_keys:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self[name] = value
        if hasattr(value, 'to_json'):
            value = value.to_json()
        elif isinstance(value, np.ndarray):
            value = "npy64_" + array_to_b64(value)
        elif type(value) is str:
            if "npy64_" in value:
                value.replace("npy64_", "")
                b64_to_array(value)
        self['extra'][name] = {'value': value, 'namespace': namespace}
    # recursive parse of 'extra' args when constructing uquake from obspy
    elif name == 'extra':
        if 'extra' not in self:  # hack for deepcopy to work
            self['extra'] = {}
        for key, adict in value.items():
            if key in self.extra_keys:
                self.__dict__[key] = parse_string_val(adict.value)
            else:
                self['extra'][key] = adict
    else:
        raise KeyError(name)


def _init_handler(self, obspy_obj, **kwargs):
    """
    Handler to initialize uquake objects which
    inherit from ObsPy class. If obspy_obj is none,
    Kwargs is expected to be a mix of obspy kwargs
    and uquake kwargs specified by the hardcoded
    extra_keys.
    """

    if obspy_obj and len(kwargs) > 0:
        raise AttributeError("Initialize from either \
                              obspy_obj or kwargs, not both")

    # default initialize the extra_keys args to None
    self['extra'] = {}
    [self.__setattr__(key, None) for key in self.extra_keys]

    if obspy_obj:
        _init_from_obspy_object(self, obspy_obj)

        if 'resource_id' in obspy_obj.__dict__.keys():
            rid = obspy_obj.resource_id.id
            self.resource_id = ResourceIdentifier(id=rid,
                                                  referred_object=self)
    else:
        extra_kwargs = pop_keys_matching(kwargs, self.extra_keys)
        super(type(self), self).__init__(**kwargs)  # init obspy_origin args
        [self.__setattr__(k, v) for k, v in extra_kwargs.items()]  # init
        # extra_args




