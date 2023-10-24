import warnings
from obspy.core.event import ResourceIdentifier
import numpy as np
import base64
import io
from obspy.core.util import AttribDict
from uquake import __package_name__ as ns


def set_extra(self, name, value, namespace=ns):
    self.extra[name] = AttribDict({'value': value, 'namespace': namespace})


def get_extra(self, name):
    return self.extra[name].value

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


def _set_attr_handler(self, name, value, namespace='UQUAKE'):
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
        if type(value) is np.ndarray:
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


def _set_attr_handler2(self, name, value, namespace='UQUAKE'):
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
        super(type(self), self).__setattr__(name, value)
    # recursive parse of extra args when constructing uquake from obspy
    elif name == 'extra':
        if 'extra' not in self:  # hack for deepcopy to work
            self['extra'] = {}
        for key, adict in value.items():
            self.__setattr__(key, parse_string_val(adict.value))
    else:  # branch for extra keys
        if name in self.extra_keys:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self[name] = value
        if type(value) is np.ndarray:
            value = "npy64_" + array_to_b64(value)
        self['extra'][name] = {'value': value, 'namespace': namespace}



