import io
import warnings
import base64
from obspy.core import AttribDict
import numpy as np


def _init_handler(self, obspy_obj, **kwargs):
    """
    Handler to initialize microquake objects which
    inherit from ObsPy class. If obspy_obj is none,
    Kwargs is expected to be a mix of obspy kwargs
    and microquake kwargs specified by the hardcoded
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
    else:
        extra_kwargs = pop_keys_matching(kwargs, self.extra_keys)
        super(type(self), self).__init__(**kwargs)  # init obspy_origin args
        [self.__setattr__(k, v) for k, v in extra_kwargs.items()]  # init
        # extra_args


def _init_from_obspy_object(mquake_obj, obspy_obj):
    """
    When initializing microquake object from obspy_obj
    checks attributes for lists of obspy objects and
    converts them to equivalent microquake objects.
    """

    class_equiv = {obsevent.Pick: Pick,
                   obsevent.Arrival: Arrival,
                   obsevent.Origin: Origin,
                   obsevent.Magnitude: Magnitude}

    for key, val in obspy_obj.__dict__.items():
        if type(val) == list:
            out = []
            for item in val:
                itype = type(item)
                if itype in class_equiv:
                    out.append(class_equiv[itype](item))
                else:
                    out.append(item)
            mquake_obj.__setattr__(key, out)
        else:
            mquake_obj.__setattr__(key, val)


def _set_attr_handler(self, name, value, namespace='MICROQUAKE'):
    """
    Generic handler to set attributes for microquake objects
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
                self.__setattr__(key, parse_string_val(adict.value))
            else:
                self['extra'][key] = adict
    else:
        raise KeyError(name)


def _set_attr_handler2(self, name, value, namespace='MICROQUAKE'):
    """
    Generic handler to set attributes for microquake objects
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


def _set_attr_handler_inventory(self, name, value, namespace='MICROQUAKE'):
    """
    Generic handler to set attributes for microquake objects
    which inherit from ObsPy objects. If 'name' is not in
    default keys then it will be set in self['extra'] dict. If
    'name' is not in default keys but in the self.extra_keys
    then it will also be set as a class attribute. When loading
    extra keys from quakeml file, those in self.extra_keys will
    be set as attributes.
    """

    #  use obspy default setattr for default keys
    if name in self.extra_keys:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.__dict__[name] = value
        if type(value) is np.ndarray:
            value = "npy64_" + array_to_b64(value)
        elif type(value) is str:
            if "npy64_" in value:
                value.replace("npy64_", "")
                b64_to_array(value)
        self.__dict__['extra'][name] = {'value': value, 'namespace': namespace}
    # recursive parse of 'extra' args when constructing uquake from obspy
    elif name == 'extra':
        if 'extra' not in self.__dict__.keys():  # hack for deepcopy to work
            self.__dict__['extra'] = {}
        for key, adict in value.items():
            if key in self.extra_keys:
                self.__setattr__(key, parse_string_val(adict.value))
            else:
                self.__dict__['extra'][key] = adict
    else:
        try:
            super(type(self), self).__setattr__(name, value)
        except Exception as e:
            raise KeyError


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def pop_keys_matching(dict_in, keys):
    # Move keys from dict_in to dict_out
    dict_out = {}
    for key in keys:
        if key in dict_in:
            dict_out[key] = dict_in.pop(key)
    return dict_out


def array_to_b64(array):
    output = io.BytesIO()
    np.save(output, array)
    content = output.getvalue()
    encoded = base64.b64encode(content).decode('utf-8')
    return encoded


def b64_to_array(b64str):
    arr = np.load(io.BytesIO(base64.b64decode(b64str)))
    return arr


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