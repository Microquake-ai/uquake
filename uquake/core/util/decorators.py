import os
import time
from functools import wraps
from io import BytesIO
from pathlib import Path
import numpy

from ..logging import logger


def expand_input_format_compatibility(func):
    """
    Decorator to enhance the compatibility of a function that expects a file path
    string or a file-like object as its first argument.

    If the first argument is provided as bytes, it is wrapped in a BytesIO. If it
    is provided as a pathlib.Path instance, it is converted to a string representation
    of the path.

    Parameters
    ----------
    func : callable
        The function to be decorated. It is expected that the function's first argument
        is a file path string or a file-like object.

    Returns
    -------
    callable
        The decorated function with enhanced input compatibility.

    Example
    -------
    @expand_input_format_compatibility
    def read_data(file):
        pass

    read_data(b"some_bytes_data")  # Bytes are wrapped into BytesIO
    read_data(Path("/path/to/file"))  # Path is converted to string

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Assuming the first argument to the function is the input in question
        input_data = args[0]

        # Check if input is bytes, if yes, wrap it in BytesIO
        if isinstance(input_data, bytes):
            args = (BytesIO(input_data),) + args[1:]

        # Check if input is a Path instance, if yes, convert to string
        elif isinstance(input_data, Path):
            args = (str(input_data),) + args[1:]

        return func(*args, **kwargs)

    return wrapper


def compress_file(func):
    """
    Decorator used to write compressed file to disk using gz or bz2 protocols
    """

    def wrapped_func(filename, *args, **kwargs):
        if not isinstance(filename):
            return func(filename, *args, **kwargs)

        if filename.endswith('gz'):
            import gzip
            f = gzip.open(filename, 'w')
        elif filename.endswith('bz2'):
            import bz2
            f = bz2.BZ2File(filename, 'w')
        elif filename.endswith('zip'):
            print('Zip protocol is not supported')

        else:
            f = filename

    return func(f, *args, **kwargs)


class buggy(object):
    """
    This is a decorator which can be used to mark functions
    as not implemented. It will result in a warning being emitted when
    the function is called.
    """

    def __init__(self, date):
        self.date = date

    def __call__(self, fct):
        @wraps(fct)
        def wrapper(*args, **kwargs):
            logger.warning(
                "%s is buggy, use at your own risk. Expected fix : %s" % (
                fct.__name__, self.date))
            try:
                return_value = fct(*args, **kwargs)
            except Exception:
                print(
                    "%s Crashed : Didn't i told you it was buggy ?" % fct.__name__)
                raise

            return return_value

        return wrapper


class unimplemented(object):
    """
    This is a decorator which can be used to mark functions
    as not implemented. It will result in a warning being emitted and
    the function is never called.
    """

    def __init__(self, date):
        self.date = date

    def __call__(self, fct):
        @wraps(fct)
        def wrapper(*args, **kwargs):
            logger.warning("%s is not_implemented ... now. Expected : %s " % (
            fct.__name__, self.date))

        return wrapper


class broken(object):
    """
    This is a decorator which can be used to mark functions
    as broken. It will result in a warning being emitted and
    the function is never called.
    """

    def __init__(self, date):
        self.date = date

    def __call__(self, fct):
        @wraps(fct)
        def wrapper(*args, **kwargs):
            logger.critical(
                "%s is broken -- DON'T USE IT. Expected fix : %s" % (
                fct.__name__, self.date))

        return wrapper


def deprecated(fct):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    @wraps(fct)
    def wrapper(*args, **kwargs):
        logger.warning("Call to deprecated function %s." % fct.__name__)

        return fct(*args, **kwargs)

    return wrapper


def loggedcall(fct):
    """
    This is a decorator which log function call.
    """

    @wraps(fct)
    def wrapper(*args, **kwargs):
        logger.info(
            "Function -- %s--  called with arguments -- %s -- and keywords -- %s --" %
            (fct.__name__, str(args), str(kwargs)))
        return_value = fct(*args, **kwargs)
        logger.info("Function -- %s -- returned -- %s --" %
                    (fct.__name__, return_value))

        return return_value

    return wrapper


def addmethod(instance):
    def decorator(fct):
        setattr(instance, fct.func_name, fct)

        return fct

    return decorator


def timedcall(fct):
    @wraps(fct)
    def wrapper(*args, **kwargs):
        t = time.time()
        return_value = fct(*args, **kwargs)
        logger.info("Function -- %s -- called : TIME -- %.4f --" %
                    (fct.__name__, time.time() - t))

        return return_value

    return wrapper


class memoizehd(object):
    """
    This is a decorator which is designed to cache to a file the
    result of a function. This function calculate a hash from the
    function and the parameters and store the result in a designed
    file. ONLY WORK WITH SCIPY/NUMPY ARRAY AND WITH HASHABLE PARAMETERS
    """

    def __init__(self, basepath='/tmp'):
        basepath = basepath[1:] if basepath.startswith('/') else basepath
        basepath = os.path.join("/tmp", basepath)
        try:
            os.stat(basepath)
        except:
            os.mkdir(basepath)
        self.basepath = basepath
        print(self.basepath)

    def __call__(self, fct):
        cachebase = os.path.join(self.basepath,
                                 str(hash(fct)) + "_" + str(os.getpid()))

        @wraps(fct)
        def wrapper(*args, **kwargs):
            cachefile = cachebase + ''.join([str(i) for i in map(hash, args)])
            cachefile += '_'.join(
                [str(hash(kwargs[i])) for i in kwargs]) + ".npy"
            try:
                os.stat(cachefile)
                logger.info(
                    "Parameters hash matched calling -- %s --, reading cached return from file " %
                    fct.__name__)
                return_value = numpy.load(cachefile)
            except:
                return_value = fct(*args, **kwargs)
                numpy.save(cachefile, return_value)

            return return_value

        return wrapper


def memoize(fct):
    """
    This is a decorator which cache the result of a function based on the
    given parameter.
    """
    return_dict = {}

    @wraps(fct)
    def wrapper(*args, **kwargs):
        if args not in return_dict:
            return_dict[args] = fct(*args, **kwargs)

        return return_dict[args]

    return wrapper


def update_doc(cls):
    """
    Decorator function to update the docstring of a class.
    Inherits the documentation from its parent class, and allows
    for additional customization via class attributes:
    __remove_parameters__, __new_section__, and __example__.

    :param cls: The class whose docstring needs to be updated
    :return: The class with the updated docstring
    """

    # Inherit parent class doc if available (assuming single inheritance)
    parent_doc = cls.__bases__[0].__doc__ if cls.__bases__ else ''

    # Replace specific substrings in the inherited doc
    parent_doc = parent_doc.replace('~obspy.', '~uquake.')
    parent_doc = parent_doc.replace('obspy', 'uquake - powered by obspy -')

    # Append parent doc to current class doc
    if parent_doc:
        cls.__doc__ = f"{parent_doc}"

    original_doc = cls.__doc__

    # Remove specified parameters from the doc
    if hasattr(cls, '__remove_parameters__'):
        for section in cls.__remove_parameters__:
            start_idx = original_doc.find(f':type {section}')
            end_idx = original_doc.find(':type',
                                        original_doc.find(f':param {section}') + 1)

            # If section exists, remove it
            if start_idx != -1:
                original_doc = original_doc[:start_idx] + original_doc[end_idx:]

        cls.__doc__ = original_doc

    # Insert new doc sections if defined
    if hasattr(cls, '__new_section__'):
        new_section = cls.__new_section__['doc']
        previous_parameter = cls.__new_section__['previous_parameter']

        if previous_parameter:
            start_idx = cls.__doc__.find(f':type {previous_parameter}')
            end_idx = cls.__doc__.find(':type',
                                       cls.__doc__.find(f':param {previous_parameter}'))
            cls.__doc__ = cls.__doc__[:start_idx] + f'{new_section}' + cls.__doc__[
                                                                       end_idx:]
        else:
            cls.__doc__ += f'{new_section}'

    # Replace example section if defined
    if hasattr(cls, '__example__'):
        start_idx = original_doc.find('.. rubric:: Example')
        if start_idx != -1:
            end_idx = original_doc.find('.. note::', start_idx)
            if end_idx == -1:
                end_idx = len(original_doc)

            original_doc = original_doc[:start_idx] + '.. rubric:: Example\n' + \
                           cls.__example__ + original_doc[end_idx:] + '\n'

        cls.__doc__ = original_doc

    return cls

