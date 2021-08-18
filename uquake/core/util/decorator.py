# Copyright (c) 2009 J-Pascal Mercier
#
#
# vim: ts=4 sw=4 sts=0 noexpandtab:
import os
import time
from functools import wraps

import numpy

from ..logging import logger


def compressFile(func):
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
