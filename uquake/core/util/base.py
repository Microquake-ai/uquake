# from obspy.core.util.base import *
from subprocess import call

from obspy.core.util.base import ENTRY_POINTS, _get_entry_points

# appending elements to the obspy ENTRY_POINTS
ENTRY_POINTS['grid'] = _get_entry_points('uquake.io.grid', 'readFormat')
ENTRY_POINTS['grid_write'] = _get_entry_points('uquake.io.grid',
                                               'writeFormat')

gfr_entry_points = _get_entry_points('uquake.io.waveform', 'readFormat')
gfw_entry_points = _get_entry_points('uquake.io.waveform', 'writeformat')

wf_entry_points = _get_entry_points('uquake.io.waveform', 'readFormat')

for key in wf_entry_points.keys():
    ENTRY_POINTS['waveform'][key] = wf_entry_points[key]

wfw_entry_points = _get_entry_points('uquake.io.waveform', 'writeFormat')

for key in wfw_entry_points.keys():
    ENTRY_POINTS['waveform_write'][key] = wfw_entry_points[key]

evt_entry_points = _get_entry_points('uquake.io.event', 'readFormat')

for key in evt_entry_points.keys():
    ENTRY_POINTS['event'][key] = evt_entry_points[key]


def proc(cmd, cwd='.', silent=True):
    from ..logging import logger

    try:
        if silent:
            cmd = '%s > /dev/null 2>&1' % cmd
        retcode = call(cmd, shell=True, cwd=cwd)

        if retcode < 0:
            logger.error('Child was terminated by signal %d' % (retcode,))
        # else:
        # print >>sys.stderr, "Child returned", retcode
    except OSError as e:
        logger.error('Execution failed: %s' % (e,))


def align_decimal(number, left_pad=7, precision=2):
    """Format a number in a way that will align decimal points."""
    outer = '{0:>%i}.{1:<%i}' % (left_pad, precision)
    inner = '{:.%if}' % (precision,)

    return outer.format(*(inner.format(number).split('.')))


def pretty_print_array(arr):
    return '(%s)' % ''.join([align_decimal(a) for a in arr])


def np_array(arr):
    new_arr = np.empty(shape=(len(arr),), dtype=object)

    for i, el in enumerate(arr):
        new_arr[i] = el

    return new_arr


def _read_from_plugin(plugin_type, filename, format=None, **kwargs):
    """
    Reads a single file from a plug-in's readFormat function.
    """
    eps = ENTRY_POINTS[plugin_type]
    # get format entry point
    format_ep = None

    if not format:
        # auto detect format - go through all known formats in given sort order

        for format_ep in eps.values():
            # search isFormat for given entry point
            is_format = load_entry_point(
                format_ep.dist.key,
                'obspy.plugin.%s.%s' % (plugin_type, format_ep.name),
                'isFormat')
            # If it is a file-like object, store the position and restore it
            # later to avoid that the isFormat() functions move the file
            # pointer.

            if hasattr(filename, "tell") and hasattr(filename, "seek"):
                position = filename.tell()
            else:
                position = None
            # check format
            is_format = is_format(filename)

            if position is not None:
                filename.seek(0, 0)

            if is_format:
                break
        else:
            raise TypeError('Unknown format for file %s' % filename)
    else:
        # format given via argument
        format = format.upper()
        try:
            format_ep = eps[format]
        except (KeyError, IndexError):
            msg = "Format \"%s\" is not supported. Supported types: %s"
            raise TypeError(msg % (format, ', '.join(eps)))
    # file format should be known by now
    try:
        # search readFormat for given entry point
        read_format = load_entry_point(
            format_ep.dist.key,
            'obspy.plugin.%s.%s' % (plugin_type, format_ep.name),
            'readFormat')
    except ImportError:
        msg = "Format \"%s\" is not supported. Supported types: %s"
        raise TypeError(msg % (format_ep.name, ', '.join(eps)))
    # read
    list_obj = read_format(filename, **kwargs)

    return list_obj, format_ep.name
