import os
import logging
import horovod.tensorflow as hvd

# Set TF CPP logs to correct level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
log = logging.getLogger('sciml-bench')
log.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(level=logging.DEBUG)
ch.setFormatter(formatter)


class _MPIRankFilter(logging.Filter):
    """Filter to eliminate info-level logging when called from this module."""

    def __init__(self, filter_levels=None):
        super(_MPIRankFilter, self).__init__()

    def filter(self, record):
        return hvd.rank() == 0


log.addFilter(_MPIRankFilter())


def decorate_emit(fn):
        # add methods we need to the class
    def new(*args):
        levelno = args[0].levelno

        if(levelno >= logging.CRITICAL):
            color = '\x1b[31;1m'
        elif(levelno >= logging.ERROR):
            color = '\x1b[31;1m'
        elif(levelno >= logging.WARNING):
            color = '\x1b[33;1m'
        elif(levelno >= logging.INFO):
            color = '\x1b[32;1m'
        elif(levelno >= logging.DEBUG):
            color = '\x1b[35;1m'
        else:
            color = '\x1b[0m'
        # add colored *** in the beginning of the message
        args[0].msg = "{0}***\x1b[0m {1}".format(color, args[0].msg)

        # new feature i like: bolder each args of message
        args[0].args = tuple(
            '\x1b[1m' + str(arg) + '\x1b[0m' for arg in args[0].args)
        return fn(*args)
    return new


ch.emit = decorate_emit(ch.emit)
log.addHandler(ch)

LOGGER = log
