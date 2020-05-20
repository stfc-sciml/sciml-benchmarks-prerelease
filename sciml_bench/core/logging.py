import os
import logging
import horovod.tensorflow as hvd

# Set TF CPP logs to correct level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
log = logging.getLogger('sciml-bench')
log.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(level=logging.INFO)
ch.setFormatter(formatter)
log.addHandler(ch)

class _MPIRankFilter(logging.Filter):
  """Filter to eliminate info-level logging when called from this module."""

  def __init__(self, filter_levels=None):
    super(_MPIRankFilter, self).__init__()

  def filter(self, record):
      return hvd.rank() == 0

log.addFilter(_MPIRankFilter())

LOGGER = log
