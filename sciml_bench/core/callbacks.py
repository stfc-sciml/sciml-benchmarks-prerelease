import time
import tensorflow as tf
from threading import Timer
from abc import abstractmethod, ABCMeta
import horovod.tensorflow as hvd
from pathlib import Path

from sciml_bench.core.tracking import TrackingClient
from sciml_bench.core.dllogger import AverageMeter
from sciml_bench.core.system import DeviceSpecs, HostSpec

class TrackingCallback(tf.keras.callbacks.Callback):

    def __init__(self, output_dir,batch_size, warmup_steps=1, log_batch=False):
        self._db = TrackingClient(Path(output_dir) / 'logs.json')
        self._current_step = 0
        self._warmup_steps = warmup_steps
        self._batch_size = batch_size

        self._train_meter = AverageMeter()
        self._predict_meter = AverageMeter()
        self._test_meter = AverageMeter()
        self._log_batch = log_batch

    def on_train_batch_begin(self, batch, logs=None):
        self._t0 = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if self._current_step < self._warmup_steps:
            return

        t1 = time.time()
        batch_time = self._batch_size / (t1 - self._t0)

        self._train_meter.record(batch_time)

        if self._log_batch:
            self._db.log_metric('train_batch_log', logs, step=batch)

    def on_predict_batch_begin(self, batch, logs=None):
        self._t0 = time.time()

    def on_predict_batch_end(self, batch, logs=None):
        t1 = time.time()
        batch_time = self._batch_size / (t1 - self._t0)

        self._predict_meter.record(batch_time)

        if self._log_batch:
            self._db.log_metric('predict_batch_log', logs, step=batch)

    def on_test_batch_begin(self, batch, logs=None):
        self._t0 = time.time()

    def on_test_batch_end(self, batch, logs=None):
        t1 = time.time()
        batch_time = self._batch_size / (t1 - self._t0)

        self._test_meter.record(batch_time)

        if self._log_batch:
            self._db.log_metric('test_batch_log', logs, step=batch)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_begin_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self._current_step = epoch
        if epoch < self._warmup_steps:
            return

        metrics = {
            'duration', time.time() - self._epoch_begin_time,
            'samples_per_sec', self._train_meter.get_value()
        }
        metrics.update(logs)
        self._db.log_metric('epoch_log', metrics, step=epoch)

    def on_train_begin(self, logs=None):
        self._train_begin_time = time.time()

    def on_train_end(self, logs=None):
        metrics = {
            'duration', time.time() - self._train_begin_time
        }
        metrics.update(logs)
        self._db.log_metric('train_log', metrics)

    def on_test_begin(self,logs=None):
        self._test_begin_time = time.time()

    def on_test_end(self,logs=None):
        metrics = {
            'duration', time.time() - self._test_begin_time,
            'samples_per_sec', self._test_meter.get_value()
        }
        metrics.update(logs)
        self._db.log_metric('test_log', metrics)

    def on_predict_begin(self, logs=None):
        self._predict_begin_time = time.time()

    def on_predict_end(self, logs=None):
        metrics = {
            'duration', time.time() - self._predict_begin_time,
            'samples_per_sec', self._predict_meter.get_value()
        }
        metrics.update(logs)
        self._db.log_metric('predict_log', metrics)


class RepeatedTimer:
    __metaclass__ = ABCMeta

    def __init__(self, interval, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False

    @abstractmethod
    def run(self):
        pass

    def _run(self):
        self.is_running = False
        self.start()
        self.run(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            # Important! Must be registered as daemon to properly exit
            # if killed by external process
            self._timer.daemon = True
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

    def __enter__(self):
        self.start()

    def __exit__(self ,type, value, traceback):
        self.stop()

class DeviceLogger(RepeatedTimer):

    def __init__(self, output_dir, name='',  prefix='', *args, **kwargs):
        super(DeviceLogger, self).__init__(*args, **kwargs)

        file_name = 'node_{}_devices.json'.format(name)
        self._db = TrackingClient(Path(output_dir) / file_name)

        self._step = 0
        self._prefix = prefix
        self._name = name
        self._spec = DeviceSpecs()

    def run(self):
        metrics = {'execution_mode': self._prefix, 'name': self._name}

        for index, device in enumerate(self._spec.device_specs()):
            metrics['gpu_{}'.format(index)] = {
                'memory': {filter_word: device.memory[filter_word] for filter_word in ['free', 'used']},
                'utilization': device.utilization_rates,
                'power': device.power_usage,
            }

        self._db.log_metric('device_log', metrics, self._step)
        self._step += 1

class HostLogger(RepeatedTimer):

    def __init__(self, output_dir, name='', prefix='', per_device=False, *args, **kwargs):
        super(HostLogger, self).__init__(*args, **kwargs)
        self._step = 0
        self._name = '_'
        self._prefix = prefix
        self._spec = HostSpec(per_device=per_device)

        file_name = 'node_{}_host.json'.format(name)
        self._db = TrackingClient(Path(output_dir) / file_name)

    def run(self):
        metrics = {
                'execution_mode': self._prefix,
                'name': self._name,
                'cpu': {'percent', self._spec.cpu_pecent},
                'memory': self._spec.memory,
                'disk': self._spec.disk_io,
                'net': self._spec.net_io
        }

        # edge case to prevent logging if the session has died
        self._db.log_metric('host_log', metrics, self._step)
        self._step += 1

class NodeLogger:

    _host_logger = None
    _device_logger = None

    def __init__(self, output_dir, name='', prefix='', interval=0.1):
        if hvd.local_rank() == 0:
            self._host_logger = HostLogger(output_dir, name=name, prefix=prefix, interval=interval)
            self._device_logger = DeviceLogger(output_dir, name=name, prefix=prefix,interval=interval)

    def has_loggers(self):
        return self._host_logger is not None and self._device_logger is not None

    def __enter__(self):
        if self.has_loggers():
            self._host_logger.start()
            self._device_logger.start()

        return self

    def __exit__(self ,type, value, traceback):
        if self.has_loggers():
            self._host_logger.stop()
            self._device_logger.stop()
