import time
import tensorflow as tf
from threading import Timer
from abc import abstractmethod, ABCMeta
import horovod.tensorflow as hvd
from pathlib import Path

from sciml_bench.core.tracking import TrackingClient
from sciml_bench.core.dllogger import AverageMeter
from sciml_bench.core.system import DeviceSpecs, HostSpec, bytesto

class TimingCallback(tf.keras.callbacks.Callback):

    def __init__(self, output_dir,batch_size, warmup_steps=1):
        self._db = TrackingClient(Path(output_dir) / 'logs.json')
        self._current_step = 0
        self._warmup_steps = warmup_steps
        self._batch_size = batch_size

        self._train_meter = AverageMeter()
        self._predict_meter = AverageMeter()
        self._test_meter = AverageMeter()

    def on_train_batch_begin(self, batch, logs=None):
        self._t0 = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if self._current_step < self._warmup_steps:
            return

        t1 = time.time()
        batch_time = self._batch_size / (t1 - self._t0)

        self._train_meter.record(batch_time)

    def on_predict_batch_begin(self, batch, logs=None):
        self._t0 = time.time()

    def on_predict_batch_end(self, batch, logs=None):
        t1 = time.time()
        batch_time = self._batch_size / (t1 - self._t0)

        self._predict_meter.record(batch_time)

    def on_test_batch_begin(self, batch, logs=None):
        self._t0 = time.time()

    def on_test_batch_end(self, batch, logs=None):
        t1 = time.time()
        batch_time = self._batch_size / (t1 - self._t0)

        self._test_meter.record(batch_time)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_begin_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self._current_step = epoch
        if epoch < self._warmup_steps:
            return

        self._db.log_metric('train_epoch_duration', time.time() - self._epoch_begin_time, step=epoch)
        self._db.log_metric('train_samples-per-sec', self._train_meter.get_value(), step=epoch)

    def on_train_begin(self, logs=None):
        self._train_begin_time = time.time()

    def on_train_end(self, logs=None):
        self._db.log_metric('train_duration', time.time() - self._train_begin_time)

    def on_test_begin(self,logs=None):
        self._test_begin_time = time.time()

    def on_test_end(self,logs=None):
        self._db.log_metric('val_duration', time.time() - self._test_begin_time)
        self._db.log_metric('val_samples-per-sec', self._test_meter.get_value())

    def on_predict_begin(self, logs=None):
        self._predict_begin_time = time.time()

    def on_predict_end(self, logs=None):
        self._db.log_metric('test_duration', time.time() - self._predict_begin_time)
        self._db.log_metric('test_samples-per-sec', self._predict_meter.get_value())


class Callback(tf.keras.callbacks.Callback):

    def __init__(self, output_dir, log_batch=False):
        self._log_batch = log_batch
        self._db = TrackingClient(Path(output_dir) / 'logs.json')

    def on_train_batch_end(self, batch, logs=None):
        if self._log_batch:
            self._db.log_metrics(logs, step=batch)

    def on_test_batch_end(self, batch, logs=None):
        if self._log_batch:
            self._db.log_metrics(logs, step=batch)

    def on_predict_batch_end(self, batch, logs=None):
        if self._log_batch:
            self._db.log_metrics(logs, step=batch)

    def on_epoch_end(self, epoch, logs=None):
        if not self._log_batch:
            self._db.log_metrics(logs, step=epoch)


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
        self._name = prefix + '_'
        self._spec = DeviceSpecs()

    def run(self):
        metrics = {}

        for k,v in self._spec.memory.items():
            for filter_word in ['free', 'used']:
                if filter_word in k:
                    metrics[k] = bytesto(v, 'm')

        for k,v in self._spec.utilization_rates.items():
            metrics[k + '_utilization'] = v

        metrics.update(self._spec.power_usage)

        # Rename the metrics to have a prefix
        metrics = {self._name + k: v for k, v in metrics.items()}

        self._db.log_metrics(metrics, self._step)
        self._step += 1

class HostLogger(RepeatedTimer):

    def __init__(self, output_dir, name='', prefix='', per_device=False, *args, **kwargs):
        super(HostLogger, self).__init__(*args, **kwargs)
        self._step = 0
        self._name = prefix + '_'
        self._spec = HostSpec(per_device=per_device)

        file_name = 'node_{}_host.json'.format(name)
        self._db = TrackingClient(Path(output_dir) / file_name)

    def run(self):
        metrics = {}

        metrics.update(self._spec.memory)
        metrics.update(self._spec.cpu_percent)
        metrics.update(self._spec.disk_io)
        metrics.update(self._spec.net_io)

        # Rename the metrics to have a prefix
        metrics = {self._name + k: v for k, v in metrics.items()}

        # edge case to prevent logging if the session has died
        self._db.log_metrics(metrics, self._step)

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
