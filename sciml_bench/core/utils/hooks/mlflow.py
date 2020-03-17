import tensorflow as tf
import mlflow
from threading import Timer
from abc import abstractmethod, ABCMeta
from sciml_bench.core.system import DeviceSpecs, HostSpec, bytesto

class MLFlowCallback(tf.keras.callbacks.Callback):

  def on_train_batch_end(self, batch, logs=None):
      mlflow.log_metrics(logs, step=batch)

  def on_test_batch_end(self, batch, logs=None):
      mlflow.log_metrics(logs, step=batch)

  def on_predict_batch_end(self, batch, logs=None):
      mlflow.log_metrics(logs, step=batch)

  def on_epoch_end(self, epoch, logs=None):
      mlflow.log_metrics(logs, step=epoch)


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
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

    def __enter__(self):
        self.start()

    def __exit__(self ,type, value, traceback):
        self.stop()


def log_host_stats(name, interval=0.5):
    """Decorator to log stats about the host system to mlflow for this function call"""
    def _wrap(function):
        def _inner(*args, **kwargs):
            with MLFlowHostLogger(name=name, interval=interval):
                return function(*args, **kwargs)
        return _inner
    return _wrap

def log_device_stats(name, interval=0.5):
    """Decorator to log stats about devices to mlflow for this function call"""
    def _wrap(function):
        def _inner(*args, **kwargs):
            with MLFlowDeviceLogger(name=name, interval=interval):
                return function(*args, **kwargs)
        return _inner
    return _wrap

class MLFlowDeviceLogger(RepeatedTimer):

    def __init__(self, name='', *args, **kwargs):
        super(MLFlowDeviceLogger, self).__init__(*args, **kwargs)

        self._step = 0
        self._name = name + '_'
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

        # edge case to prevent logging if the session has died
        if mlflow.active_run() is not None:
            mlflow.log_metrics(metrics, self._step)
        self._step += 1

class MLFlowHostLogger(RepeatedTimer):

    def __init__(self, name='', per_device=False, *args, **kwargs):
        super(MLFlowHostLogger, self).__init__(*args, **kwargs)
        self._step = 0
        self._name = name + '_'
        self._spec = HostSpec(per_device=per_device)

    def run(self):
        metrics = {}

        metrics.update(self._spec.memory)
        metrics.update(self._spec.cpu_percent)
        metrics.update(self._spec.disk_io)
        metrics.update(self._spec.net_io)

        # Rename the metrics to have a prefix
        metrics = {self._name + k: v for k, v in metrics.items()}

        # edge case to prevent logging if the session has died
        if mlflow.active_run() is not None:
            mlflow.log_metrics(metrics, self._step)

        self._step += 1


