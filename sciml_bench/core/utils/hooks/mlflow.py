import tensorflow as tf
import mlflow
from threading import Timer
from abc import abstractmethod, ABCMeta
from sciml_bench.core.system import DeviceSpecs, HostSpec

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

    def __exit__(self):
        self.stop()

def bytesto(bytes, to, bsize=1024):
    size = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(bytes)
    for i in range(size[to]):
        r = r / bsize
    return(r)

class MLFlowDeviceLogger(RepeatedTimer):

    def __init__(self, *args, **kwargs):
        super(MLFlowDeviceLogger, self).__init__(*args, **kwargs)

        self._step = 0
        self._spec = DeviceSpecs()

    def run(self):
        metrics = {}

        for k,v in self._spec.memory.items():
            for filter_word in ['free', 'used']:
                if filter_word in k:
                    metrics[k] = bytesto(v, 'm')

        for k,v in self._spec.utilization_rates.items():
            metrics[k + '_utilization'] = v

        for k,v in self._spec.power_usage.items():
            metrics[k] = v

        # edge case to prevent logging if the session has died
        if mlflow.active_run() is not None:
            mlflow.log_metrics(metrics, self._step)
        self._step += 1

class MLFlowHostLogger(RepeatedTimer):

    def __init__(self, *args, **kwargs):
        super(MLFlowHostLogger, self).__init__(*args, **kwargs)
        self._step = 0
        self._spec = HostSpec()


    def run(self):
        metrics = {}

        cpu_utilization = self._spec.cpu_percent
        for i in range(self._spec.num_cores):
            metrics['cpu_{}_utilization'.format(i)] = cpu_utilization[i]

        host_memory = self._spec.memory
        metrics['host_memory_free'] = bytesto(host_memory['memory_free'], 'm')
        metrics['host_memory_used'] = bytesto(host_memory['memory_used'], 'm')
        metrics['host_memory_available'] = bytesto(host_memory['memory_available'], 'm')
        metrics['host_memory_utilization'] = host_memory['memory_percent']

        # edge case to prevent logging if the session has died
        if mlflow.active_run() is not None:
            mlflow.log_metrics(metrics, self._step)

        self._step += 1
