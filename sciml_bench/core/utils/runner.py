import tensorflow as tf
from pathlib import Path
import horovod.tensorflow.keras as hvd

from sciml_bench.core.logging import LOGGER
from sciml_bench.core.tracking import TrackingClient
from sciml_bench.core.system import HostSpec, DeviceSpecs
from sciml_bench.core.callbacks import NodeLogger
from sciml_bench.core.utils.benchmark import MultiNodeBenchmark


class MultiNodeBenchmarkRunner:

    def __init__(self, output_dir, benchmark):
        self._benchmark = benchmark
        self._output_dir = output_dir

        Path(self._output_dir).mkdir(parents=True, exist_ok=True)

        host_spec = HostSpec()
        self._node_name = host_spec.node_name

        # Log system information if on local rank 0
        if hvd.local_rank() == 0:

            # Log host information
            file_name = '{}_host.json'.format(self._node_name)
            db = TrackingClient(Path(self._output_dir) / file_name)

            host_info = {
                'name': host_spec.name,
                'node_name': host_spec.node_name,
                'ip': host_spec.node_name,
                'num_cores': host_spec.num_cores,
                'release': host_spec.release,
                'system': host_spec.system,
                'cpu_info': host_spec.cpu_info,
            }

            db.log_tag('host_info', host_info)

            # Log device information
            device_specs = DeviceSpecs()

            file_name = '{}_devices.json'.format(self._node_name)
            db = TrackingClient(Path(self._output_dir) / file_name)

            device_info = {}
            device_info['gpu_count'] = device_specs.device_count
            device_info.update({'gpu_{}'.format(i): device_specs.get_device_info(i) for i in range(device_specs.device_count)})

            db.log_tag('device_info', device_info)

    def setup(self, **params):
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.experimental.list_physical_devices('GPU')

        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                LOGGER.warning('Could not set GPU memory growth == True')

        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

        params['num_replicas'] = hvd.size()
        num_replicas = params['num_replicas']
        params['global_batch_size'] = params['batch_size'] * num_replicas

        return params

    def run(self, log_interval=0.5, **params):

        params = self.setup(**params)
        self._benchmark.build(**params)

        if hvd.rank() == 0:
            db = TrackingClient(Path(self._output_dir) / 'logs.json')
            db.log_param('params', params)

        LOGGER.info('Number of Replicas: {}'.format(params['num_replicas']))
        LOGGER.info('Global Batch Size: {}'.format(params['global_batch_size']))
        LOGGER.info('Replica Batch Size: {}'.format(params['batch_size']))

        if 'train' in params['exec_mode']:
            with NodeLogger(self._output_dir, name=self._node_name, prefix='train', interval=log_interval):
                self._benchmark.train(**params)

        if 'predict' in params['exec_mode']:
            with NodeLogger(self._output_dir, name=self._node_name, prefix='predict', interval=log_interval):
                self._benchmark.predict(**params)


def build_benchmark(model_dir, model_fn, dataset, validation_dataset):
    benchmark = MultiNodeBenchmark(model_fn, dataset, validation_dataset)
    return MultiNodeBenchmarkRunner(model_dir, benchmark)
