import tensorflow as tf
import mlflow
from pathlib import Path
import horovod.tensorflow.keras as hvd

from sciml_bench.core.logging import LOGGER
from sciml_bench.core.system import HostSpec, DeviceSpecs
from sciml_bench.core.utils.hooks.mlflow import NodeLogger, MLFlowLoggerProxy
from sciml_bench.core.utils.benchmark import MultiNodeBenchmark

class MultiNodeBenchmarkRunner:

    def __init__(self, benchmark):
        self._benchmark = benchmark
        assert isinstance(self._benchmark, MultiNodeBenchmark), "Benchmark is not a MultiNode benchmark!"

        host_spec = HostSpec()
        self._node_name = host_spec.node_name

        mlflow_logger = MLFlowLoggerProxy(self._node_name)

        # Log system information if on rank 0
        if hvd.local_rank() == 0:
            mlflow_logger.set_tag('host_name', host_spec.name)
            mlflow_logger.set_tag('host_node_name', host_spec.node_name)
            mlflow_logger.set_tag('host_ip', host_spec.node_name)
            mlflow_logger.set_tag('host_num_cores', host_spec.num_cores)
            mlflow_logger.set_tag('host_release', host_spec.release)
            mlflow_logger.set_tag('host_system', host_spec.system)
            mlflow_logger.set_tags(host_spec.cpu_info)

            # Log device information
            device_specs = DeviceSpecs()

            mlflow_logger.set_tag('gpu_count', device_specs.device_count)
            mlflow_logger.set_tags(device_specs.names)
            mlflow_logger.set_tags(device_specs.brands)
            mlflow_logger.set_tags(device_specs.uuids)
            mlflow_logger.set_tags({k: v for k, v in device_specs.memory.items() if 'total' in k})
            mlflow_logger.set_tags(device_specs.is_multigpu_board)

    def setup(self, **params):
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.experimental.list_physical_devices('GPU')

        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                raise RuntimeWarning("Cannot set GPU memory growth == True")

        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

        params['num_replicas' ] = hvd.size()
        num_replicas = params['num_replicas']
        params['global_batch_size'] = params['batch_size'] * num_replicas
        Path(params['model_dir']).mkdir(parents=True, exist_ok=True)

        return params

    def run(self, log_interval=0.5, **params):

        params = self.setup(**params)
        self._benchmark.build(**params)

        if hvd.rank() == 0:
            mlflow.log_params(params)

        LOGGER.info('Number of Replicas: {}'.format(params['num_replicas']))
        LOGGER.info('Global Batch Size: {}'.format(params['global_batch_size']))
        LOGGER.info('Replica Batch Size: {}'.format(params['batch_size']))

        if 'train' in params['exec_mode']:
            name = '_'.join(['train', self._node_name])
            with NodeLogger(name=name, interval=log_interval):
                self._benchmark.train(**params)

        if 'predict' in params['exec_mode']:
            name = '_'.join(['predict', self._node_name])
            with NodeLogger(name=name, interval=log_interval):
                self._benchmark.predict(**params)

def build_benchmark(model_fn, dataset, using_mpi=True):
    benchmark = MultiNodeBenchmark(model_fn, dataset)
    return MultiNodeBenchmarkRunner(benchmark)
