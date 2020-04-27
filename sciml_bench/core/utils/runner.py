import tensorflow as tf
import mlflow
from pathlib import Path

try:
    import horovod.tensorflow.keras as hvd
except ImportError:
    pass

from sciml_bench.core.dllogger.logger import LOGGER
from sciml_bench.core.system import HostSpec, DeviceSpecs
from sciml_bench.core.utils.hooks.mlflow import MLFlowDeviceLogger, MLFlowHostLogger, MLFlowLoggerProxy
from sciml_bench.core.utils.benchmark import Benchmark, MultiNodeBenchmark

class BenchmarkRunner:

    def __init__(self, benchmark):
        self._benchmark = benchmark

        # Log system information
        host_spec = HostSpec()
        mlflow.set_tag('host_name', host_spec.name)
        mlflow.set_tag('host_node_name', host_spec.node_name)
        mlflow.set_tag('host_ip', host_spec.node_name)
        mlflow.set_tag('host_num_cores', host_spec.num_cores)
        mlflow.set_tag('host_release', host_spec.release)
        mlflow.set_tag('host_system', host_spec.system)
        mlflow.set_tags(host_spec.cpu_info)

        # Log device information
        device_specs = DeviceSpecs()
        mlflow.set_tag('gpu_count', device_specs.device_count)
        mlflow.set_tags(device_specs.names)
        mlflow.set_tags(device_specs.brands)
        mlflow.set_tags(device_specs.uuids)
        mlflow.set_tags({k: v for k, v in device_specs.memory.items() if 'total' in k})
        mlflow.set_tags(device_specs.is_multigpu_board)

    def setup(self, **params):
        num_replicas = params['num_replicas']
        params['global_batch_size'] = params['batch_size'] * num_replicas
        Path(params['model_dir']).mkdir(parents=True, exist_ok=True)
        return params

    def run(self, log_interval=0.5, **params):
        host_logger = MLFlowHostLogger(interval=log_interval)
        device_logger = MLFlowDeviceLogger(interval=log_interval)

        host_logger.start()
        device_logger.start()

        strategy = tf.distribute.MirroredStrategy()
        num_replicas = strategy.num_replicas_in_sync
        params['num_replicas' ] = num_replicas
        params = self.setup(**params)

        mlflow.log_params(params)

        eith strategy.scope():
        dataset = dataset.batch(batch_size)
            self._benchmark.build(**params)

        LOGGER.log('Number of Replicas: {}'.format(params['num_replicas']))
        LOGGER.log('Global Batch Size: {}'.format(params['global_batch_size']))
        LOGGER.log('Replica Batch Size: {}'.format(params['batch_size']))

        if 'train' in params['exec_mode']:
            self._benchmark.train(**params)

        if 'predict' in params['exec_mode']:
            self._benchmark.predict(**params)

        self._benchmark.save_results(**params)

        host_logger.stop()
        device_logger.stop()

        mlflow.log_artifact(Path(params['model_dir']) / 'final_weights.h5')
        mlflow.log_artifact(Path(params['model_dir']) / 'params.yml')


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
            tf.config.experimental.set_memory_growth(gpu, True)

        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

        params['num_replicas' ] = hvd.size()
        num_replicas = params['num_replicas']
        params['global_batch_size'] = params['batch_size'] * num_replicas
        Path(params['model_dir']).mkdir(parents=True, exist_ok=True)

        return params

    def run(self, log_interval=0.5, **params):
        host_logger = MLFlowHostLogger(name=self._node_name, interval=log_interval)
        device_logger = MLFlowDeviceLogger(name=self._node_name, interval=log_interval)

        host_logger.start()
        device_logger.start()

        params = self.setup(**params)

        mlflow.log_params(params)

        self._benchmark.build(**params)

        if hvd.rank() == 0:
            LOGGER.log('MPI Enabled: ', hvd.mpi_enabled())
            LOGGER.log('Number of Replicas: {}'.format(params['num_replicas']))
            LOGGER.log('Global Batch Size: {}'.format(params['global_batch_size']))
            LOGGER.log('Replica Batch Size: {}'.format(params['batch_size']))

        if 'train' in params['exec_mode']:
            self._benchmark.train(**params)

        if 'predict' in params['exec_mode']:
            self._benchmark.predict(**params)

        self._benchmark.save_results(**params)

        host_logger.stop()
        device_logger.stop()

        if hvd.rank() == 0:
            mlflow.log_artifact(Path(params['model_dir']) / 'final_weights.h5')
            mlflow.log_artifact(Path(params['model_dir']) / 'params.yml')


def build_benchmark(model_fn, dataset, using_mpi=True):
    if not using_mpi:
        benchmark = Benchmark(model_fn, dataset)
        return BenchmarkRunner(benchmark)
    else:
        benchmark = MultiNodeBenchmark(model_fn, dataset)
        return MultiNodeBenchmarkRunner(benchmark)

