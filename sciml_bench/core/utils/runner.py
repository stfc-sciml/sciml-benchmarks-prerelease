import tensorflow as tf
import mlflow
from pathlib import Path
from sciml_bench.core.dllogger.logger import LOGGER
from sciml_bench.core.system import HostSpec, DeviceSpecs
from sciml_bench.core.utils.hooks.mlflow import MLFlowDeviceLogger, MLFlowHostLogger, RepeatedTimer


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
        mlflow.set_tag('host_memory_total', host_spec.memory['memory_total'])
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

        if params['lr_scaling'] == 'linear':
            params['learning_rate'] *= num_replicas

        Path(params['model_dir']).mkdir(parents=True, exist_ok=True)
        return params

    def run(self, **params):
        host_logger = RepeatedTimer(0.5, MLFlowHostLogger())
        device_logger = RepeatedTimer(0.5, MLFlowDeviceLogger())

        host_logger.start()
        device_logger.start()

        strategy = tf.distribute.MirroredStrategy()
        num_replicas = strategy.num_replicas_in_sync
        params['num_replicas' ] = num_replicas
        params = self.setup(**params)

        mlflow.log_params(params)

        with strategy.scope():
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
