import tensorflow as tf
from pathlib import Path
from sciml_bench.core.dllogger.logger import LOGGER


class BenchmarkRunner:

    def __init__(self, benchmark):
        self._benchmark = benchmark

    def setup(self, **params):
        num_replicas = params['num_replicas']
        params['global_batch_size'] = params['batch_size'] * num_replicas

        if params['lr_scaling'] == 'linear':
            params['learning_rate'] *= num_replicas

        Path(params['model_dir']).mkdir(parents=True, exist_ok=True)
        return params

    def run(self, **params):
        strategy = tf.distribute.MirroredStrategy()
        num_replicas = strategy.num_replicas_in_sync
        params['num_replicas' ] = num_replicas
        params = self.setup(**params)

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
