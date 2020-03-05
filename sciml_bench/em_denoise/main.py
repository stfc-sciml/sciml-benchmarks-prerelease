import tensorflow as tf

from sciml_bench.core.dllogger.logger import LOGGER
from sciml_bench.core.utils.benchmark import Benchmark

from sciml_bench.em_denoise.data_loader import EMGrapheneDataset
from sciml_bench.em_denoise.constants import IMG_SIZE
from sciml_bench.em_denoise.model import autoencoder

def main(**params):
    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync

    params['num_replicas'] = num_replicas
    params['global_batch_size'] = params['batch_size'] * num_replicas

    LOGGER.log('Number of Replicas: {}'.format(params['num_replicas']))
    LOGGER.log('Global Batch Size: {}'.format(params['global_batch_size']))
    LOGGER.log('Replica Batch Size: {}'.format(params['batch_size']))

    with strategy.scope():
        model = autoencoder(IMG_SIZE, IMG_SIZE, 1)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']), loss='mse')

    dataset = EMGrapheneDataset(data_dir=params['data_dir'],
                               batch_size=params['global_batch_size'],
                               seed=params['seed'])

    benchmark = Benchmark(model, dataset)

    if 'train' in params['exec_mode']:
        benchmark.train(params)

    if 'predict' in params['exec_mode']:
        benchmark.predict(params)

    benchmark.save_results(params['model_dir'])
