import tensorflow as tf

from sciml_bench.core.dllogger.logger import LOGGER
from sciml_bench.core.utils.benchmark import Benchmark

from sciml_bench.slstr_cloud.model.unet import unet_v1
from sciml_bench.slstr_cloud.constants import PATCH_SIZE
from sciml_bench.slstr_cloud.data_loader import Sentinel3Dataset


def main(**params):
    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync

    params['num_replicas'] = num_replicas
    params['global_batch_size'] = params['batch_size'] * num_replicas

    if params['lr_scaling'] == 'linear':
        params['learning_rate'] *= num_replicas

    LOGGER.log('Number of Replicas: {}'.format(params['num_replicas']))
    LOGGER.log('Global Batch Size: {}'.format(params['global_batch_size']))
    LOGGER.log('Replica Batch Size: {}'.format(params['batch_size']))

    with strategy.scope():
        model = unet_v1((PATCH_SIZE, PATCH_SIZE, 9))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=[
                          'accuracy',
                          # tf.keras.metrics.TruePositives(),
                          # tf.keras.metrics.TrueNegatives(),
                          # tf.keras.metrics.FalsePositives(),
                          # tf.keras.metrics.FalseNegatives()
                      ])

    dataset = Sentinel3Dataset(data_dir=params['data_dir'],
                               batch_size=params['global_batch_size'],
                               seed=params['seed'])

    benchmark = Benchmark(model, dataset)

    if 'train' in params['exec_mode']:
        benchmark.train(params)

    if 'predict' in params['exec_mode']:
        benchmark.predict(params)

    benchmark.save_results(params['model_dir'])
