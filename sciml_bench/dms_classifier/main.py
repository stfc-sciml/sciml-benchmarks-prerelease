import tensorflow as tf

from sciml_bench.core.dllogger.logger import LOGGER
from sciml_bench.core.utils.benchmark import Benchmark

from sciml_bench.dms_classifier.data_loader import DMSDataset
from sciml_bench.dms_classifier.constants import IMG_HEIGHT, IMG_WIDTH
from sciml_bench.dms_classifier.model import small_cnn_classifier


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
        model = small_cnn_classifier(IMG_WIDTH, IMG_HEIGHT, 1, n_classes=1)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                    loss='binary_crossentropy', metrics=['accuracy'])

    dataset = DMSDataset(data_dir=params['data_dir'],
                               batch_size=params['global_batch_size'],
                               seed=params['seed'])

    benchmark = Benchmark(model, dataset)

    if 'train' in params['exec_mode']:
        benchmark.train(params)

    if 'predict' in params['exec_mode']:
        benchmark.predict(params)

    benchmark.save_results(params['model_dir'])
