import os
import tensorflow as tf

from sciml_bench.core.dllogger.logger import LOGGER
from sciml_bench.core.utils.benchmark import Benchmark
from sciml_bench.core.utils.cmd_util import PARSER, _cmd_params

from sciml_bench.slstr_cloud.model.unet import unet_v1
from sciml_bench.slstr_cloud.constants import PATCH_SIZE
from sciml_bench.slstr_cloud.data_loader import Sentinel3Dataset


def set_environment_variables(use_amp=False, **kwargs):
    # Optimization flags
    os.environ['CUDA_CACHE_DISABLE'] = '0'

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

    os.environ['TF_ADJUST_HUE_FUSED'] = 'data'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = 'data'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = 'data'

    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'

    if use_amp:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'


def main():
    """
    Starting point of the application
    """

    flags = PARSER.parse_args()
    params = _cmd_params(flags)

    set_environment_variables(**params)

    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync

    params['num_replicas'] = num_replicas
    params['global_batch_size'] = params['batch_size'] * num_replicas

    LOGGER.log('Number of Replicas: {}'.format(params['num_replicas']))
    LOGGER.log('Global Batch Size: {}'.format(params['global_batch_size']))
    LOGGER.log('Replica Batch Size: {}'.format(params['batch_size']))

    with strategy.scope():
        model = unet_v1((PATCH_SIZE, PATCH_SIZE, 9))
        model.compile(optimizer='adam', lr=params['learning_rate'],
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

if __name__ == '__main__':
    main()
