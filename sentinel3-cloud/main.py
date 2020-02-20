# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry point of the application.

This file serves as entry point to the training of UNet for segmentation of neuronal processes.

Example:
    Training can be adjusted by modifying the arguments specified below::

        $ python main.py --exec_mode train --model_dir /datasets ...

"""

import os
import json
import tensorflow as tf

from pathlib import Path
from dllogger import tags
from dllogger.logger import LOGGER
from model.unet import unet_v1
from utils.cmd_util import PARSER, _cmd_params
from utils.data_loader import Sentinel3Dataset
from utils.hooks.profiling_hook import ProfilingHook
from utils.constants import PATCH_SIZE

def simulate_multi_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Create 2 virtual GPUs with 1GB memory each
      try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

def main():
    """
    Starting point of the application
    """

    flags = PARSER.parse_args()

    params = _cmd_params(flags)

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

    if params['use_amp']:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    # simulate_multi_gpu()
    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync
    LOGGER.log('Number of Replicas: {}'.format(num_replicas))

    with strategy.scope():
        model = unet_v1((PATCH_SIZE, PATCH_SIZE, 9))
        model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    dataset = Sentinel3Dataset(data_dir=params['data_dir'],
                               batch_size=params['batch_size'],
                               augment=params['augment'],
                               num_gpus=num_replicas,
                               seed=params['seed'])

    benchmark_results = {}

    if 'train' in params['exec_mode']:
        hooks = []
        train_profiler_hook = ProfilingHook(params['batch_size'],
                                            params['log_every'],
                                            params['warmup_steps'], num_replicas=num_replicas)
        hooks.append(train_profiler_hook)

        train_dataset = dataset.train_fn()

        LOGGER.log('Begin Training...')
        LOGGER.log('Training for {} steps'.format(params['max_steps']))

        LOGGER.log(tags.RUN_START)
        model.fit(
            train_dataset,
            epochs=params['max_steps'],
            steps_per_epoch=params['warmup_steps'],
            callbacks=hooks)
        LOGGER.log(tags.RUN_STOP)

        benchmark_results['train'] = train_profiler_hook.get_results()

    if 'predict' in params['exec_mode']:
        test_profiler_hook = ProfilingHook(params['batch_size'],
                                           params['log_every'],
                                           warmup_steps=-1, num_replicas=num_replicas)
        hooks = [test_profiler_hook]

        predict_steps = dataset.test_size

        LOGGER.log('Begin Predict...')
        LOGGER.log(tags.RUN_START)
        LOGGER.log('Predicting for {} steps'.format(predict_steps))

        test_dataset = dataset.test_fn()

        model.predict(test_dataset, callbacks=hooks)

        LOGGER.log("Predict finished")

        benchmark_results['test'] = test_profiler_hook.get_results()

        results_file = Path(params['model_dir']).joinpath('results.json')
        with results_file.open('w') as handle:
            json.dump(benchmark_results, handle)


if __name__ == '__main__':
    main()
