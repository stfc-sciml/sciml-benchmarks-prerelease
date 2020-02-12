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
from utils.cmd_util import PARSER, _cmd_params
from utils.data_loader import Sentinel3Dataset
from utils.hooks.profiling_hook import ProfilingHook
from utils.hooks.training_hook import TrainingHook
from utils.model_fn import unet_fn


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
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION']='1'

    tf.autograph.set_verbosity(3)
    strategy = tf.distribute.MirroredStrategy()

    # strategy = tf.distribute.MirroredStrategy()
    num_replicas = 1#strategy.num_replicas_in_sync
    LOGGER.log('Number of Replicas: {}'.format(num_replicas))
    # with strategy.scope():
    # Build run config
    gpu_options = tf.compat.v1.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.force_gpu_compatible = True
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = max(2, 40 // num_replicas - 2)

    run_config = tf.estimator.RunConfig(
        save_summary_steps=1,
        tf_random_seed=None,
        session_config=config,
        save_checkpoints_steps=params['max_steps'],
        keep_checkpoint_max=1)

    # Build the estimator model
    estimator = tf.estimator.Estimator(
        model_fn=unet_fn,
        model_dir=params['model_dir'],
        config=run_config,
        params=params)

    dataset = Sentinel3Dataset(data_dir=params['data_dir'],
                      batch_size=params['batch_size'],
                      augment=params['augment'],
                      num_gpus=num_replicas,
                      seed=params['seed'])

    benchmark_results = {}

    if 'train' in params['exec_mode']:
        hooks = [TrainingHook(params['log_every'])]

        train_profiler_hook = ProfilingHook(params['batch_size'],
                                   params['log_every'],
                                   params['warmup_steps'], num_replicas=num_replicas)
        hooks.append(train_profiler_hook)

        LOGGER.log('Begin Training...')
        LOGGER.log('Training for {} steps'.format(params['max_steps']))

        LOGGER.log(tags.RUN_START)
        estimator.train(
            input_fn=dataset.train_fn,
            steps=params['max_steps'],
            hooks=hooks)
        LOGGER.log(tags.RUN_STOP)

        benchmark_results['train'] = train_profiler_hook.get_results()

    if 'predict' in params['exec_mode']:
        test_profiler_hook = ProfilingHook(params['batch_size'],
                               params['log_every'],
                              0, num_replicas=num_replicas)
        hooks = [test_profiler_hook]

        predict_steps = params['warmup_steps'] + dataset.test_size

        LOGGER.log('Begin Predict...')
        LOGGER.log(tags.RUN_START)
        LOGGER.log('Predicting for {} steps'.format(predict_steps))

        gen = estimator.predict(input_fn=lambda:
                dataset.test_fn(predict_steps), hooks=hooks)

        # Call the generator actually make the predictions
        for result in gen:
            pass

        LOGGER.log("Predict finished")

        benchmark_results['test'] = test_profiler_hook.get_results()

    results_file = Path(params['model_dir']).joinpath('results.json')
    with results_file.open('w') as handle:
        json.dump(benchmark_results, handle)


if __name__ == '__main__':
    main()
