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

import horovod.tensorflow as hvd
import math
import tensorflow as tf

from dllogger import tags
from dllogger.logger import LOGGER
from utils.cmd_util import PARSER, _cmd_params
from utils.data_loader import Sentinel3Dataset
from utils.hooks.profiling_hook import ProfilingHook
from utils.hooks.training_hook import TrainingHook
from utils.model_fn import unet_fn


def main(_):
    """
    Starting point of the application
    """

    flags = PARSER.parse_args()

    params = _cmd_params(flags)

    tf.logging.set_verbosity(tf.logging.ERROR)

    # Optimization flags
    os.environ['CUDA_CACHE_DISABLE'] = '0'

    os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'

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

    tf.logging.set_verbosity(tf.logging.INFO)
    hvd.init()

    # Build run config
    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.force_gpu_compatible = True
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = max(2, 40 // hvd.size() - 2)

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
                      gpu_id=hvd.rank(),
                      num_gpus=hvd.size(),
                      seed=params['seed'])

    if 'train' in params['exec_mode']:
        hooks = [hvd.BroadcastGlobalVariablesHook(0),
                 TrainingHook(params['log_every'])]

        if params['benchmark']:
            hooks.append(ProfilingHook(params['batch_size'],
                                       params['log_every'],
                                       params['warmup_steps']))

        LOGGER.log('Begin Training...')

        LOGGER.log(tags.RUN_START)
        estimator.train(
            input_fn=dataset.train_fn,
            steps=params['max_steps'],
            hooks=hooks)
        LOGGER.log(tags.RUN_STOP)

    if 'predict' in params['exec_mode']:
        if hvd.rank() == 0:
            predict_steps = dataset.test_size
            hooks = None
            if params['benchmark']:
                hooks = [ProfilingHook(params['batch_size'],
                                       params['log_every'],
                                       params['warmup_steps'])]
                predict_steps = params['warmup_steps'] * 2 * params['batch_size']

            LOGGER.log('Begin Predict...')
            LOGGER.log(tags.RUN_START)
            count = math.ceil(predict_steps/dataset.test_size)
            LOGGER.log('Predicting for {} steps'.format(count))
            predictions = estimator.predict(
                input_fn=lambda: dataset.test_fn(count),
                hooks=hooks)


            output_dir = os.path.join(params['model_dir'], 'pred')

            LOGGER.log("Predict finished")
            LOGGER.log("Results available in: {}".format(output_dir))


if __name__ == '__main__':
    tf.app.run()
