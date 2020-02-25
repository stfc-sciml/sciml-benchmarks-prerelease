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

import time

import tensorflow as tf
from dllogger import LOGGER, AverageMeter


class ProfilingHook(tf.keras.callbacks.Callback):

    def __init__(self, batch_size, warmup_steps, num_replicas=1):
        self._num_replicas = num_replicas
        self._current_step = 0
        self._global_batch_size = batch_size * self._num_replicas
        self._meter = AverageMeter()
        self._t0 = 0

        self._warmup_steps = warmup_steps
        self._warmup_meter = AverageMeter()
        self._warmup_duration = 0
        self._warmup_finished = False

        self._full_meter = AverageMeter()

    def _update_batch_start(self):

        if self._current_step > self._warmup_steps:
            # Check if this was the first step passed the warmup
            if not self._warmup_finished:
                # Ok, this was the first step after the warmup
                # calculate the duration and mark warmup as finished
                self._warmup_duration = time.time() - self._session_begin_time
                self._warmup_finished = True

        self._t0 = time.time()

    def _update_batch_end(self):
        batch_time = time.time() - self._t0
        ips = self._global_batch_size / batch_time
        self._full_meter.record(ips)

        if self._current_step <= self._warmup_steps:
            self._warmup_meter.record(ips)

        if self._current_step > self._warmup_steps:
            self._meter.record(ips)

        self._current_step += 1

    def on_train_batch_begin(self, batch, logs=None):
        self._update_batch_start()

    def on_train_batch_end(self, batch, logs=None):
        self._update_batch_end()

    def on_predict_batch_begin(self, batch, logs=None):
        self._update_batch_start()

    def on_predict_batch_end(self, batch, logs=None):
        self._update_batch_end()

    def on_test_batch_begin(self, batch, logs=None):
        self._update_batch_start()

    def on_test_batch_end(self, batch, logs=None):
        self._update_batch_end()

    def on_train_begin(self, logs=None):
        self._session_begin_time = time.time()

    def on_train_end(self, logs=None):
        self._session_end_time = time.time()
        LOGGER.log('average_images_per_second', self._meter.get_value())

    def on_predict_begin(self, logs=None):
        self._session_begin_time = time.time()

    def on_predict_end(self, logs=None):
        self._session_end_time = time.time()
        LOGGER.log('average_images_per_second', self._meter.get_value())

    def on_test_begin(self, logs=None):
        self._session_begin_time = time.time()

    def on_test_end(self, logs=None):
        self._session_end_time = time.time()
        LOGGER.log('average_images_per_second', self._meter.get_value())

    def get_results(self):
        return {
                'warmup_avg_ips': self._warmup_meter.get_value(),
                'warmup_duration': self._warmup_duration,
                'avg_ips': self._meter.get_value(),
                'total_duration': self._session_end_time - self._session_begin_time,
                'full_avg_ips': self._full_meter.get_value()
                }
