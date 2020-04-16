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

""" Model construction utils

This module provides a convenient way to create different topologies
based around UNet.

"""
# import nvtx.plugins.tf as nvtx_tf
import tensorflow as tf
from sciml_bench.slstr_cloud.model.layers import output_block, upsample_block, bottleneck, downsample_block, input_block


def unet_v1(input_shape, learning_rate=0.001, **params):
    """ U-Net: Convolutional Networks for Biomedical Image Segmentation

    Source:
        https://arxiv.org/pdf/1505.04597

    """

    inputs = tf.keras.layers.Input(input_shape)
    skip_connections = []

    out, skip = input_block(inputs, filters=64)

    skip_connections.append(skip)

    # out, context = nvtx_tf.ops.start(out, message='Encoder',
    #         domain_name='Forward', grad_domain_name='Gradient')

    for idx, filters in enumerate([128, 256, 512]):
        out, skip = downsample_block(out, filters=filters, idx=idx)
        skip_connections.append(skip)

    # out, nvtx_tf.ops.end(out, context)
    # out, context = nvtx_tf.ops.start(out, message='Bottleneck',
    #         domain_name='Forward', grad_domain_name='Gradient')
    out = bottleneck(out, filters=1024)
    # out, nvtx_tf.ops.end(out, context)

    # out, context = nvtx_tf.ops.start(out, message='Decoder',
    #         domain_name='Forward', grad_domain_name='Gradient')
    for idx, filters in enumerate([512, 256, 128]):
        out = upsample_block(out,
                             residual_input=skip_connections.pop(),
                             filters=filters,
                             idx=idx)
    # out, nvtx_tf.ops.end(out, context)

    out = output_block(out, residual_input=skip_connections.pop(), filters=64, n_classes=2)
    model = tf.keras.Model(inputs, out)
    return model
