import tensorflow as tf
import numpy as np
import pytest
import horovod.tensorflow as hvd
from pathlib import Path

from sciml_bench.benchmarks.slstr_cloud.data_loader import SLSTRDataLoader
from sciml_bench.benchmarks.slstr_cloud.constants import PATCH_SIZE, N_CHANNELS

hvd.init()


@pytest.fixture()
def data_dir():
    return Path("data/slstr_cloud/pixbox")


@pytest.mark.loadtest
def test_sentinel3_dataset_train_fn(data_dir):
    dataset = SLSTRDataLoader(data_dir, batch_size=2).to_dataset()
    batch = next(dataset.as_numpy_iterator())
    img, msk = batch

    # Image shape
    _, h, w, c = img.shape

    assert h == PATCH_SIZE
    assert w == PATCH_SIZE
    assert c == N_CHANNELS
    assert np.count_nonzero(img[0]) > 0
    assert np.all(np.isfinite(img[..., 6:]))
    assert np.all(np.isfinite(img[..., :6]))

    # Mask shape
    _, h, w, c = msk.shape

    assert h == PATCH_SIZE
    assert w == PATCH_SIZE
    assert c == 1
    assert np.count_nonzero(msk[0]) > 0
    assert np.all(np.isfinite(msk))
    assert msk.max() == 1
    assert msk.min() == 0


def test_create_data_properties(data_dir):
    data_loader = SLSTRDataLoader(data_dir, batch_size=2)
    assert data_loader.input_shape == (PATCH_SIZE, PATCH_SIZE, N_CHANNELS)
    assert data_loader.output_shape == (PATCH_SIZE, PATCH_SIZE, 1)


@pytest.mark.loadtest
def test_sentinel3_dataset_test_fn(data_dir):
    dataset = SLSTRDataLoader(data_dir, batch_size=2).to_dataset()
    batch = next(dataset.as_numpy_iterator())
    img, msk = batch

    # Image shape
    _, h, w, c = img.shape

    assert h == PATCH_SIZE
    assert w == PATCH_SIZE
    assert c == N_CHANNELS

    # Mask shape
    _, h, w, c = msk.shape

    assert h == PATCH_SIZE
    assert w == PATCH_SIZE
    assert c == 1

    assert img.shape == tf.TensorShape((2, PATCH_SIZE, PATCH_SIZE, N_CHANNELS))
    assert msk.shape == tf.TensorShape((2, PATCH_SIZE, PATCH_SIZE, 1))
