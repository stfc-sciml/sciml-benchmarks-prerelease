import tensorflow as tf
import numpy as np
import pytest
import horovod.tensorflow as hvd
from pathlib import Path

from sciml_bench.slstr_cloud.data_loader import SLSTRDataLoader
from sciml_bench.slstr_cloud.constants import PATCH_SIZE

hvd.init()

@pytest.mark.loadtest
def test_sentinel3_dataset_train_fn():
    path = Path("data/slstr_cloud")
    dataset = SLSTRDataLoader(path).train_fn(batch_size=2)
    batch = next(dataset.as_numpy_iterator())
    img, msk = batch

    # Image shape
    _, h, w, c = img.shape

    assert h == PATCH_SIZE
    assert w == PATCH_SIZE
    assert c == 9
    assert np.count_nonzero(img[0]) > 0
    assert np.all(np.isfinite(img[..., 6:]))
    assert np.all(np.isfinite(img[..., :6]))

    # Mask shape
    _, h, w, c = msk.shape

    assert h == PATCH_SIZE
    assert w == PATCH_SIZE
    assert c == 2
    assert np.count_nonzero(msk[0]) > 0
    assert np.all(np.isfinite(msk))
    assert msk.max() == 1
    assert msk.min() == 0

@pytest.mark.loadtest
def test_sentinel3_dataset_test_fn():
    path = Path("data/slstr_cloud")
    dataset = SLSTRDataLoader(path).test_fn(batch_size=2)
    batch = next(dataset.as_numpy_iterator())
    img, msk = batch

    # Image shape
    _, h, w, c = img.shape

    assert h == PATCH_SIZE
    assert w == PATCH_SIZE
    assert c == 9

    # Mask shape
    _, h, w, c = msk.shape

    assert h == PATCH_SIZE
    assert w == PATCH_SIZE
    assert c == 2

    assert img.shape == tf.TensorShape((2, PATCH_SIZE, PATCH_SIZE, 9))
    assert msk.shape == tf.TensorShape((2, PATCH_SIZE, PATCH_SIZE, 2))

@pytest.mark.loadtest
@pytest.mark.benchmark(
    max_time=30,
    min_rounds=3,
    warmup=False
)
def test_sentinel3_dataset_load_data(benchmark):
    path = next(Path("data/slstr_cloud/train").glob('S3A*.hdf'))
    path = str(path)
    dataset = SLSTRDataLoader(Path('data/slstr_cloud'))

    benchmark(dataset._load_data, path)


@pytest.mark.loadtest
@pytest.mark.benchmark(
    max_time=30,
    min_rounds=3,
    warmup=False
)
def test_sentinel3_dataset_parse_file(benchmark):
    path = next(Path("data/slstr_cloud/train").glob('S3A*.hdf'))
    path = str.encode(str(path))
    dataset = SLSTRDataLoader(Path('data/slstr_cloud'))

    benchmark(lambda x: next(dataset._parse_file(x)), path)
