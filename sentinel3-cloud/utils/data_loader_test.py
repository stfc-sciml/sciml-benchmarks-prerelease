import tensorflow as tf
import numpy as np
import pytest
from pathlib import Path
from utils.data_loader import Sentinel3Dataset, ImageLoader
from utils.constants import PATCH_SIZE

def test_sentinel3_dataset_train_fn():
    path = Path("dataset")
    dataset = Sentinel3Dataset(path, batch_size=2).train_fn()
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

def test_sentinel3_dataset_test_fn():
    path = Path("dataset")
    dataset = Sentinel3Dataset(path, batch_size=2).test_fn()
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

@pytest.mark.benchmark(
    max_time=30,
    min_rounds=3,
    warmup=False
)
def test_sentinel3_dataset_load_data(benchmark):
    path = next(Path("dataset/train").glob('S3A*'))
    path = str(path)
    dataset = Sentinel3Dataset(Path('dataset'), batch_size=16)

    benchmark(dataset._load_data, path)


@pytest.mark.benchmark(
    max_time=30,
    min_rounds=3,
    warmup=False
)
def test_sentinel3_dataset_parse_file(benchmark):
    path = next(Path("dataset/train").glob('S3A*'))
    path = str.encode(str(path))
    dataset = Sentinel3Dataset(Path('dataset'), batch_size=16)

    benchmark(lambda x: next(dataset._parse_file(x)), path)
