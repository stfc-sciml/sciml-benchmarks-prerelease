import tensorflow as tf
import pytest
from pathlib import Path
from data_loader import Sentinel3Dataset, ImageLoader
from constants import PATCH_SIZE

def test_sentinel3_dataset_train_fn():
    path = Path("dataset")
    dataset = Sentinel3Dataset(path, batch_size=2).train_fn()
    batch = next(dataset.as_numpy_iterator())

    # Image shape
    _, h, w, c = batch[0].shape

    assert h == PATCH_SIZE
    assert w == PATCH_SIZE
    assert c == 9

    # Mask shape
    _, h, w, c = batch[1].shape

    assert h == PATCH_SIZE
    assert w == PATCH_SIZE
    assert c == 2

    output = list(dataset)
    assert len(output) == 1
    output = output[0]
    assert output[0].shape == tf.TensorShape((2, PATCH_SIZE, PATCH_SIZE, 9))

def test_sentinel3_dataset_test_fn():
    path = Path("dataset")
    dataset = Sentinel3Dataset(path, batch_size=2).test_fn()
    batch = next(dataset.as_numpy_iterator())

    # Image shape
    _, h, w, c = batch[0].shape

    assert h == PATCH_SIZE
    assert w == PATCH_SIZE
    assert c == 9

    # Mask shape
    _, h, w, c = batch[1].shape

    assert h == PATCH_SIZE
    assert w == PATCH_SIZE
    assert c == 2

    assert batch[0].shape[0] == 2
    output = batch[0]
    assert output[0].shape == tf.TensorShape((2, PATCH_SIZE, PATCH_SIZE, 9))


@pytest.mark.benchmark(
    max_time=5,
    min_rounds=1,
    warmup=False
)
def test_sentinel3_dataset_load_single_batch(benchmark):
    path = Path("dataset")
    dataset = Sentinel3Dataset(path, batch_size=2).train_fn()
    iterator = dataset.as_numpy_iterator()

    benchmark(next, iterator)


@pytest.mark.benchmark(
    max_time=5,
    min_rounds=1,
    warmup=False
)
def test_image_loader_to_load_bts(benchmark):
    path = next(Path("dataset/train").glob('S3A*'))
    loader = ImageLoader(path)
    benchmark(loader.load_bts)

@pytest.mark.benchmark(
    max_time=5,
    min_rounds=1,
    warmup=False
)
def test_image_loader_to_load_radiances(benchmark):
    path = next(Path("dataset/train").glob('S3A*'))
    loader = ImageLoader(path)
    benchmark(loader.load_radiances)

