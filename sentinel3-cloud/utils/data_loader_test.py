import tensorflow as tf
tf.enable_eager_execution()

import pytest
from pathlib import Path
from data_loader import Sentinel3Dataset, ImageLoader

def test_sentinel3_dataset_train_fn():
    path = Path("sentinel3-cloud/dataset")
    dataset = Sentinel3Dataset(path, batch_size=2).train_fn()
    dataset = dataset.take(1)

    # Image shape
    _, h, w, c = dataset.output_shapes[0]

    assert h == 256
    assert w == 256
    assert c == 9

    # Mask shape
    _, h, w, c = dataset.output_shapes[1]

    assert h == 256
    assert w == 256
    assert c == 2

    output = list(dataset)
    assert len(output) == 1
    output = output[0]
    assert output[0].shape == tf.TensorShape((2, 256, 256, 9))

def test_sentinel3_dataset_test_fn():
    path = Path("sentinel3-cloud/dataset")
    dataset = Sentinel3Dataset(path, batch_size=20).test_fn(20)

    # Image shape
    _, h, w, c = dataset.output_shapes[0]

    assert h == 256
    assert w == 256
    assert c == 9

    # Mask shape
    _, h, w, c = dataset.output_shapes[1]

    assert h == 256
    assert w == 256
    assert c == 2

    output = list(dataset)
    assert len(output) == 20
    output = output[0]
    assert output[0].shape == tf.TensorShape((20, 256, 256, 9))


@pytest.mark.benchmark
def test_sentinel3_dataset_load_single_batch(benchmark):
    path = Path("sentinel3-cloud/dataset")
    dataset = Sentinel3Dataset(path, batch_size=20).train_fn()
    dataset = dataset.take(1)

    def test_take():
        for d in dataset:
            return

    benchmark(test_take)


@pytest.mark.benchmark
def test_image_loader_to_load_bts(benchmark):
    path = next(Path("sentinel3-cloud/dataset/train").glob('S3A*'))
    loader = ImageLoader(path)
    benchmark(loader.load_bts)

@pytest.mark.benchmark
def test_image_loader_to_load_radiances(benchmark):
    path = next(Path("sentinel3-cloud/dataset/train").glob('S3A*'))
    loader = ImageLoader(path)
    benchmark(loader.load_radiances)

