import tensorflow as tf
tf.enable_eager_execution()

import pytest
from pathlib import Path
from data_loader import Sentinel3Dataset, ImageLoader

def test_load_image():
    path = Path.home() / "git/ml-cloud/data/pixbox-2"
    s3d = Sentinel3Dataset(path, batch_size=2)
    dataset = s3d.train_fn()
    dataset = dataset.take(1)

    # Image shape
    _, h, w, c = dataset.output_shapes[0]

    assert h == 64
    assert w == 64
    assert c == 9

    # Mask shape
    _, h, w, c = dataset.output_shapes[1]

    assert h == 64
    assert w == 64
    assert c == 2

    output = list(dataset)
    assert len(output) == 1
    output = output[0]
    assert output[0].shape == tf.TensorShape((2, 64, 64, 9))


@pytest.mark.benchmark
def test_image_loader_to_load_bts(benchmark):
    path = list((Path().home() / "git/ml-cloud/data/pixbox-2").glob('S3A*'))[0]
    loader = ImageLoader(path)
    benchmark(loader.load_bts)

@pytest.mark.benchmark
def test_image_loader_to_load_radiances(benchmark):
    path = list((Path().home() / "git/ml-cloud/data/pixbox-2").glob('S3A*'))[0]
    loader = ImageLoader(path)
    benchmark(loader.load_radiances)

@pytest.mark.benchmark
def test_image_loader_to_load_reflectances(benchmark):
    path = list((Path().home() / "git/ml-cloud/data/pixbox-2").glob('S3A*'))[0]
    loader = ImageLoader(path)
    benchmark(loader.load_reflectance)
