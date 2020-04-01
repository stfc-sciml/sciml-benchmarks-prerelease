import pytest
from pathlib import Path
from sciml_bench.optics.data_loader import OpticsDataLoader

@pytest.fixture()
def data_dir():
    path = Path("data/optics")
    return path

def test_OpticsDataLoader_dimensions(data_dir):
    loader = OpticsDataLoader(data_dir)
    assert loader.dimensions == (296, 394, 1)

def test_OpticsDataLoader_train(data_dir):
    loader = OpticsDataLoader(data_dir)
    dataset = loader.train_dataset()
    batch = next(dataset.as_numpy_iterator())
    imgs, labels = batch

    assert imgs.shape == (10, 296, 394, 1)
    assert imgs.min() >= 0
    assert imgs.max() <= 1
    assert labels.shape == (10,)

def test_OpticsDataLoader_valid(data_dir):
    loader = OpticsDataLoader(data_dir)
    dataset = loader.validation_dataset()
    batch = next(dataset.as_numpy_iterator())
    imgs, labels = batch

    assert imgs.shape == (10, 296, 394, 1)
    assert imgs.min() >= 0
    assert imgs.max() <= 1
    assert labels.shape == (10,)

def test_OpticsDataLoader_test(data_dir):
    loader = OpticsDataLoader(data_dir)
    dataset = loader.test_dataset()
    batch = next(dataset.as_numpy_iterator())
    imgs, labels = batch

    assert imgs.shape == (10, 296, 394, 1)
    assert imgs.min() >= 0
    assert imgs.max() <= 1
    assert labels.shape == (10,)
