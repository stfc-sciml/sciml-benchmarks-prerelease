import pytest
from pathlib import Path
from sciml_bench.em_denoise.constants import IMG_SIZE
from sciml_bench.em_denoise.data_loader import EMGrapheneDataset

@pytest.fixture(scope='module')
def data_loader():
    data_dir = Path('data/em_denoise')
    data_loader = EMGrapheneDataset(data_dir)
    return data_loader

def test_create_data_properties(data_loader):
    assert isinstance(data_loader, EMGrapheneDataset)
    assert data_loader.dimensions == (IMG_SIZE, IMG_SIZE, 1)
    assert data_loader.train_size == 8000
    assert data_loader.test_size == 2000


def test_train_fn(data_loader):
    dataset = data_loader.train_fn()
    inputs, outputs = next(dataset.as_numpy_iterator())

    assert inputs.shape == (10, IMG_SIZE, IMG_SIZE, 1)
    assert outputs.shape == (10, IMG_SIZE, IMG_SIZE, 1)


def test_test_fn(data_loader):
    dataset = data_loader.test_fn()
    inputs, outputs = next(dataset.as_numpy_iterator())

    assert inputs.shape == (10, IMG_SIZE, IMG_SIZE, 1)
    assert outputs.shape == (10, IMG_SIZE, IMG_SIZE, 1)
