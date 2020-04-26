import pytest
import horovod.tensorflow as hvd
from pathlib import Path
from sciml_bench.dms_classifier.constants import IMG_HEIGHT, IMG_WIDTH, N_CHANNELS
from sciml_bench.dms_classifier.data_loader import DMSDataset

@pytest.fixture(scope='module')
def data_loader():
    hvd.init()
    data_dir = Path('data/dms_classifier')
    data_loader = DMSDataset(data_dir)
    return data_loader

def test_data_properties(data_loader):
    assert isinstance(data_loader, DMSDataset)
    assert data_loader.dimensions == (IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)
    assert data_loader.train_size == 6448
    assert data_loader.test_size == 1612

@pytest.mark.loadtest
def test_train_fn(data_loader):
    dataset = data_loader.train_fn()
    inputs, outputs = next(dataset.as_numpy_iterator())

    assert inputs.shape == (10, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)
    assert outputs.shape == (10, )

@pytest.mark.loadtest
def test_test_fn(data_loader):
    dataset = data_loader.test_fn()
    inputs, outputs = next(dataset.as_numpy_iterator())

    assert inputs.shape == (10, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)
    assert outputs.shape == (10, )
