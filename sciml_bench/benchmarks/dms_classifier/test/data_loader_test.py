import pytest
import horovod.tensorflow as hvd
from pathlib import Path
from sciml_bench.benchmarks.dms_classifier.constants import IMG_HEIGHT, IMG_WIDTH, N_CHANNELS, N_CLASSES
from sciml_bench.benchmarks.dms_classifier.data_loader import DMSDataset


@pytest.fixture(scope='module')
def data_dir():
    hvd.init()
    data_dir = Path('data/dms_classifier')
    return data_dir


def test_data_properties(data_dir):
    data_loader = DMSDataset(data_dir)
    assert isinstance(data_loader, DMSDataset)
    assert data_loader.input_shape == (IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)
    assert data_loader.output_shape == (1, )


@pytest.mark.loadtest
def test_train_fn(data_dir):
    data_loader = DMSDataset(data_dir)
    dataset = data_loader.to_dataset()
    inputs, outputs = next(dataset.as_numpy_iterator())

    assert inputs.shape == (10, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)
    assert outputs.shape == (10, N_CLASSES)


@pytest.mark.loadtest
def test_test_fn(data_dir):
    data_loader = DMSDataset(data_dir)
    dataset = data_loader.to_dataset()
    inputs, outputs = next(dataset.as_numpy_iterator())

    assert inputs.shape == (10, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)
    assert outputs.shape == (10, N_CLASSES)
