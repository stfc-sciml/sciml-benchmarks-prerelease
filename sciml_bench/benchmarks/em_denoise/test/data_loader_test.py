import pytest
import horovod.tensorflow as hvd
from pathlib import Path
from sciml_bench.benchmarks.em_denoise.constants import IMG_SIZE
from sciml_bench.benchmarks.em_denoise.data_loader import EMGrapheneDataset


@pytest.fixture(scope='module')
def data_loader():
    hvd.init()
    data_dir = Path('data/em_denoise')
    data_loader = EMGrapheneDataset(data_dir)
    return data_loader


def test_create_data_properties(data_loader):
    assert isinstance(data_loader, EMGrapheneDataset)
    assert data_loader.input_shape == (IMG_SIZE, IMG_SIZE, 1)
    assert data_loader.output_shape == (IMG_SIZE, IMG_SIZE, 1)


@pytest.mark.loadtest
def test_to_dataset(data_loader):
    dataset = data_loader.to_dataset()
    inputs, outputs = next(dataset.as_numpy_iterator())

    assert inputs.shape == (10, IMG_SIZE, IMG_SIZE, 1)
    assert outputs.shape == (10, IMG_SIZE, IMG_SIZE, 1)
