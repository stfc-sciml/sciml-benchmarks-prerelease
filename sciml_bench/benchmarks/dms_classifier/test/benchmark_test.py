import pytest
from pathlib import Path
from sciml_bench.benchmarks.dms_classifier.model import small_cnn_classifier
from sciml_bench.benchmarks.dms_classifier.data_loader import DMSDataset


@pytest.fixture
def data_dir():
    return Path('data/dms_classifier/')


def test_dms_closed_benchmark(data_dir):
    benchmark = DMSBenchmarkClosed(model_func=small_cnn_classifier, data_dir=data_dir)
    assert isinstance(benchmark.data_loader, DMSDataset)
