import pytest
from sciml_bench.core.test.helpers import FakeDataLoader, fake_model_fn
from sciml_bench.core.utils.benchmark import Benchmark

@pytest.fixture()
def mocked_mlflow(mocker):
    mocker.patch('mlflow.set_tag')
    mocker.patch('mlflow.set_tags')
    mocker.patch('mlflow.log_artifacts')
    mocker.patch('mlflow.log_artifact')
    mocker.patch('mlflow.log_param')
    mocker.patch('mlflow.log_params')
    mocker.patch('mlflow.log_metric')
    mocker.patch('mlflow.log_metrics')
    return mocker

def test_create_benchmark(mocked_mlflow):
    data_loader = FakeDataLoader((10, 10, 3), (1, ))
    benchmark = Benchmark(fake_model_fn, data_loader)
    assert isinstance(benchmark, Benchmark)


def test_build_benchmark(tmpdir, mocked_mlflow):
    data_loader = FakeDataLoader((10, 10, 3), (1, ))

    cfg = dict(batch_size=10, lr_scaling='none', model_dir=tmpdir)
    benchmark = Benchmark(fake_model_fn, data_loader)
    benchmark.build(**cfg)

def test_train_benchmark(tmpdir, mocked_mlflow):
    data_loader = FakeDataLoader((10, 10, 3), (1, ))

    cfg = dict(batch_size=10, lr_scaling='none', model_dir=tmpdir, global_batch_size=10, num_replicas=1, epochs=1)
    benchmark = Benchmark(fake_model_fn, data_loader)
    benchmark.build(**cfg)
    benchmark.train(**cfg)


def test_predict_benchmark(tmpdir, mocked_mlflow):
    data_loader = FakeDataLoader((10, 10, 3), (1, ))

    cfg = dict(batch_size=10, lr_scaling='none', model_dir=tmpdir, global_batch_size=10, num_replicas=1, epochs=1)
    benchmark = Benchmark(fake_model_fn, data_loader)
    benchmark.build(**cfg)
    benchmark.predict(**cfg)
