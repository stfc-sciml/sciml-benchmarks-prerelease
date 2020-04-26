import pytest
import tensorflow as tf
import horovod.tensorflow as hvd
from sciml_bench.core.test.helpers import FakeDataLoader, fake_model_fn
from sciml_bench.core.utils.benchmark import MultiNodeBenchmark
from sciml_bench.core.utils.runner import MultiNodeBenchmarkRunner

@pytest.fixture()
def mocked_mlflow(mocker):
    hvd.init()
    mocker.patch('mlflow.set_tag')
    mocker.patch('mlflow.set_tags')
    mocker.patch('mlflow.log_artifacts')
    mocker.patch('mlflow.log_artifact')
    mocker.patch('mlflow.log_param')
    mocker.patch('mlflow.log_params')
    mocker.patch('mlflow.log_metric')
    mocker.patch('mlflow.log_metrics')
    return mocker

# def test_create_benchmark_runner(mocked_mlflow):
#     data_loader = FakeDataLoader((10, 10, 3), (1, ))
#     benchmark = Benchmark(fake_model_fn, data_loader)
#     runner = BenchmarkRunner(benchmark)
#     assert isinstance(runner, BenchmarkRunner)

# def test_run_benchmark_runner(tmpdir, mocked_mlflow):
#     data_loader = FakeDataLoader((10, 10, 3), (1, ))
#     benchmark = Benchmark(fake_model_fn, data_loader)

#     cfg = dict(batch_size=10, lr_warmup=3, model_dir=tmpdir,
#             exec_mode='train_and_predict', epochs=1)
#     BenchmarkRunner(benchmark).run(**cfg)

def test_run_benchmark_runner_multi(tmpdir, mocked_mlflow):
    tf.keras.backend.clear_session()
    data_loader = FakeDataLoader((10, 10, 3), (1, ))
    benchmark = MultiNodeBenchmark(fake_model_fn, data_loader)

    cfg = dict(batch_size=10, lr_warmup=3, model_dir=tmpdir,
            exec_mode='train_and_predict', epochs=1)
    MultiNodeBenchmarkRunner(benchmark).run(**cfg)

