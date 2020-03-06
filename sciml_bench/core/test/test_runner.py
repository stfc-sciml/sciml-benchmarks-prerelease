from sciml_bench.core.test.helpers import FakeDataLoader, fake_model_fn
from sciml_bench.core.utils.benchmark import Benchmark
from sciml_bench.core.utils.runner import BenchmarkRunner

def test_create_benchmark_runner():
    data_loader = FakeDataLoader((10, 10, 3), (1, ))
    benchmark = Benchmark(fake_model_fn, data_loader)
    runner = BenchmarkRunner(benchmark)
    assert isinstance(runner, BenchmarkRunner)

def test_run_benchmark_runner(tmpdir):
    data_loader = FakeDataLoader((10, 10, 3), (1, ))
    benchmark = Benchmark(fake_model_fn, data_loader)

    cfg = dict(batch_size=10, lr_scaling='none', model_dir=tmpdir,
            exec_mode='train_and_predict', epochs=1)
    BenchmarkRunner(benchmark).run(**cfg)
