import pytest
import horovod.tensorflow as hvd
from sciml_bench.core.test.helpers import FakeDataLoader, fake_model_fn
from sciml_bench.core.utils.benchmark import MultiNodeBenchmark
from sciml_bench.core.utils.runner import MultiNodeBenchmarkRunner


@pytest.fixture()
def horovod():
    hvd.init()


def test_run_benchmark_runner_multi(tmpdir, horovod):
    data_loader = FakeDataLoader((10, 10, 3), (1, ))
    validation_data_loader = FakeDataLoader((10, 10, 3), (1, ))
    benchmark = MultiNodeBenchmark(fake_model_fn, data_loader, validation_data_loader)

    cfg = dict(batch_size=10, lr_warmup=3, model_dir=tmpdir,
            exec_mode='train_and_predict', epochs=1, verbosity=3)
    MultiNodeBenchmarkRunner(tmpdir, benchmark).run(**cfg)
