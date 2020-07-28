import pytest
import horovod.tensorflow as hvd

from sciml_bench.core.test.helpers import FakeBenchmark
from sciml_bench.core.runner import TensorflowKerasBenchmarkRunner


@pytest.fixture()
def horovod():
    hvd.init()


def test_run_benchmark_runner_multi(tmpdir, horovod):
    cfg = dict(batch_size=10, lr_warmup=3, model_dir=tmpdir,
            exec_mode='train_and_predict', epochs=1, verbosity=3)

    benchmark = FakeBenchmark()

    runner = TensorflowKerasBenchmarkRunner(benchmark, tmpdir)
    runner.run(**cfg)
