import pytest
import horovod.tensorflow as hvd
import sciml_bench.mark
from sciml_bench.core.test.helpers import FakeDataLoader, fake_model_fn, FakeSpec
from sciml_bench.core.benchmark import TensorflowKerasBenchmark
from sciml_bench.core.runner import TensorflowKerasBenchmarkRunner


@pytest.fixture()
def horovod():
    hvd.init()


@pytest.fixture()
def fake_spec():
    sciml_bench.mark.model_function('fake_spec')(fake_model_fn)
    sciml_bench.mark.data_loader('fake_spec')(FakeDataLoader)
    sciml_bench.mark.validation_data_loader('fake_spec')(FakeDataLoader)


def test_run_benchmark_runner_multi(tmpdir, horovod, fake_spec):
    cfg = dict(batch_size=10, lr_warmup=3, model_dir=tmpdir,
            exec_mode='train_and_predict', epochs=1, verbosity=3)

    params = dict(input_dims=(10, 10, 3), output_dims=(1, ))
    fake_spec = FakeSpec(tmpdir, **params)
    benchmark = TensorflowKerasBenchmark(fake_spec)

    runner = TensorflowKerasBenchmarkRunner(benchmark, tmpdir)
    runner.run(**cfg)
