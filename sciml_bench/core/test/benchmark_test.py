import pytest
import horovod.tensorflow as hvd
from sciml_bench.core.test.helpers import FakeDataLoader, fake_model_fn
from sciml_bench.core.utils.benchmark import MultiNodeBenchmark

@pytest.fixture()
def horovod():
    hvd.init()

def test_create_multi_node_benchmark(horovod):
    data_loader = FakeDataLoader((10, 10, 3), (1, ))
    benchmark = MultiNodeBenchmark(fake_model_fn, data_loader)
    assert isinstance(benchmark, MultiNodeBenchmark)

def test_build_multi_node_benchmark(tmpdir, horovod):
    data_loader = FakeDataLoader((10, 10, 3), (1, ))

    cfg = dict(batch_size=10, lr_warmup=3, model_dir=tmpdir)
    benchmark = MultiNodeBenchmark(fake_model_fn, data_loader)
    benchmark.build(**cfg)

def test_train_multi_node_benchmark(tmpdir, horovod):
    data_loader = FakeDataLoader((10, 10, 3), (1, ))

    cfg = dict(batch_size=10, lr_warmup=3, model_dir=tmpdir, global_batch_size=10, num_replicas=1, epochs=1)
    benchmark = MultiNodeBenchmark(fake_model_fn, data_loader)
    benchmark.build(**cfg)
    benchmark.train(**cfg)

    assert (tmpdir / 'final_weights.h5').exists()

def test_predict_multi_node_benchmark(tmpdir, horovod):
    data_loader = FakeDataLoader((10, 10, 3), (1, ))

    cfg = dict(batch_size=10, lr_warmup=3, model_dir=tmpdir, global_batch_size=10, num_replicas=1, epochs=1)
    benchmark = MultiNodeBenchmark(fake_model_fn, data_loader)
    benchmark.build(**cfg)
    benchmark.predict(**cfg)
