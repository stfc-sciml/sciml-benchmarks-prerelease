import pytest
import tensorflow as tf
import horovod.tensorflow as hvd
from sciml_bench.core.test.helpers import FakeDataLoader, FakeBenchmark, FakeBenchmarkDerived


@pytest.fixture()
def horovod():
    hvd.init()


def test_tensorflow_keras_benchmark(tmpdir):
    benchmark = FakeBenchmark()

    assert benchmark.loss_.__name__ == 'binary_crossentropy'
    assert isinstance(benchmark.optimizer_, tf.keras.optimizers.Adam)
    assert benchmark.metrics == []

    assert isinstance(benchmark.data_loader_, FakeDataLoader)
    assert isinstance(benchmark.validation_data_loader_, FakeDataLoader)

    model = benchmark.model((200, 200, 1))
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape[1:] == (200, 200, 1)
    assert model.output_shape[1:] == (1, )

    benchmark_dervied = FakeBenchmarkDerived()
    assert benchmark_dervied.epochs == 10
    assert benchmark_dervied.epochs != benchmark.epochs

    assert benchmark_dervied.loss_ == 'a loss'
    assert benchmark_dervied.optimizer_ == 'an optimizer'


def test_create_benchmark_with_config():
    benchmark = FakeBenchmark()
    assert benchmark.epochs == 0

    benchmark = FakeBenchmark(epochs=100)
    assert benchmark.epochs == 100
