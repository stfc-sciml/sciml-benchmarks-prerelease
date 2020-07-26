import pytest
import tensorflow as tf
import horovod.tensorflow as hvd
import sciml_bench.mark
from sciml_bench.core.test.helpers import FakeDataLoader, fake_model_fn, FakeSpec
from sciml_bench.core.benchmark import TensorflowKerasBenchmark


@pytest.fixture()
def horovod():
    hvd.init()


@pytest.fixture()
def fake_spec():
    sciml_bench.mark.model_function('fake_spec')(fake_model_fn)
    sciml_bench.mark.data_loader('fake_spec')(FakeDataLoader)
    sciml_bench.mark.validation_data_loader('fake_spec')(FakeDataLoader)


def test_tensorflow_keras_benchmark(tmpdir, fake_spec):
    spec = FakeSpec(tmpdir, input_dims=(10, 10, 3), output_dims=(1,))
    benchmark = TensorflowKerasBenchmark(spec)

    assert benchmark.loss.__name__ == 'binary_crossentropy'
    assert isinstance(benchmark.optimizer, tf.keras.optimizers.Adam)
    assert benchmark.metrics == []

    assert isinstance(benchmark.data_loader, FakeDataLoader)
    assert isinstance(benchmark.validation_data_loader, FakeDataLoader)

    assert isinstance(benchmark.model, tf.keras.Model)
    assert benchmark.model.input_shape[1:] == spec.data_loader.input_shape
    assert benchmark.model.output_shape[1:] == spec.data_loader.output_shape
