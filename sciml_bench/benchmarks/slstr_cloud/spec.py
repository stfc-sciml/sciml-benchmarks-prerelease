from sciml_bench.core.benchmark import TensorflowKerasMixin, Benchmark, register_benchmark
from sciml_bench.benchmarks.slstr_cloud.model import unet
from sciml_bench.benchmarks.slstr_cloud.data_loader import SLSTRDataLoader


@register_benchmark
class SLSTRCloud(TensorflowKerasMixin, Benchmark):
    """Default benchmark implementation for SLSTR Cloud"""
    name = 'slstr_cloud'

    train_dir = 'one-day'
    test_dir = 'one-day'
    epochs = 30
    loss = 'binary_crossentropy'
    batch_size = 32
    metrics = ['accuracy']
    optimizer_params = dict(learning_rate=0.001)

    def model(self, input_shape, **params):
        return unet(input_shape, **params)

    def data_loader(self, data_dir, **params):
        return SLSTRDataLoader(data_dir / self.train_dir, **params)

    def validation_data_loader(self, data_dir, **params):
        return SLSTRDataLoader(data_dir / self.test_dir, **params)
