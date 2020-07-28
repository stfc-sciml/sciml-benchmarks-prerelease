from sciml_bench.core.benchmark import TensorflowKerasMixin, Benchmark, register_benchmark
from sciml_bench.benchmarks.em_denoise.model import autoencoder
from sciml_bench.benchmarks.em_denoise.data_loader import EMGrapheneDataset


@register_benchmark
class EMDenoiseBenchmark(TensorflowKerasMixin, Benchmark):
    """Default benchmark implementation for EMDenoise"""
    name = 'em_denoise'

    train_dir = 'train'
    test_dir = 'test'
    epochs = 10
    loss_function = 'mse'
    batch_size = 256
    optimizer_params = dict(learning_rate=0.01)

    def model(self, input_shape, **params):
        return autoencoder(input_shape, **params)

    def data_loader(self, data_dir, **params):
        return EMGrapheneDataset(data_dir / self.train_dir, **params)

    def validation_data_loader(self, data_dir, **params):
        return EMGrapheneDataset(data_dir / self.test_dir, **params)
