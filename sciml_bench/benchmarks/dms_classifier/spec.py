from sciml_bench.core.benchmark import TensorflowKerasMixin, Benchmark, register_benchmark
from sciml_bench.benchmarks.dms_classifier.model import dms_classifier
from sciml_bench.benchmarks.dms_classifier.data_loader import DMSDataset


@register_benchmark
class DMSBenchmark(TensorflowKerasMixin, Benchmark):
    """Default benchmark implementation for DMS Classifier"""
    name = 'dms_classifier'

    epochs = 10
    loss = 'binary_crossentropy'
    batch_size = 64
    metrics = ['accuracy']
    optimizer_params = dict(learning_rate=0.01)
    n_classes = 7

    def model(self, input_shape, **params):
        return dms_classifier(input_shape, **params)

    def data_loader(self, data_dir, **params):
        return DMSDataset(data_dir, is_training_data=True, **params)

    def validation_data_loader(self, data_dir, **params):
        return DMSDataset(data_dir, is_training_data=False, **params)
