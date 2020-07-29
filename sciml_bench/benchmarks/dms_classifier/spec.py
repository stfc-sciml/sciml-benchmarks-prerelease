import tensorflow as tf
from sciml_bench.core.benchmark import TensorflowKerasMixin, Benchmark, register_benchmark
from sciml_bench.benchmarks.dms_classifier.model import dms_classifier
from sciml_bench.benchmarks.dms_classifier.data_loader import DMSDataset


@register_benchmark
class DMSBenchmark(TensorflowKerasMixin, Benchmark):
    """Default benchmark implementation for DMS Classifier"""
    name = 'dms_classifier'

    epochs = 100
    loss = 'binary_crossentropy'
    batch_size = 64
    metrics = ['accuracy', tf.keras.metrics.TruePositives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalseNegatives()]
    optimizer_params = dict(learning_rate=0.0001)

    # Model currently has class imbalance issues with weight intialisation.
    # These class weights are calculated empirically on the first 2000 samples of
    # the training set.
    fit_params = dict(class_weight={0: 1.7391304347826089, 1: 0.7017543859649122})

    def model(self, input_shape, **params):
        return dms_classifier(input_shape, **params)

    def data_loader(self, data_dir, **params):
        return DMSDataset(data_dir, is_training_data=True, **params)

    def validation_data_loader(self, data_dir, **params):
        return DMSDataset(data_dir, is_training_data=False, **params)
