import sciml_bench.mark
from sciml_bench.core.benchmark import BenchmarkSpec


@sciml_bench.mark.benchmark_spec
class DMSClassfierSpec(BenchmarkSpec):
    name = 'dms_classifier'

    epochs = 10
    loss_function = 'categorical_crossentropy'
    batch_size = 32
    metrics = ['categorical_accuracy']
    optimizer_params = dict(learning_rate=0.01)
    n_classes = 10
