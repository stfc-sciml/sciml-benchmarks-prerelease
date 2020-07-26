import sciml_bench.mark
from sciml_bench.core.benchmark import BenchmarkSpec


@sciml_bench.mark.benchmark_spec
class DMSClassfierSpec(BenchmarkSpec):
    name = 'dms_classifier'
    train_dir = 'train'
    test_dir = 'test'

    epochs = 10
    loss_function = 'binary_crossentropy'
    batch_size = 256
    metrics = ['accuracy']
    optimizer_params = dict(learning_rate=0.01)
