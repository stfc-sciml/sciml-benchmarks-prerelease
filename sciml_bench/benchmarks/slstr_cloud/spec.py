import sciml_bench.mark
from sciml_bench.core.benchmark import BenchmarkSpec


@sciml_bench.mark.benchmark_spec
class SLSTRCloudSpec(BenchmarkSpec):
    name = 'slstr_cloud'
    train_dir = 'pixbox'
    test_dir = 'pixbox'

    epochs = 30
    loss_function = 'binary_crossentropy'
    batch_size = 32
    metrics = ['accuracy']
    optimizer_params = dict(learning_rate=0.001)
