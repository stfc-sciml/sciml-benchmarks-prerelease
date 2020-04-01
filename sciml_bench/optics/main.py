from sciml_bench.core.utils.benchmark import Benchmark
from sciml_bench.optics.data_loader import OpticsDataLoader
from sciml_bench.optics.model import cnn_model
from sciml_bench.core.utils.runner import BenchmarkRunner

def main(**params):
    dataset = OpticsDataLoader(data_dir=params['data_dir'],
                               seed=params['seed'])

    benchmark = Benchmark(cnn_model, dataset)
    BenchmarkRunner(benchmark).run(**params)
