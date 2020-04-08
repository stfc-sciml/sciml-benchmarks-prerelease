from sciml_bench.core.utils.benchmark import Benchmark
from sciml_bench.em_denoise.data_loader import EMGrapheneDataset
from sciml_bench.em_denoise.model import autoencoder
from sciml_bench.core.utils.runner import BenchmarkRunner

def main(data_dir=None, seed=42, **params):
    dataset = EMGrapheneDataset(data_dir=params['data_dir'],
                               seed=params['seed'])

    benchmark = Benchmark(autoencoder, dataset)
    BenchmarkRunner(benchmark).run(**params)
