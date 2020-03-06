from sciml_bench.core.utils.benchmark import Benchmark

from sciml_bench.slstr_cloud.model.unet import unet_v1
from sciml_bench.slstr_cloud.data_loader import Sentinel3Dataset
from sciml_bench.core.utils.runner import BenchmarkRunner


def main(**params):
    dataset = Sentinel3Dataset(data_dir=params['data_dir'],
                               seed=params['seed'])
    benchmark = Benchmark(unet_v1, dataset)
    BenchmarkRunner(benchmark).run(**params)
