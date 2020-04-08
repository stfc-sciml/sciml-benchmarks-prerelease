from sciml_bench.core.utils.benchmark import Benchmark

from sciml_bench.dms_classifier.data_loader import DMSDataset
from sciml_bench.dms_classifier.model import small_cnn_classifier
from sciml_bench.core.utils.runner import BenchmarkRunner


def main(data_dir, **params):
    dataset = DMSDataset(data_dir=data_dir,
                               seed=params['seed'])
    benchmark = Benchmark(small_cnn_classifier, dataset)
    BenchmarkRunner(benchmark).run(**params)
