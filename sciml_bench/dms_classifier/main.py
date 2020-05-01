from sciml_bench.core.logging import LOGGER
from sciml_bench.dms_classifier.data_loader import DMSDataset
from sciml_bench.dms_classifier.model import small_cnn_classifier
from sciml_bench.core.utils.runner import build_benchmark


def main(data_dir, **params):
    LOGGER.info('Beginning DMS Classifier Benchmark')
    dataset = DMSDataset(data_dir=data_dir,
                               seed=params['seed'])
    runner = build_benchmark(small_cnn_classifier, dataset)
    runner.run(**params)
