from sciml_bench.core.logging import LOGGER
from sciml_bench.core.utils.runner import build_benchmark
from sciml_bench.benchmarks.dms_classifier.data_loader import DMSDataset
from sciml_bench.benchmarks.dms_classifier.model import small_cnn_classifier


def main(data_dir, **params):
    LOGGER.info('Beginning DMS Classifier Benchmark')
    dataset = DMSDataset(data_dir=data_dir / 'train', **params)
    validation_dataset = DMSDataset(data_dir=data_dir / 'test', **params)
    runner = build_benchmark(params.get('model_dir'), small_cnn_classifier, dataset, validation_dataset)
    runner.run(**params)
