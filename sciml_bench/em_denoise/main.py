from sciml_bench.core.logging import LOGGER
from sciml_bench.em_denoise.data_loader import EMGrapheneDataset
from sciml_bench.em_denoise.model import autoencoder
from sciml_bench.core.utils.runner import build_benchmark


def main(data_dir, **params):
    LOGGER.info('Beginning EM Denoise Benchmark')
    dataset = EMGrapheneDataset(data_dir=data_dir / 'train', **params)
    validation_dataset = EMGrapheneDataset(data_dir=data_dir / 'test', **params)
    runner = build_benchmark(params.get('model_dir'), autoencoder, dataset, validation_dataset)
    runner.run(**params)
