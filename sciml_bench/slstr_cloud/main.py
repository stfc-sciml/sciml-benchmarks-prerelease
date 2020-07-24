from sciml_bench.slstr_cloud.model import unet
from sciml_bench.core.logging import LOGGER
from sciml_bench.slstr_cloud.data_loader import SLSTRDataLoader
from sciml_bench.core.utils.runner import build_benchmark


def main(data_dir, **params):
    data_dir = data_dir / 'pixbox'
    LOGGER.info('Beginning SLSTR Cloud Benchmark')
    LOGGER.debug('Data directory is {}'.format(data_dir))

    dataset = SLSTRDataLoader(data_dir=data_dir, **params)
    validation_dataset = SLSTRDataLoader(data_dir=data_dir, **params)

    runner = build_benchmark(params.get('model_dir'), unet, dataset, validation_dataset)
    runner.run(**params)
