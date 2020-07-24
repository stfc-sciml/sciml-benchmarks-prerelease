from sciml_bench.core.benchmark import BenchmarkSpec
from sciml_bench.benchmarks.dms_classifier.data_loader import DMSDataset
from sciml_bench.benchmarks.dms_classifier.model import small_cnn_classifier

from sciml_bench.benchmarks.em_denoise.data_loader import EMGrapheneDataset
from sciml_bench.benchmarks.em_denoise.model import autoencoder

from sciml_bench.benchmarks.slstr_cloud.data_loader import SLSTRDataLoader
from sciml_bench.benchmarks.slstr_cloud.model import unet


class DefaultDMSClassfierSpec(BenchmarkSpec):
    name = 'dms_classifier'

    def __init__(self, data_dir):
        data_loader = DMSDataset(data_dir / 'train')
        validation_data_loader = DMSDataset(data_dir / 'test')
        super().__init__(model_func=small_cnn_classifier, data_loader=data_loader, validation_data_loader=validation_data_loader)


class DefaultEMDenoiseSpec(BenchmarkSpec):
    name = 'em_denoise'

    def __init__(self, data_dir):
        data_loader = EMGrapheneDataset(data_dir / 'train')
        validation_data_loader = EMGrapheneDataset(data_dir / 'test')
        super().__init__(model_func=autoencoder, data_loader=data_loader, validation_data_loader=validation_data_loader)


class DefaultSLSTRCloudSpec(BenchmarkSpec):
    name = 'slstr_cloud'

    def __init__(self, data_dir):
        data_loader = SLSTRDataLoader(data_dir / 'train')
        validation_data_loader = SLSTRDataLoader(data_dir / 'test')
        super().__init__(model_func=unet, data_loader=data_loader, validation_data_loader=validation_data_loader)


BENCHMARKS = [
    DefaultDMSClassfierSpec,
    DefaultEMDenoiseSpec,
    DefaultSLSTRCloudSpec
]
