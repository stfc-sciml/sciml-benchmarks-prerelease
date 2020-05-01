from sciml_bench.slstr_cloud.model.unet import unet_v1
from sciml_bench.slstr_cloud.data_loader import SLSTRDataLoader
from sciml_bench.core.utils.runner import build_benchmark


def main(data_dir,  **params):
    dataset = SLSTRDataLoader(data_dir=data_dir)
    runner = build_benchmark(unet_v1, dataset)
    runner.run(**params)
