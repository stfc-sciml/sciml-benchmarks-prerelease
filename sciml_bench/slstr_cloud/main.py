from sciml_bench.slstr_cloud.model.unet import unet_v1
from sciml_bench.slstr_cloud.data_loader import SLSTRDataLoader
from sciml_bench.core.utils.runner import build_benchmark


def main(data_dir, using_mpi=True, **params):
    dataset = SLSTRDataLoader(data_dir=data_dir,
                               seed=params['seed'])
    runner = build_benchmark(unet_v1, dataset, using_mpi=using_mpi)
    runner.run(**params)
