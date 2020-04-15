from sciml_bench.em_denoise.data_loader import EMGrapheneDataset
from sciml_bench.em_denoise.model import autoencoder
from sciml_bench.core.utils.runner import build_benchmark

def main(data_dir, seed=42, using_mpi=False, **params):
    dataset = EMGrapheneDataset(data_dir=data_dir,
                               seed=seed)

    runner = build_benchmark(autoencoder, dataset, using_mpi=using_mpi)
    runner.run(**params)
