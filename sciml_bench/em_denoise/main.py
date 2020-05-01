from sciml_bench.em_denoise.data_loader import EMGrapheneDataset
from sciml_bench.em_denoise.model import autoencoder
from sciml_bench.core.utils.runner import build_benchmark

def main(data_dir, **params):
    dataset = EMGrapheneDataset(data_dir=data_dir)
    runner = build_benchmark(autoencoder, dataset)
    runner.run(**params)
