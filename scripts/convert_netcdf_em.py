import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path


def convert_netcdf():
    path = Path('data/em_denoise').absolute()
    paths = list(path.glob('**/*.npy'))

    for path in tqdm(paths):
        images = np.load(path)
        with h5py.File(path.with_suffix('.h5'), 'w') as handle:
            handle.create_dataset("images", data=images)


if __name__ == "__main__":
    convert_netcdf()
