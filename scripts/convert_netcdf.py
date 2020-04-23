import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage.transform import resize
from sciml_bench.slstr_cloud.data_loader import ImageLoader

def convert_netcdf():
    path = Path('../data/slstr_cloud').absolute()
    paths = list(path.glob('**/*.SEN3'))

    for path in tqdm(paths):
        loader = ImageLoader(path)

        bts = loader.load_bts().to_array().values
        rads = loader.load_radiances().to_array().values
        refs = loader.load_radiances().to_array().values
        bayes = loader.load_flags().bayes_in.values
        summary = loader.load_flags().summary_cloud.values

        rads = np.transpose(rads, [1, 2, 0])
        refs = np.transpose(refs, [1, 2, 0])
        bts = np.transpose(bts, [1, 2, 0])
        bayes = np.expand_dims(bayes, -1)
        summary = np.expand_dims(summary, -1)

        # Resize to 1km grid
        rads = resize(rads, (bts.shape[0], bts.shape[1], 6))
        refs = resize(refs, (bts.shape[0], bts.shape[1], 6))

        with h5py.File(path.with_suffix('.hdf'), 'w') as handle:
            handle.create_dataset("bts", data=bts)
            handle.create_dataset("rads", data=rads)
            handle.create_dataset("refs", data=refs)
            handle.create_dataset("bayes", data=bayes)
            handle.create_dataset("summary", data=summary)


if __name__ == "__main__":
    convert_netcdf()
