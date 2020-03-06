import subprocess
from pathlib import Path

SERVER = 'scarf.rl.ac.uk'
LOCATION = Path('/work3/projects/sciml/benchmarks')

def get_dataset(name, destination, user):
    folder = LOCATION / name
    source = '{}@{}:{}'.format(user, SERVER, folder)
    destination = Path(destination) / name
    subprocess.call(["rsync", "-vaP", source, destination])

def download_datasets(name, destination, user):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    if name == 'all' or name == 'em_denoise':
        get_dataset('em_denoise', destination, user)
    if name == 'all' or name == 'dms_classifier':
        get_dataset('dms_classifier', destination, user)
    if name == 'all' or name == 'slstr_cloud':
        get_dataset('slstr_cloud', destination, user)
