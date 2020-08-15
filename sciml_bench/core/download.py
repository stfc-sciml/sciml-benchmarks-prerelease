import math
import sqlite3
import requests
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from sciml_bench.core.bench_logger import LOGGER

STFC_S3_URI = 'https://s3.echo.stfc.ac.uk/'
DB_FILE_NAME = 'echo_index.db'
DB_URI = "{}sciml-db/{}".format(STFC_S3_URI, DB_FILE_NAME)


def download_file(uri: str, file_name: str):
    with requests.get(uri, stream=True) as response:
        response.raise_for_status()
        chunk_size = 1000000
        total_length = int(response.headers.get('content-length'))
        total_chunks = math.ceil(total_length / chunk_size)
        with open(file_name, 'wb') as handle:
            for chunk in tqdm(response.iter_content(chunk_size=chunk_size), total=total_chunks, unit='MB', ncols=100):
                handle.write(chunk)


def sync_datasets(benchmark_name, data_dir):
    LOGGER.info('Downloading Dataset Database at {}'.format(DB_URI))
    download_file(DB_URI, DB_FILE_NAME)

    conn = sqlite3.connect(DB_FILE_NAME)
    exports_db = pd.read_sql("select * from exports", con=conn)

    bucket_name = exports_db.loc[exports_db.detail ==
                                 'em_denoise'].bucket.values[0]
    dataset_uri = ''.join([STFC_S3_URI, bucket_name])

    LOGGER.info('Dataset uri {}'.format(dataset_uri))
    response = requests.get(dataset_uri)

    tree = BeautifulSoup(response.content, 'lxml')

    bucket_contents = tree.findAll('contents')
    bucket_contents = [{item.name: item.text for item in c}
                       for c in bucket_contents]
    bucket_contents = pd.DataFrame(bucket_contents)

    bucket_contents['name'] = bucket_contents.key.str.split(
        '/').map(lambda s: s[0])
    bucket_contents = bucket_contents.loc[benchmark_name ==
                                          bucket_contents.name]

    for index, row in bucket_contents.iterrows():
        file_name = Path(data_dir) / Path(row.key)

        LOGGER.info('Downloading {}'.format(file_name))
        if file_name.exists():
            LOGGER.info('{} already downloaded'.format(file_name))
            continue

        file_uri = '/'.join([dataset_uri, str(file_name)])

        file_name.parent.mkdir(parents=True, exist_ok=True)
        download_file(file_uri, file_name)
