import os
from sciml_bench import __version__
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

dependancies = requirements

# Install Horovod with NCCL unless the user specifies otherwise
os.environ['HOROVOD_GPU_ALLREDUCE'] = os.environ.get('HOROVOD_GPU_ALLREDUCE', 'NCCL')
os.environ['HOROVOD_GPU_BROADCAST'] = os.environ.get('HOROVOD_GPU_BROADCAST', 'NCCL')

setup(
    name='sciml-bench',
    version=__version__,
    packages=find_packages(),
    package_data={'sciml_bench': ['config/*']},
    install_requires=dependancies,
    entry_points={
        'console_scripts': ['sciml-bench=sciml_bench.core.command:cli'],
    },
)
