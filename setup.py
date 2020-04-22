import os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

dependancies = []
for req in requirements:
    # if 'git+' in req:
    #     req = 'sciml_tools @ git+ssh://git@github.com/samueljackson92/sciml-tools@master#egg=some-pkg'
    dependancies.append(req)

# Install Horovod with NCCL unless the user specifies otherwise
os.environ['HOROVOD_GPU_ALLREDUCE'] = os.environ.get('HOROVOD_GPU_ALLREDUCE', 'NCCL')
os.environ['HOROVOD_GPU_BROADCAST'] = os.environ.get('HOROVOD_GPU_BROADCAST', 'NCCL')

setup(
    name='sciml-bench',
    version='0.1',
    packages=find_packages(),
    install_requires=dependancies,
    extras_require={
        'mpi': ['mpi4py', 'horovod']
        },
    entry_points={
        'console_scripts': ['sciml-bench=sciml_bench.core.command:cli'],
    },
)
