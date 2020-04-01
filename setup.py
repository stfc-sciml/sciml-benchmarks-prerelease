from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

dependancies = []
for req in requirements:
    if 'git+' in req:
        req = 'sciml_tools @ git+ssh://git@github.com/samueljackson92/sciml-tools@master#egg=some-pkg'
    dependancies.append(req)

setup(
    name='sciml-bench',
    version='0.1',
    packages=find_packages(),
    install_requires=dependancies,
    entry_points={
        'console_scripts': ['sciml-bench=sciml_bench.core.command:cli'],
    },
)
