![sciml-benchmark-logo](img/logo.png "title-1")

Suite of scientific machine learning benchmarks. This repository contains a 
selection of reference implementations for machine learning problems in 
facilities science. The code in this repository also implements command line 
tools for easily configuring and running the benchmarks.

## Installation

 - Clone the git repository
 - Install the python package:

```bash
pip install .
```

All requirements are specified in the `requirements.txt` but some might need
tweaking depending on your platform. In general we currently require:

 - Tensorflow == 2.1.0
 - Horovod >= 0.19

## Quickstart

 - Run the `sciml-bench`:

```bash
sciml-bench --help
```

 - To run all of the benchmarks 

```bash
sciml-bench run all
```

 - To run an individual benchmark

```bash
sciml-bench run <name>
```

For example

```bash
sciml-bench run em_denoise
```

 - To list all available benchmarks

```bash
sciml-bench list
```

 - To download the benchmark data

```bash
sciml-bench download
```

```bash
sciml-bench list
```

### Using the Singularity container

We provide a singularity container with all of the software dependencies
pre-installed. You can pull the image from singularity hub like so:

```bash
singularity pull library://stfcsciml/default/sciml-bench
```

Then run all the benchmarks with the following command:

```bash
singularity run --nv sciml-bench.sif
```

The `--nv` flag is required if you wish to use host OS GPU devices. This flag
binds the NVIDIA drivers on the host machine to the container.

Run individual benchmarks using: 

```bash
singularity run --nv sciml-bench.sif <benchmark-name>
```

For example to run the Electron Microscopy denoise benchmark:

```bash
singularity run --nv sciml-bench.sif em-denoise
```


## Configuration

By default `sciml-bench` expects data to be found in a folder called `data` from
the directory it is run in. To output to use a different data directory
using the `--data-dir` flag. 

```bash
sciml-bench run --data-dir=/my/data/location
```

Likewise, `sciml-bench` will output to a directory called `sciml-bench-out` by
default. This can be changed to passing the `--model-dir` option. For example:

```bash
sciml-bench run --model-dir=/my/output/location
```

### Configuring Benchmarks

Each benchmark has a default parameter configuration specifying the number of
epochs, batch size, optimizer parameters etc. to run with. These defaults can be
overwritten using a configuration file. Currently `sciml-bench` searches for a
file called `config.yml` in the directory it is being run in. Any entries in
this file will overwrite the default configuration for the benchmarks. For
example to set a different configuration for the `em_denoise` benchmark we can
write the following:

```yaml
em_denoise:
    epochs: 30
    optimizer: 'adam'
    optimizer_params:
        learning_rate: 0.001
    batch_size: 50
```

This with override the default settings and set:
 - The number of epochs to 30
 - The optimizer to Adam
 - The learning rate for the optimizer to 0.001
 - The batch size to 50

You can set different entries for each benchmark.
