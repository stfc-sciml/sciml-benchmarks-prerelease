![sciml-benchmark-logo](img/logo.png "title-1")

Suite of scientific machine learning benchmarks. This repository contains a 
selection of reference implementations for machine learning problems in 
facilities science. The code in this repository also implements command line 
tools for easily configuring and running the benchmarks.

## Benchmarks


### EM Denoise

### DMS

### SLSTR Cloud

## Installation

 - Clone the git repository
 - Install the python package:

```
pip install -e .
```

 - Run the `sciml-bench`:

```
sciml-bench --help
```

## Accessing Data

The `sciml-bench` command provides a method for downloading datasets from the 
remote data store. You can choose to download a single dataset or all of the 
datasets. For example, to download the EM denoise dataset we can run the following:

```
sciml-bench download em_denoise <scarf-user-name> ./data/ 
```

Replace `<scarf-user-name>` with you actual scarf username.

## Running Benchmarks
