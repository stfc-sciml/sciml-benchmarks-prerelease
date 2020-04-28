![sciml-benchmark-logo](img/logo.png "title-1")

Suite of scientific machine learning benchmarks. This repository contains a 
selection of reference implementations for machine learning problems in 
facilities science. The code in this repository also implements command line 
tools for easily configuring and running the benchmarks.

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
### Using `rsync`

The easiest way to sync the data is to run the following command:
```
rsync -vaP <user-name>@scarf.rl.ac.uk/work3/projects/sciml/benchmarks ./data
```

Where the `<user-name>` is your SCARF username.


### Using `sciml-bench`
The `sciml-bench` command provides a method for downloading datasets from the 
remote data store. You can choose to download a single dataset or all of the 
datasets. For example, to download the EM denoise dataset we can run the following:

```
sciml-bench download all ./data/ --user <scarf-user-name>
```

Replace `<scarf-user-name>` with you actual SCARF username.

## Running Benchmarks

### Using the `sciml-bench` command 

Once installed, to run all benchmarks with default configurations, run:

```
sciml-bench
```

To run a specific benchmark use the following:

```
sciml-bench <benchmark-name>
```

For example, to run the the `em_denoise` benchmark the syntax would be:

```
sciml-bench em_denoise
```

Additionally, each benchmark takes a list of arguments such as the `batch_size`,
`learning_rate` etc. that control the run. To see a full list of options for each
benchmark run:

```
sciml-bench <benchmark-name> --help
```

The parameters for a benchmark can also be passed with a configuration YAML file 
using the `--config` option. For example:

```
sciml-bench em-denoise --config config.yml
```

Some examples of the syntax for configuration files can be found in the 
[examples](examples) folder.

### Using the Singularity container

First pull the image from singularity hub

```bash
singularity pull library://sljack/sciml/sciml-bench-tf
```

Then run all the benchmarks with the following command:

```bash
singularity run --nv sciml-bench-tf.sif
```

Run induvidual benchmarks using: 

```bash
singularity run --nv sciml-bench-tf.sif <benchmark-name>
```

For example to run the Electron Microscopy denoise benchmark:

```bash
singularity run --nv sciml-bench-tf.sif em-denoise
```

### Using the Docker container
The easiest way to run all benchmarks is to grab the docker container:

```
docker pull samueljackson/sciml-bench:latest
```

Then you can run the benchmarks with the following command:
```
sudo docker run --gpus all -v $PWD/data:/data -v $PWD/out:/sciml-bench-out samueljackson/sciml_bench
```

 - `--gpus`: specifies the number of GPUs to use from the host system
 - `-v $PWD/data:/data`: mounts the data folder on the host to the container. The part before the colon should point to location of the downloaded data
 - `-v $PWD/out:/out`:  mounts the output folder in the container to the host system. This ensures that the captured run data is saved after exection.


## Benchmarks

### EM Denoise
Increased frame collection rates on modern electron microscopes allows for the observation of dynamic processes, such as defect migration and surface reconstruction. To capture these dynamic processes, images are often collected at a high frequency, resulting in huge volumes of data. Among the range of techniques used for analysing these large datasets machine learning techniques proves to be a promising option, offering rapid identification of features and objects within the image through  semantic segmentation. Furthermore, rapid machine learning facilitated analysis and processing of images offers the promise of “self-driving'' microscopes which automatically optimise data acquisition, or even act as a monitor feeding-back in a nano-fabrication setting. In almost all instances where micrographs are analysed, it is desirable to have techniques to improve signal to noise ratios of the images. For example, being able to image at lower electron doses can facilitate experiments with greatly reduced beam induced phenomena taking place in samples, however these images are inevitably noisier than at higher doses. Effective denoising can facilitate low-dose experiments, with image quality comparable to high-dose experiments. Likewise greater time resolution can be achieved with the aid of effective image denoising procedures. 

This benchmark includes seven baseline models - Class Aware Fully Convolutional Networks, De-noising CNN, FFD-Net, U-Net, Deep Encoder Decoder with Skip Connections, Multiscale CNN, and Mixed Scale Dense Networks. The dataset for this benchmark, namely DS-EM, is of size 5GB and consists of 10,000 pairs of 256x256 electron micrographs, which are single channel images

### DMS Classification
Diffuse Multiple Scattering (DMS) is a phenomenon that has been observed in X-ray patterns for many years, but has only become accessible as a useful tool for analysis with the advent of modern X-ray sources and sensitive detectors in the past decade. The method is very promising, allowing for investigation of multi-phase materials from a single measurement – something not possible with standard X-ray experiments. However, analysis currently relies on extremely laborious searching of patterns to identify important motifs (triple intersections) that allow for inference of information. This task can only be performed by expert beam scientists and severely limits the application of this promising technique. 

This benchmark involves learning to distinguish between two possible crystal structures based on the DMS pattern. The benchmark includes a baseline CNN model. The dataset for this benchmark, DS-DMS, is of size 8.6GB and consists of 8,060 DMS diffraction patterns, with each pattern of  487x195 pixels with  three channels.

### SLSTR Cloud
Estimation of sea surface temperature (SST) from space-borne sensors, such as satellites, is crucial for a number of applications in environmental sciences. One of the aspects that underpins the derivation of SST is cloud screening, which is a step that marks each and every pixel of thousands of satellite imageries as containing cloud or clear sky, historically performed using either thresholding or Bayesian methods. This benchmark focuses on using a machine learning-based model for masking clouds, in the Sentinel-3 satellite, which carries the Sea and Land Surface Temperature Radiometer (SLSTR) instrument. More specifically, the benchmark operates on multispectral image data. 

The baseline implementation is a variation of the U-Net deep neural network. The benchmark includes two datasets of DS1-Cloud and DS2-Cloud, with sizes of 187GB and 1.6TB, respectively. Each dataset is made up of two parts: reflectance and brightness temperature. The reflectance is captured across six channels with the resolution of 2400 x 3000 pixels, and the brightness temperature is captured across three channels with the resolution of 1200 x 1500 pixels.
