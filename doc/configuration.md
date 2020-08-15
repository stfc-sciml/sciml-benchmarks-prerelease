# Configuration

Benchmarks `sciml-bench` can be configured with a YAML file. This file can be used to configure individual benchmarks with parameters other
than the default configuration. The `sciml-bench` command line tool will look in
the following places for configuration information with increasing order of precedence:

 * The configuration packaged with `sciml-bench`
 * A user configuration file at `~/.sciml-bench-config.yml`
 * A file called `sciml-bench-config.yml` in the current directory
 * Command line options

For example, command line options with always overwrite options in the YAML
file.

## General Configuration

Several general properties of `sciml-bench` can be configured. The following
listing shows the current options and their defaults:

```YAML
data_dir: ~/sciml-bench-datasets  # Where to download and look for datasets
model_dir: sciml-bench-out        # Where to output benchmark results
search_path: []                   # Additional paths to search for user defined benchmarks
```


## Benchmark Specific Configurations

Each top level group should correspond to the name of a benchmark you with to
configure. For example, to change the batch size of the `em_denoise` benchmark
you can provide the following `config.yml`:

```yaml
em_denoise:
    batch_size: 50
```

If we also wished to change the number of epochs for the `slstr_cloud` benchmark
we could write:


```yaml
em_denoise:
    batch_size: 50

slstr_cloud:
    epochs: 100
```

A full template configuration file, with all possible options listed is shown
below:


```yaml
em_denoise:                      # benchmark name
    epochs: 10                   # number of epochs to train for
    batch_size: 50               # batch size to use
    loss: 'binary_crossentropy'  # loss function name as defined by keras
    metrics: ['accuracy']        # metric names as defined by keras
    optimizer: 'adam'            # optimizer name as defined by keras
    optimizer_params:            # parameters to pass to the optimizer object
        learning_rate: 0.001 
```
