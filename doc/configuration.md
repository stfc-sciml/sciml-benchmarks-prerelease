# Configuration

 `sciml-bench` looks for a file named `config.yml` in the current directory.
This file can be used to configure individual benchmarks with parameters other
than the default configuration.

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
em_denoise:
    epochs: 10                   # number of epochs to train for
    batch_size: 50               # batch size to use
    loss: 'binary_crossentropy'  # loss function name as defined by keras
    metrics: ['accuracy']        # metric names as defined by keras
    optimizer: 'adam'            # optimizer name as defined by keras
    optimizer_params:            # parameters to pass to the optimizer object
        learning_rate: 0.001 
```
