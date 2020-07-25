from collections import defaultdict
import tensorflow as tf

# Registry of model functions for each benchmark spec
# This provides a mapping from str -> list, where the key is the name of the
# benchmark spec and the value is a list of registered functions for that spec.
# Currently we take the last function added as the one to use.
MODEL_FUNCS_REGISTRY = defaultdict(list)

# Registry of data loaders for each benchmark spec
# This provides a mapping from str -> list, where the key is the name of the
# benchmark spec and the value is a list of registered data loader classes for that spec.
# Currently we take the last registered loader added as the one to use.
DATA_LOADER_REGISTRY = defaultdict(list)

# Registry of validation data loaders for each benchmark spec
# This provides a mapping from str -> list, where the key is the name of the
# benchmark spec and the value is a list of registered data loader classes for that spec.
# Currently we take the last registered loader added as the one to use.
VALIDATION_DATA_LOADER_REGISTRY = defaultdict(list)


class BenchmarkSpec:

    def __init__(self, data_dir, optimizer='adam', loss_function='binary_crossentropy', model_params={}, loss_params={}, optimizer_params={}, metrics={}, **kwargs):

        self.model_func = MODEL_FUNCS_REGISTRY[self.name][-1]

        data_loader_class = DATA_LOADER_REGISTRY[self.name][-1]
        validation_data_loader_class = VALIDATION_DATA_LOADER_REGISTRY[self.name][-1]

        self.data_loader = data_loader_class(data_dir / self.train_dir, **kwargs)
        self.validation_data_loader = validation_data_loader_class(data_dir / self.test_dir, **kwargs)

        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = metrics

        # Params for model objects
        self.model_params = model_params
        self.loss_params = loss_params
        self.optimizer_params = optimizer_params


class Benchmark:

    def __init__(self, spec):
        self._spec = spec


class TensorflowKerasBenchmark(Benchmark):

    def __init__(self, spec):
        super().__init__(spec)

    @property
    def data_loader(self):
        return self._spec.data_loader

    @property
    def validation_data_loader(self):
        return self._spec.validation_data_loader

    @property
    def model(self):
        return self._spec.model_func(self._spec.data_loader.input_shape, **self._spec.model_params)

    @property
    def loss(self):
        return tf.keras.losses.get(self._spec.loss_function, **self._spec.loss_params)

    @property
    def optimizer(self):
        return tf.keras.optimizers.get(self._spec.optimizer, **self._spec.optimizer_params)

    @property
    def metrics(self):
        return self._spec.metrics
