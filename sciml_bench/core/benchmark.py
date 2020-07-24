import tensorflow as tf


class BenchmarkSpec:

    def __init__(self, model_func, data_loader, validation_data_loader, optimizer='adam', loss_function='binary_crossentropy', model_params={}, loss_params={}, optimizer_params={}, metrics={}):
        self.data_loader = data_loader
        self.validation_data_loader = validation_data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model_func = model_func
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
