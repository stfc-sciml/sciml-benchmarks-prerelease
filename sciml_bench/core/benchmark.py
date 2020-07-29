from collections import defaultdict
from sciml_bench.core.bench_logger import LOGGER
from abc import ABC, abstractmethod
import tensorflow as tf

# Registry of model functions for each benchmark spec
# This provides a mapping from str -> list, where the key is the name of the
# benchmark spec and the value is a list of registered functions for that spec.
# Currently we take the last function added as the one to use.
BENCHMARK_REGISTRY = defaultdict(list)


def create_benchmark(name):
    if name not in BENCHMARK_REGISTRY or len(BENCHMARK_REGISTRY[name]) == 0:
        raise RuntimeError("Benchmark {} does not exist in registry!".format(name))

    benchmark_cls = BENCHMARK_REGISTRY[name][-1]
    LOGGER.debug('Benchmark implementation is {}'.format(benchmark_cls.__name__))
    benchmark = benchmark_cls()
    return benchmark


def register_benchmark(cls):
    if not hasattr(cls, 'name'):
        raise RuntimeError('Cannot register benchmark without name property!')

    BENCHMARK_REGISTRY[cls.name] = cls
    return cls


class Benchmark(ABC):

    epochs = None
    loss = None
    batch_size = None
    optimizer = 'adam'

    train_dir = ''
    test_dir = ''

    loss_params={}
    optimizer_params={}
    fit_params={}
    metrics=[]

    def __init__(self, **config):
        for name, value in config.items():
            if not hasattr(self, name):
                setattr(self, name, value)
            else:
                attribute = getattr(self, name)
                if isinstance(attribute, list):
                    attribute.extend(value)
                    setattr(self, name, attribute)
                if isinstance(attribute, dict):
                    attribute.update(value)
                    setattr(self, name, attribute)
                else:
                    setattr(self, name, value)

        self.data_loader_ = self.data_loader(**config)
        self.validation_data_loader_ = self.validation_data_loader(**config)

    @abstractmethod
    def model(self, *arg, **kwargs):
        pass

    @abstractmethod
    def data_loader(self, *args, **kwargs):
        pass

    def validation_data_loader(self, *args, **kwargs):
        # Default implementation just forwards everything to the same data
        # data loader as for the training data
        return self.data_loader(*args, **kwargs)


class TensorflowKerasMixin:

    @property
    def loss_(self) -> tf.keras.losses.Loss:
        if isinstance(self.loss, str):
            loss = tf.keras.losses.get(self.loss)
            if hasattr(loss, 'get_config'):
                cfg = loss.get_config()
                cfg.update(self.spec.loss_params)
                loss = loss.from_config(cfg)
        else:
            loss = self.loss
        return loss

    @property
    def optimizer_(self) -> tf.keras.optimizers.Optimizer:
        opt = tf.keras.optimizers.get(self.optimizer)
        cfg = opt.get_config()
        cfg.update(self.optimizer_params)
        opt = opt.from_config(cfg)
        return opt
