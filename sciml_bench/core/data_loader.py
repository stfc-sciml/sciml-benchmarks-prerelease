from abc import abstractmethod, ABCMeta


class DataLoader():
    """Base class for data loaders

    This defines the interface that new data loaders must adhere to
    """
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def input_shape(self):
        pass

    @property
    @abstractmethod
    def output_shape(self):
        pass

    @abstractmethod
    def to_dataset(self):
        pass
