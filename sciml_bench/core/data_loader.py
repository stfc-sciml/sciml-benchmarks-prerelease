from abc import abstractmethod, ABCMeta

class DataLoader():
    """Base class for data loaders

    This defines the interface that new data loaders must adhere to
    """
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def dimensions(self):
        pass

    @property
    @abstractmethod
    def train_size(self):
        pass

    @property
    @abstractmethod
    def test_size(self):
        pass

    @abstractmethod
    def train_fn(self):
        pass

    @abstractmethod
    def test_fn(self):
        pass
