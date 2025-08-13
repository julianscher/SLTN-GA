import abc

class Dataset(abc.ABC):
    def __init__(self):
        pass

    @property
    def name(self):
        return self.__class__.__name__

    def plot_dataset(self, train_dataset, **kwargs):
        pass

    def plot_learned_function(self, train_dataset, model, **kwargs):
        pass