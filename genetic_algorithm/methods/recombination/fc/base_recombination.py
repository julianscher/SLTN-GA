from abc import ABC, abstractmethod

class RecombinationMethod(ABC):
    """Abstract logging class for all crossover recombination strategies."""

    def __init__(self, **kwargs):
        """Optional parameters can be stored in subclasses if needed."""
        self.params = kwargs  # Stores any extra parameters for flexibility

    @abstractmethod
    def recombine(self, genome1, genome2, **kwargs):
        """Performs crossover and returns the offspring genome."""
        pass