from abc import ABC, abstractmethod


class MutationMethod(ABC):
    """Abstract logging class for all mutation strategies."""

    def __init__(self, **kwargs):
        """Optional parameters can be stored in subclasses if needed."""
        self.params = kwargs

    @abstractmethod
    def mutate(self, genome, **kwargs):
        """Performs mutation and returns the mutated genome."""
        pass