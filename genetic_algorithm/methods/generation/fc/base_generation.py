from abc import ABC, abstractmethod


class GenerationMethod(ABC):
    """Abstract logging class for all individual generation strategies."""

    def __init__(self, **kwargs):
        """Optional parameters can be stored in subclasses if needed."""
        self.params = kwargs

    @abstractmethod
    def generate(self, **kwargs):
        """Generates a new individual."""
        pass