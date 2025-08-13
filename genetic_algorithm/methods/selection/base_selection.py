from abc import ABC, abstractmethod

class SelectionMethod(ABC):
    """Abstract logging class for all selection strategies."""

    @abstractmethod
    def select(self, population, size):
        """Selects individuals from the population."""
        pass
