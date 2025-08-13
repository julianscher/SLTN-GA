from abc import ABC, abstractmethod


class PopulationEvaluationMethod(ABC):
    """Abstract base class for population and individual evaluation."""

    def __init__(self, **kwargs):
        """Optional parameters can be stored in subclasses if needed."""
        self.params = kwargs

    @abstractmethod
    def evaluate_population(self, population, evaluation_cache):
        """Evaluates a population using a specific evaluation strategy."""
        pass