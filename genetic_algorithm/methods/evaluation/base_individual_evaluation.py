from abc import ABC, abstractmethod


class IndividualEvaluationMethod(ABC):
    """Abstract base class for population and individual evaluation."""

    def __init__(self, **kwargs):
        """Optional parameters can be stored in subclasses if needed."""
        self.params = kwargs

    @abstractmethod
    def evaluate_individual(self, bit_vector, evaluation_cache):
        """Evaluates a single bit vector (genome)."""
        pass