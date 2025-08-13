import numpy as np

from genetic_algorithm.methods.evaluation.base_population_evaluation import PopulationEvaluationMethod


class PopulationRankingEvaluation(PopulationEvaluationMethod):
    """Evaluates population by sorting based on performance and sparsity."""

    def __init__(self, performance_maximizing, sparsity_maximizing):
        super(PopulationRankingEvaluation, self).__init__(performance_maximizing=performance_maximizing, sparsity_maximizing=sparsity_maximizing)
        self.performance_maximizing = performance_maximizing
        self.sparsity_maximizing = sparsity_maximizing

    def evaluate_population(self, population, evaluation_cache):
        population.individuals = sorted(
            population.individuals,
            key=lambda ind: (
                -ind.evaluate() if self.performance_maximizing else ind.evaluate(),
                -np.count_nonzero(ind.genome == 0) if self.sparsity_maximizing else np.count_nonzero(ind.genome == 0)
            )
        )
        return population