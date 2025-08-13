from genetic_algorithm.methods.evaluation.base_population_evaluation import PopulationEvaluationMethod


class PopulationPerformanceEvaluation(PopulationEvaluationMethod):
    """Evaluates individuals based on performance objective."""

    def __init__(self, performance_maximizing):
        super(PopulationPerformanceEvaluation, self).__init__(performance_maximizing=performance_maximizing)
        self.performance_maximizing = performance_maximizing

    def evaluate_population(self, population, evaluation_cache):
        population.individuals.sort(
            key=lambda ind: ind.evaluate(),
            reverse=self.performance_maximizing
        )
        return population