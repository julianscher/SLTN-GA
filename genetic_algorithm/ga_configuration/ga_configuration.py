from dataclasses import dataclass

from genetic_algorithm.methods.evaluation.base_population_evaluation import PopulationEvaluationMethod
from genetic_algorithm.methods.evaluation.fc.hashing import FitnessHashTable
from genetic_algorithm.methods.generation.fc.base_generation import GenerationMethod
from genetic_algorithm.methods.mutation.fc.base_mutation import MutationMethod
from genetic_algorithm.methods.recombination.fc.base_recombination import RecombinationMethod
from genetic_algorithm.methods.selection.base_selection import SelectionMethod


@dataclass
class GAConfiguration:
    generation: GenerationMethod
    selection: SelectionMethod
    mutation: MutationMethod
    recombination: RecombinationMethod
    survivor_evaluation: PopulationEvaluationMethod
    parents_evaluation: PopulationEvaluationMethod
    individual_evaluation: PopulationEvaluationMethod
    evaluation_cache: FitnessHashTable

    def generate(self):
        # generate one valid genome that conforms to the problem requirements
        return self.generation.generate()

    def mutate(self, genome):
        return self.mutation.mutate(genome)

    def recombine(self, genome1, genome2):
        return self.recombination.recombine(genome1, genome2)

    def select(self, population, size):
        return self.selection.select(population, size)

    def evaluate_survivors(self, population):
        return self.survivor_evaluation.evaluate_population(population, self.evaluation_cache)

    def evaluate_parents(self, population):
        return self.parents_evaluation.evaluate_population(population, self.evaluation_cache)

    def evaluate_individual(self, genome):
        return self.individual_evaluation.evaluate_individual(genome, self.evaluation_cache)
