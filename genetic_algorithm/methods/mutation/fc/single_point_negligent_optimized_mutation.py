from genetic_algorithm.methods.mutation.fc.base_mutation import MutationMethod
from utilities.ga_utils import randi


class SinglePointNegligentOptimizedMutation(MutationMethod):
    """ Mutate random bit"""

    def mutate(self, genome, **kwargs):
        random_idx = randi(min=0, max=len(genome))
        bit = genome[random_idx]
        mutant = genome.copy()
        if bit:
            mutant[random_idx] = 0
        else:
            mutant[random_idx] = 1
        return mutant
