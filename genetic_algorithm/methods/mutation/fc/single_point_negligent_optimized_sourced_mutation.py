import numpy as np

from genetic_algorithm.methods.mutation.fc.base_mutation import MutationMethod


class SinglePointNegligentOptimizedSourcedMutation(MutationMethod):
    """ Mutate random bit without leaving the scope """

    def __init__(self, source):
        super(SinglePointNegligentOptimizedSourcedMutation, self).__init__(source=source)
        self.source = source

    def mutate(self, genome, **kwargs):
        ones_indices = [i for i in range(0, len(self.source)) if self.source[i] == 1]
        random_idx = np.random.choice(ones_indices)
        bit = genome[random_idx]
        mutant = genome.copy()
        if bit:
            mutant[random_idx] = 0
        else:
            mutant[random_idx] = 1
        return mutant
