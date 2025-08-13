import numpy as np

from genetic_algorithm.methods.recombination.fc.base_recombination import RecombinationMethod


class RandomPointCrossoverNegligentRecombination(RecombinationMethod):

    def recombine(self, genome1, genome2, **kwargs):
        # strong recombination: produce new genome that inherits from either of the parent genomes
        """
        dimensionality = problem_instance.dimensionality

        probabilities = torch.rand(dimensionality)
        take_from_genome1 = probabilities < 0.5
        take_from_genome1 = take_from_genome1.int()
        take_from_genome2 = torch.ones(dimensionality).int() - take_from_genome1
        genome1_tensor = torch.from_numpy(genome1)
        genome2_tensor = torch.from_numpy(genome2)
        masked_genome1 = take_from_genome1 * genome1_tensor
        masked_genome2 = take_from_genome2 * genome2_tensor
        child = masked_genome1 + masked_genome2
        child = child.detach().cpu().numpy()
        """

        # for every dim with some probability
        mask = np.random.randint(2, size=len(genome1))

        # Create offspring using the mask: take from genome1 where mask is 0, and from genome2 where mask is 1
        child = (genome1 & ~mask) | (genome2 & mask)
        return child