import numpy as np

from genetic_algorithm.methods.recombination.fc.base_recombination import RecombinationMethod
from utilities.fc.fc_bit_vector_operations import check_for_input_output_path, apply_connectivity_strictness_level


class RandomPointCrossoverRecombination(RecombinationMethod):

    def __init__(self, nn_architecture, connectivity_scheme: str = "lax"):
        super(RandomPointCrossoverRecombination, self).__init__(nn_architecture=nn_architecture, connectivity_scheme=connectivity_scheme)
        self.nn_architecture = nn_architecture
        self.connectivity_scheme = connectivity_scheme

    def recombine(self, genome1, genome2, **kwargs):
        while True:
            mask = np.random.randint(2, size=len(genome1))

            # Create offspring using the mask: take from genome1 where mask is 0, and from genome2 where mask is 1
            child = (genome1 & ~mask) | (genome2 & mask)

            # Test if child contains input-output-path
            if check_for_input_output_path(self.nn_architecture, child):
                return apply_connectivity_strictness_level(child, self.nn_architecture, self.connectivity_scheme)