import numpy as np
from genetic_algorithm.methods.recombination.fc.base_recombination import RecombinationMethod
from utilities.fc.fc_bit_vector_operations import get_input_output_paths, get_bit_position, \
    apply_connectivity_strictness_level
from utilities.ga_utils import randi


class FixedPathCrossoverRecombination(RecombinationMethod):

    def __init__(self, nn_architecture, connectivity_scheme: str = "lax"):
        super(FixedPathCrossoverRecombination, self).__init__(nn_architecture=nn_architecture, connectivity_scheme=connectivity_scheme)
        self.nn_architecture = nn_architecture
        self.connectivity_scheme = connectivity_scheme

    def recombine(self, genome1, genome2, **kwargs):
        # Choose random parent to inherit input-output-path from
        basis = [genome1, genome2][randi(min=0, max=2)].copy()

        # Get positions of bits of one input-output-path
        paths = get_input_output_paths(self.nn_architecture, basis)
        random_idx = randi(min=0, max=len(paths))
        path = paths[random_idx]
        pos = []
        for (layer, conn) in path:
            pos.append(get_bit_position(self.nn_architecture, layer, conn))

        child = np.array([np.random.choice([genome1[idx], genome2[idx]]) if idx not in pos
                          else basis[idx] for idx in range(len(basis))])

        return apply_connectivity_strictness_level(child, self.nn_architecture, self.connectivity_scheme)
