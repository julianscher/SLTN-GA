import numpy as np
from genetic_algorithm.methods.recombination.fc.base_recombination import RecombinationMethod
from utilities.fc.fc_bit_vector_operations import get_input_output_paths, activate_connections, \
    apply_connectivity_strictness_level
from utilities.fc.fc_utils import calculate_dimensionality
from utilities.ga_utils import randi


class InputOutputPathCrossoverRecombination(RecombinationMethod):

    def __init__(self, nn_architecture, connectivity_scheme: str = "lax"):
        super(InputOutputPathCrossoverRecombination, self).__init__(nn_architecture=nn_architecture, connectivity_scheme=connectivity_scheme)
        self.nn_architecture = nn_architecture
        self.connectivity_scheme = connectivity_scheme

    def recombine(self, genome1, genome2, **kwargs):
        child = np.array(["?" for _ in range(calculate_dimensionality(self.nn_architecture))])

        # For both parent genomes, get all their input_output_paths
        input_output_paths_genome1 = get_input_output_paths(self.nn_architecture, genome1)
        input_output_paths_genome2 = get_input_output_paths(self.nn_architecture, genome2)

        # Choose how many of the input_output_paths to inherit from each parent
        # input_output_paths_genome1 and input_output_paths_genome2 contain at least 1 path each
        count1 = randi(min=0, max=len(input_output_paths_genome1))
        if count1 == 0:
            if len(input_output_paths_genome2) == 1:
                count2 = 1
            else:
                count2 = randi(min=1, max=len(input_output_paths_genome2))
        else:
            count2 = randi(min=0, max=len(input_output_paths_genome2))

        # Get random indices
        random_indices1 = np.random.choice(np.arange(len(input_output_paths_genome1)), count1, replace=False)
        random_indices2 = np.random.choice(np.arange(len(input_output_paths_genome2)), count2, replace=False)

        # Get selected input_output_paths
        input_output_paths_child = [input_output_paths_genome1[idx] for idx in random_indices1] + \
                                   [input_output_paths_genome2[idx] for idx in random_indices2]

        # Activate the input_output_paths in child
        for input_output_path in input_output_paths_child:
            child = activate_connections(child, self.nn_architecture, input_output_path)

        # Randomly assign values from parent genomes to remaining child bits that are not part of an input_output_path
        for idx, bit in enumerate(child):
            if bit == "?":
                child[idx] = genome1[idx] if randi(min=0, max=2) else genome2[idx]

        return apply_connectivity_strictness_level(child, self.nn_architecture, self.connectivity_scheme)