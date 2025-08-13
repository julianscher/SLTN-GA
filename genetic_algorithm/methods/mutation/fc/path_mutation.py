import numpy as np

from genetic_algorithm.methods.mutation.fc.base_mutation import MutationMethod
from utilities.fc.fc_bit_vector_operations import get_random_path_through_network, activate_connections, \
    apply_connectivity_strictness_level, deactivate_connections, check_for_input_output_path


class PathMutation(MutationMethod):

    def __init__(self, nn_architecture, connectivity_scheme: str = "lax", alpha=0.5):
        super(PathMutation, self).__init__(nn_architecture=nn_architecture, connectivity_scheme=connectivity_scheme)
        self.nn_architecture = nn_architecture
        self.connectivity_scheme = connectivity_scheme
        self.alpha = alpha

    def mutate(self, genome, **kwargs):
        mutant = genome.copy()
        while True:
            r = np.random.random()

            # Get random path through network (can contain deactivated connections)
            random_path = get_random_path_through_network(self.nn_architecture)

            # Activate or deactivate all connections on random_path
            if r < self.alpha:
                mutant = activate_connections(mutant, self.nn_architecture, random_path)
                return apply_connectivity_strictness_level(mutant, self.nn_architecture, self.connectivity_scheme)
            else:
                mutant = deactivate_connections(mutant, self.nn_architecture, random_path)
                if check_for_input_output_path(self.nn_architecture, mutant):
                    return apply_connectivity_strictness_level(mutant, self.nn_architecture, self.connectivity_scheme)