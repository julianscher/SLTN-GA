from genetic_algorithm.methods.mutation.fc.base_mutation import MutationMethod
from genetic_algorithm.methods.mutation.fc.utils import perform_bit_flip
from utilities.fc.fc_bit_vector_operations import get_random_path_through_network, check_for_input_output_path, \
    apply_connectivity_strictness_level


class PathFlipMutation(MutationMethod):

    def __init__(self, nn_architecture, connectivity_scheme: str = "lax"):
        super(PathFlipMutation, self).__init__(nn_architecture=nn_architecture, connectivity_scheme=connectivity_scheme)
        self.nn_architecture = nn_architecture
        self.connectivity_scheme = connectivity_scheme

    def mutate(self, genome, **kwargs):
        while True:
            mutant = genome.copy()

            # Get random path through network (can contain deactivated connections)
            random_path = get_random_path_through_network(self.nn_architecture)

            # Perform bit-flip mutation on path
            for layer, conn in random_path:
                mutant = perform_bit_flip(mutant, self.nn_architecture, layer, conn)

            # Check if resulting bit_vector has an input_output_path
            if check_for_input_output_path(self.nn_architecture, mutant):
                return apply_connectivity_strictness_level(mutant, self.nn_architecture, self.connectivity_scheme)