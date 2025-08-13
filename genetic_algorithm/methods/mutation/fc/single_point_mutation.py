from genetic_algorithm.methods.mutation.fc.base_mutation import MutationMethod
from genetic_algorithm.methods.mutation.fc.utils import apply_mutation
from utilities.fc.fc_bit_vector_operations import get_connections, check_for_input_output_path, \
    apply_connectivity_strictness_level


class SinglePointMutation(MutationMethod):
    """ Mutate random bit, then check if there is still an input_output_path """

    def __init__(self, nn_architecture, connectivity_scheme: str = "lax"):
        super(SinglePointMutation, self).__init__(nn_architecture=nn_architecture, connectivity_scheme=connectivity_scheme)
        self.nn_architecture = nn_architecture
        self.connectivity_scheme = connectivity_scheme

    def mutate(self, genome, **kwargs):
        connections = get_connections(self.nn_architecture, genome, True)

        while True:
            mutant = apply_mutation(genome.copy(), self.nn_architecture, connections)

            # Check if resulting bit_vector has an input_output_path
            if check_for_input_output_path(self.nn_architecture, mutant):
                return apply_connectivity_strictness_level(mutant, self.nn_architecture, self.connectivity_scheme)