from genetic_algorithm.methods.mutation.fc.base_mutation import MutationMethod
from genetic_algorithm.methods.mutation.fc.utils import apply_mutation
from utilities.fc.fc_bit_vector_operations import get_valid_positions, apply_connectivity_strictness_level


class SinglePointPathFixedMutation(MutationMethod):

    def __init__(self, nn_architecture, connectivity_scheme: str = "lax"):
        super(SinglePointPathFixedMutation, self).__init__(nn_architecture=nn_architecture, connectivity_scheme=connectivity_scheme)
        self.nn_architecture = nn_architecture
        self.connectivity_scheme = connectivity_scheme

    def mutate(self, genome, **kwargs):
        connections = get_valid_positions(self.nn_architecture, genome)
        mutant = apply_mutation(genome.copy(), self.nn_architecture, connections)
        return apply_connectivity_strictness_level(mutant, self.nn_architecture, self.connectivity_scheme)
