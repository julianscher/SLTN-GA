import numpy as np

from genetic_algorithm.methods.mutation.fc.base_mutation import MutationMethod
from utilities.fc.fc_bit_vector_operations import get_neuron_connections, activate_connections, \
    check_for_input_output_path, apply_connectivity_strictness_level, deactivate_connections
from utilities.fc.fc_graph_operations import get_list_of_all_vertices
from utilities.fc.fc_utils import calculate_dimensionality
from utilities.ga_utils import randi


class NeuronMutation(MutationMethod):

    def __init__(self, nn_architecture, connectivity_scheme: str = "lax", alpha = 0.5):
        super(NeuronMutation, self).__init__(nn_architecture=nn_architecture, connectivity_scheme=connectivity_scheme)
        self.nn_architecture = nn_architecture
        self.connectivity_scheme = connectivity_scheme
        self.alpha = alpha

    def mutate(self, genome, **kwargs):
        mutant = genome.copy()

        # a neuron is defined by (layer, neuron_number)
        neurons = get_list_of_all_vertices(self.nn_architecture)
        valid_neurons = [neuron for neuron in neurons if neuron[0] % 2 != 0]

        while True:
            # Pick random neuron from valid_neurons
            random_idx = randi(min=0, max=len(valid_neurons))
            neuron = valid_neurons[random_idx]

            # Get all incoming and outgoing connections
            connections = get_neuron_connections(self.nn_architecture,
                                                 [1 for _ in range(calculate_dimensionality(self.nn_architecture))], neuron)

            r = np.random.random()

            # Activate or deactivate all incoming or outgoing connections of the neuron
            if r < self.alpha:
                mutant = activate_connections(mutant, self.nn_architecture, connections)
                if check_for_input_output_path(self.nn_architecture, mutant):
                    return apply_connectivity_strictness_level(mutant, self.nn_architecture, self.connectivity_scheme)
            else:
                mutant = deactivate_connections(mutant, self.nn_architecture, connections)
                if check_for_input_output_path(self.nn_architecture, mutant):
                    return apply_connectivity_strictness_level(mutant, self.nn_architecture, self.connectivity_scheme)