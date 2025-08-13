import numpy as np
from genetic_algorithm.methods.recombination.fc.base_recombination import RecombinationMethod
from utilities.fc.fc_bit_vector_operations import get_neuron_connections, activate_connections, \
    check_for_input_output_path, apply_connectivity_strictness_level
from utilities.fc.fc_graph_operations import get_list_of_all_vertices
from utilities.fc.fc_utils import calculate_dimensionality
from utilities.ga_utils import randi


class NeuronCrossoverRecombination(RecombinationMethod):

    def __init__(self, nn_architecture, connectivity_scheme: str = "lax"):
        super(NeuronCrossoverRecombination, self).__init__(nn_architecture=nn_architecture, connectivity_scheme=connectivity_scheme)
        self.nn_architecture = nn_architecture
        self.connectivity_scheme = connectivity_scheme

    def recombine(self, genome1, genome2, **kwargs):
        neurons = get_list_of_all_vertices(self.nn_architecture)
        valid_neurons = [neuron for neuron in neurons if neuron[0] % 2 != 0]
        child = np.array([0 for _ in range(calculate_dimensionality(self.nn_architecture))])

        while True:
            # Pick randomly valid neuron from one of the parents
            for neuron in valid_neurons:
                neuron_giver = genome1 if randi(min=0, max=2) else genome2

                # Get all activated incoming and outgoing connections of neuron
                connections = get_neuron_connections(self.nn_architecture, neuron_giver, neuron)

                # Activate these connections in child genome
                child = activate_connections(child, self.nn_architecture, connections)

            if check_for_input_output_path(self.nn_architecture, child):
                return apply_connectivity_strictness_level(child, self.nn_architecture, self.connectivity_scheme)