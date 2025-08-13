import numpy as np

from genetic_algorithm.methods.generation.fc.base_generation import GenerationMethod
from genetic_algorithm.methods.generation.fc.max_bit_vector_generation import MaxBitVectorGeneration
from utilities.fc.fc_bit_vector_operations import get_input_output_paths, activate_connections
from utilities.fc.fc_utils import calculate_dimensionality, get_number_of_connections_between_layers
from utilities.ga_utils import randi


class StrictConnectivityGeneration(GenerationMethod):

    def __init__(self, nn_architecture):
        super(StrictConnectivityGeneration, self).__init__(nn_architecture=nn_architecture)
        self.nn_architecture = nn_architecture
        self.input_output_paths = None


    def generate(self):
        dimensionality = calculate_dimensionality(self.nn_architecture)
        # Get input_output_paths of max()
        if self.input_output_paths is None:
            max_bit_vector = MaxBitVectorGeneration(self.nn_architecture).generate()
            input_output_paths = get_input_output_paths(self.nn_architecture, max_bit_vector)
            self.input_output_paths = input_output_paths
        max_connections_between_layers = max(get_number_of_connections_between_layers(self.nn_architecture))
        bit_vector = np.array([0 for _ in range(dimensionality)])

        # Choose how many input_output_paths the bit_vector should have
        count = randi(min=1, max=max_connections_between_layers + 1)

        # Select which input_output_paths the bit_vector should have
        random_indices = np.random.choice(np.arange(len(self.input_output_paths)), count, replace=False)
        input_output_paths_bit_vector = [self.input_output_paths[idx] for idx in random_indices]

        # Activate the input_output_paths in bit_vector
        for input_output_path in input_output_paths_bit_vector:
            bit_vector = activate_connections(bit_vector, self.nn_architecture, input_output_path)
        return bit_vector