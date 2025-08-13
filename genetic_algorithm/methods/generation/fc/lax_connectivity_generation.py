import numpy as np

from genetic_algorithm.methods.generation.fc.base_generation import GenerationMethod
from utilities.fc.fc_bit_vector_operations import check_for_input_output_path
from utilities.fc.fc_utils import calculate_dimensionality
from utilities.ga_utils import randi


class LaxConnectivityGeneration(GenerationMethod):

    def __init__(self, nn_architecture):
        super(LaxConnectivityGeneration, self).__init__(nn_architecture=nn_architecture)
        self.nn_architecture = nn_architecture

    def generate(self):
        dimensionality = calculate_dimensionality(self.nn_architecture)
        not_input_output_path_exists = True

        while not_input_output_path_exists:
            # Create random bit_vector
            random_bit_vector = np.array([randi(min=0, max=2) for _ in range(dimensionality)])

            # Check if input-output-path exists
            if check_for_input_output_path(self.nn_architecture, random_bit_vector):
                return random_bit_vector