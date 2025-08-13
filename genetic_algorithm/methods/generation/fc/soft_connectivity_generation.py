import numpy as np

from genetic_algorithm.methods.generation.fc.base_generation import GenerationMethod
from utilities.fc.fc_bit_vector_operations import check_for_input_output_path, remove_floating_connections
from utilities.fc.fc_utils import calculate_dimensionality
from utilities.ga_utils import randi


class SoftConnectivityGeneration(GenerationMethod):

    def __init__(self, nn_architecture):
        super(SoftConnectivityGeneration, self).__init__(nn_architecture=nn_architecture)
        self.nn_architecture = nn_architecture

    def generate(self):
        dimensionality = calculate_dimensionality(self.nn_architecture)
        ones_count_min = len(self.nn_architecture) - 1  # the number of 1s in a min()

        while True:
            bit_vector = np.array([1 for _ in range(dimensionality)])
            # Prune random weights
            count = randi(min=0, max=dimensionality - ones_count_min)
            random_indices = np.random.choice(np.arange(dimensionality), count, replace=False)
            bit_vector = [0 if idx in random_indices else bit for idx, bit in enumerate(bit_vector)]

            # Check if input-output-path still exists -> Instead, we could also use evaluate_bit_vector_accuracy
            if check_for_input_output_path(self.nn_architecture, bit_vector):
                break

        # Remove floating connections from bit_vector
        bit_vector = remove_floating_connections(self.nn_architecture, bit_vector)

        return bit_vector