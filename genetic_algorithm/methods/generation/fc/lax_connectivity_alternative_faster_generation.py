import numpy as np
from accelerate import data_loader

from genetic_algorithm.methods.generation.fc.base_generation import GenerationMethod
from utilities.fc.fc_bit_vector_operations import check_for_input_output_path, \
    check_if_bit_vector_passes_accuracy_filter
from utilities.ga_utils import randi


class LaxConnectivityAlternativeFasterGeneration(GenerationMethod):

    def __init__(self, nn_architecture, device, bound, train_loader, model):
        super(LaxConnectivityAlternativeFasterGeneration, self).__init__(nn_architecture=nn_architecture, device=device, bound=bound, model=model)
        self.nn_architecture = nn_architecture
        self.device = device
        self.bound = bound
        self.model = model
        self.train_loader = train_loader

    def generate(self):
        dimensionality = self.model.number_of_parameters_to_be_masked()

        number_of_unsuccessful_tries = 0
        while True:
            # Create random bit_vector
            random_bit_vector = np.array([randi(min=0, max=2) for _ in range(dimensionality)])

            # Check if bit_vector's accuracy has at least the boundary value
            if number_of_unsuccessful_tries <= 500 \
                    and check_if_bit_vector_passes_accuracy_filter(random_bit_vector, self.bound, self.model,
                                                                   self.train_loader, self.device):
                return random_bit_vector
            elif 500 < number_of_unsuccessful_tries <= 1000 and self.bound - 0.1 >= 0.6 and \
                    check_if_bit_vector_passes_accuracy_filter(random_bit_vector, 0.6, self.model,
                                                               self.train_loader, self.device):
                return random_bit_vector
            elif 1000 < number_of_unsuccessful_tries <= 1500 and self.bound >= 0.51 and \
                    check_if_bit_vector_passes_accuracy_filter(random_bit_vector, 0.51, self.model,
                                                               self.train_loader, self.device):
                return random_bit_vector
            elif number_of_unsuccessful_tries > 1500 and check_for_input_output_path(self.nn_architecture,
                                                                                     random_bit_vector):
                return random_bit_vector

            number_of_unsuccessful_tries += 1