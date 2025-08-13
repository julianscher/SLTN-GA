import numpy as np

from genetic_algorithm.methods.generation.fc.base_generation import GenerationMethod
from utilities.fc.fc_bit_vector_operations import check_if_bit_vector_passes_accuracy_filter
from utilities.ga_utils import randi


class LaxConnectivityAlternativeGeneration(GenerationMethod):

    def __init__(self, nn_architecture, device, bound, train_loader, model):
        super(LaxConnectivityAlternativeGeneration, self).__init__(nn_architecture=nn_architecture, device=device, bound=bound, model=model)
        self.nn_architecture = nn_architecture
        self.device = device
        self.bound = bound
        self.model = model
        self.train_loader = train_loader

    def generate(self):
        dimensionality = self.model.number_of_parameters_to_be_masked()

        while True:
            # Create random bit_vector
            random_bit_vector = np.array([randi(min=0, max=2) for _ in range(dimensionality)])

            # Check if bit_vector's accuracy has at least the boundary value
            if check_if_bit_vector_passes_accuracy_filter(random_bit_vector, self.bound, self.model, self.train_loader,
                                                                 self.device):
                return random_bit_vector