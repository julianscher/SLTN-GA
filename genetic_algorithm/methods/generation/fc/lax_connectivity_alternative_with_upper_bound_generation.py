import numpy as np

from genetic_algorithm.methods.generation.fc.base_generation import GenerationMethod
from utilities.fc.fc_bit_vector_operations import check_if_bit_vector_has_accuracy_in_bound
from utilities.fc.fc_utils import calculate_dimensionality
from utilities.ga_utils import randi


class LaxConnectivityAlternativeWithUpperBoundGeneration(GenerationMethod):

    def __init__(self, nn_architecture, device, lower_bound, upper_bound, train_loader, model):
        super(LaxConnectivityAlternativeWithUpperBoundGeneration, self).__init__(nn_architecture=nn_architecture, device=device, lower_bound=lower_bound, upper_bound=upper_bound, model=model)
        self.nn_architecture = nn_architecture
        self.device = device
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.model = model
        self.train_loader = train_loader

    def generate(self):
        dimensionality = calculate_dimensionality(self.nn_architecture)

        while True:
            # Create random bit_vector
            random_bit_vector = np.array([randi(min=0, max=2) for _ in range(dimensionality)])

            # Check if bit_vector's accuracy is in bound of lower and upper bound
            if check_if_bit_vector_has_accuracy_in_bound(random_bit_vector, self.lower_bound, self.upper_bound,
                                                         self.model, self.train_loader, self.device):
                return random_bit_vector
