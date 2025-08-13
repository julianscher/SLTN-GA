import numpy as np

from genetic_algorithm.methods.generation.fc.base_generation import GenerationMethod
from utilities.fc.fc_utils import calculate_dimensionality


class MaxBitVectorGeneration(GenerationMethod):

    def __init__(self, nn_architecture):
        super(MaxBitVectorGeneration, self).__init__(nn_architecture=nn_architecture)
        self.nn_architecture = nn_architecture

    def generate(self):
        dimensionality = calculate_dimensionality(self.nn_architecture)
        return np.ones(dimensionality, dtype=int)