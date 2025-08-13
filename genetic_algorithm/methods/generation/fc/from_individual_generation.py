import numpy as np

from genetic_algorithm.methods.generation.fc.base_generation import GenerationMethod


class FromIndividualGeneration(GenerationMethod):

    def __init__(self, bit_vector):
        super(FromIndividualGeneration, self).__init__(bit_vector = bit_vector)
        self.bit_vector = bit_vector

    def generate(self, **kwargs):
        sampled_bit_vector = np.array([0 for _ in range(len(self.bit_vector))])
        for idx, bit in enumerate(self.bit_vector):
            if bit == 1:
                if np.random.random() < 0.5:
                    sampled_bit_vector[idx] = 1

        return sampled_bit_vector