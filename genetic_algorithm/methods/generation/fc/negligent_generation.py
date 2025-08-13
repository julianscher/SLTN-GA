import torch

from genetic_algorithm.methods.generation.fc.base_generation import GenerationMethod


class NegligentGeneration(GenerationMethod):

    def __init__(self, model, prune_rate_init):
        super(NegligentGeneration, self).__init__(model=model, prune_rate_init=prune_rate_init)
        self.model = model
        self.prune_rate_init = prune_rate_init

    def generate(self):
        dimensionality = self.model.number_of_parameters_to_be_masked()
        random_probabilities = torch.rand(dimensionality)  # Create tensor with random floats between 0.0 and 1.0
        boolean_vector = random_probabilities > self.prune_rate_init  # Create boolean tensor where all values < prune_rate_init are True  else False (pruned)
        bit_vector = boolean_vector.int()  # convert boolean tensor to bit-tensor
        bit_vector = bit_vector.detach().cpu().numpy()

        return bit_vector
