from copy import deepcopy
import torch
from src.genetic_algorithm.helper_functions import get_number_of_connections_between_layers
from src.neural_network.helper_functions import path
from src.neural_network.helper_functions import get_bit_vector_range_weight_matrix_form
from src.neural_network.nn_framework import NNFramework


class Subnetwork(NNFramework):
    def __init__(self, bit_vector, NN_architecture, model):
        super().__init__(NN_architecture)
        self.NN_architecture = NN_architecture
        self.model = deepcopy(model)
        for param in self.model.parameters():
            param.requires_grad = False
        self.mask(bit_vector)

    def mask(self, bit_vector):
        number_of_connections_between_layers = get_number_of_connections_between_layers(self.NN_architecture)
        for idx, ele in enumerate(number_of_connections_between_layers):
            sub_ranges = get_bit_vector_range_weight_matrix_form(bit_vector, self.NN_architecture,
                                                                 number_of_connections_between_layers, idx)

            # Apply bit matrix to weight matrix
            current_layer = getattr(self.model, f"layer{idx + 1}")
            current_layer.weight.data *= torch.tensor(sub_ranges)

    def save(self):
        # save model parameters
        torch.save(self.model.state_dict(), f"{path}/model_parameters_{str(self.NN_architecture)}_pruned.pth")