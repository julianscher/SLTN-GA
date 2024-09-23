import math
import numpy as np
from torch import nn
import sys
from pathlib import Path
import os

# Get relative path to neural network resources
sys.path.append(str(Path(__file__).parent))
rel_path = "resources/neural_network"
path = Path(os.path.join(Path(__file__).parent.parent.parent, rel_path))


def get_bit_vector_range_weight_matrix_form(bit_vector, NN_architecture, number_of_connections_between_layers, idx):
    layer_range = (sum(number_of_connections_between_layers[:idx]),
                   sum(number_of_connections_between_layers[:idx]) +
                   number_of_connections_between_layers[idx])

    bit_vector_range = bit_vector[layer_range[0]:layer_range[1]]

    # Get the corresponding connection ranges for every neuron from the layer before, that connects to a neuron
    # from this layer
    sub_ranges = np.array_split(bit_vector_range, NN_architecture[idx])
    sub_ranges = [i[:, np.newaxis] for i in sub_ranges]  # reshape matrices and get type 2 array
    sub_ranges = np.hstack([i for i in sub_ranges])

    return sub_ranges


def get_network_architecture(model: nn.Module):
    weight_tensors = []
    NN_architecture = []
    for l, tensor in enumerate(model.state_dict().values()):
        if l % 2 == 0:
            weight_tensors.append(tensor)

    for l, tensor in enumerate(weight_tensors):
        if l != len(weight_tensors) - 1:
            NN_architecture.append(tensor.shape[1])
        else:
            NN_architecture.append(tensor.shape[1])
            NN_architecture.append(tensor.shape[0])

    return NN_architecture


def determine_large_nn_architecture(NN_architecture_target_network, C=3, epsilon=0.01, delta=0.01):
    NN_architecture_large_NN = []
    dimensionality_weight_matrices = []
    for i in range(1, len(NN_architecture_target_network)):
        dimensionality_weight_matrices.append(
            (NN_architecture_target_network[i], NN_architecture_target_network[i - 1]))
    l = len(NN_architecture_target_network) - 1
    for (di, di_minus_1) in dimensionality_weight_matrices:
        intermediate_layer_dim = math.ceil(C * di_minus_1 * math.log2((di_minus_1 * di * l) / min(epsilon, delta)))
        NN_architecture_large_NN.extend([di_minus_1, intermediate_layer_dim])
    NN_architecture_large_NN.append(NN_architecture_target_network[-1])
    return NN_architecture_large_NN
