import numpy as np

from models.fc.mlp import MLP


def get_bit_vector_range_weight_matrix_form(bit_vector, NN_architecture, number_of_connections_between_layers, idx):
    # Get range of connections [start, finish) in bit-vector that correspond to the current layer
    layer_range = (sum(number_of_connections_between_layers[:idx]),
                   sum(number_of_connections_between_layers[:idx]) +
                   number_of_connections_between_layers[idx])

    # Get actual bits in that layer range
    bit_vector_range = bit_vector[layer_range[0]:layer_range[1]]

    # Get the corresponding connection ranges for every neuron from the layer before, that connects to a neuron
    # from this layer
    # Split layer range in sub-ranges containing the incoming connections for individual neurons
    sub_ranges = bit_vector_range.reshape(-1, NN_architecture[idx+1])
    sub_ranges = [i[:, np.newaxis] for i in sub_ranges]  # reshape matrices and get type 2 array
    sub_ranges = np.hstack([i for i in sub_ranges])

    return sub_ranges


def get_NN_architecture_from_model(model):
    NN_architecture = []
    if isinstance(model, MLP):
        parameters = []
        for name, parameter in model.named_parameters():
            if name.count("weight") > 0:
                parameters.append(parameter)
        for idx, parameter in enumerate(parameters):
            NN_architecture.append(parameter.shape[1])
            if idx == len(parameters) - 1:
                NN_architecture.append(parameter.shape[0])

    print("NN_architecture: ", NN_architecture)

    return NN_architecture


def get_net_architecture(model_name, architecture):
    if model_name in ["MLP", "MLPEmbedded"]:
        return architecture


def get_number_of_connections_between_layers(NN_architecture):
    if len(NN_architecture) > 1:
        return [NN_architecture[i] * NN_architecture[i + 1]
                for i in range(len(NN_architecture) - 1)]
    elif len(NN_architecture) == 1:
        return NN_architecture[0]
    else:
        return 0


def calculate_dimensionality(NN_architecture):
    num = get_number_of_connections_between_layers(NN_architecture)
    if type(num) == int:
        return num
    else:
        return sum(num)
