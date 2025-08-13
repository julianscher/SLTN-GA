from copy import deepcopy
import torch

from models.fc.mlp_embedded import MLPEmbedded
from models.fc.mlp import MLP
from utilities.fc.fc_utils import get_bit_vector_range_weight_matrix_form, get_number_of_connections_between_layers


def get_subnetwork(model, bit_vector, NN_architecture, device, secondary_model=None):
    subnet = deepcopy(model)

    if isinstance(subnet, (MLP, MLPEmbedded)):
        number_of_connections_between_layers = get_number_of_connections_between_layers(NN_architecture)
        for param in subnet.parameters():
            param.requires_grad = False

        if secondary_model is not None:
            for param in secondary_model.parameters():
                param.requires_grad = False

        affected_layers = []
        for name, _ in subnet.named_parameters():
            if "weight" in name.split("."):
                affected_layers.append(name)

        for idx, ele in enumerate(number_of_connections_between_layers):
            sub_ranges = get_bit_vector_range_weight_matrix_form(bit_vector, NN_architecture,
                                                                 number_of_connections_between_layers, idx)

            # Apply bit matrix to weight matrix
            subnet.state_dict()[affected_layers[idx]] *= torch.tensor(sub_ranges).to(device)

            if secondary_model is not None:
                # Invert bit-matrix sub-range
                sub_ranges_inverted = 1 - sub_ranges
                secondary_model_parameters = secondary_model.state_dict()[affected_layers[idx]]
                secondary_model_parameters_pruned = secondary_model_parameters * torch.tensor(sub_ranges_inverted).to(device)
                subnet.state_dict()[affected_layers[idx]] += secondary_model_parameters_pruned

    return subnet
