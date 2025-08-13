import copy

import numpy as np

from models.fc.mlp import MLP
from utilities.fc.fc_utils import get_NN_architecture_from_model
from utilities.fc.fc_visualize_subnetwork import DrawSubnetwork


class MLPEmbedded(MLP):
    """ Create network with ReLU activation functions which has a trained network randomly embedded """

    def __init__(self, NN_architecture, init, prune_rate, trained_model, swap_ratio, activation, device):
        super(MLPEmbedded, self).__init__(NN_architecture, init, prune_rate, activation, device)

        trained_model_architecture = get_NN_architecture_from_model(trained_model)
        assert len(trained_model_architecture) == len(NN_architecture)  # Networks must have the same number of layers
        assert trained_model_architecture[0] == NN_architecture[0]  # The number of input neurons must be equal
        assert trained_model_architecture[-1] == NN_architecture[-1]  # The number of output neurons must match
        # For embedding the parameter tensors, the randomly initialized network needs at least as many neurons
        # in the hidden layers as the trained network
        for i in range(1, len(NN_architecture) - 1):
            assert NN_architecture[i] >= trained_model_architecture[i]

        self.trained_model_state_dict = trained_model.state_dict()
        # Initialize zero tensors of network shape to mark the position of the embedded network
        self.embedding_mask = self._initialize_embedding_mask()

        # Embed trained model into randomly initialized network (places parameters in the left upper corner)
        state_dict = self.net.state_dict()
        for name, param in trained_model.named_parameters():
            if param.ndim == 2:
                # Embed weights
                for i in range(param.shape[0]):
                    for j in range(param.shape[1]):
                        state_dict[name[4:]][i][j] = param[i][j]  # 4:, because we don't include "net." in name
                        # Save embedding position
                        self.embedding_mask[name[4:]][i][j] = 1
            else:
                # Embed biases
                for i in range(param.shape[0]):
                    state_dict[name[4:]][i] = param[i]
                    # Save embedding position
                    self.embedding_mask[name[4:]][i] = 1
        self.net.load_state_dict(state_dict)
        self.net.to(device)

        if swap_ratio > 0.0:
            structured_parameters = {layer_num: {} for layer_num in [name[0] for name, _ in self.net.named_parameters()]}
            for name, param in self.net.named_parameters():
                layer_num = name[0]
                parameter_type = name[2:]
                structured_parameters[layer_num][parameter_type] = param

            for idx, layer_num in enumerate(list(structured_parameters.keys())[:-1]):
                next_layer_num = list(structured_parameters.keys())[idx+1]
                # Get random row numbers according to swap_ratio
                current_row_matrix = structured_parameters[layer_num]["weight"].data
                current_bias_matrix = structured_parameters[layer_num]["bias"].data
                current_column_matrix = structured_parameters[next_layer_num]["weight"].data
                number_of_rows = len(current_row_matrix)
                selected_rows = np.random.choice(range(number_of_rows), size=int(np.ceil(swap_ratio * number_of_rows)),
                                                 replace=False)

                rows_to_be_swapped = []
                for selected_row in selected_rows:
                    # Randomly choose a swap partner
                    while True:
                        swap_partner = np.random.choice(range(number_of_rows))
                        if selected_row != swap_partner:
                            rows_to_be_swapped.append((selected_row, swap_partner))
                            break

                # Perform swap on current row matrix, on bias nodes if existent and on the successive column matrix
                modified_row_matrix = self._swap_rows(current_row_matrix, rows_to_be_swapped)
                modified_bias = self._swap_columns(current_bias_matrix, rows_to_be_swapped) \
                    if current_bias_matrix is not None else None
                modified_column_matrix = self._swap_columns(current_column_matrix, rows_to_be_swapped)

                # Perform the swap also on the embedding indicator
                modified_embedding_row_matrix = self._swap_rows(self.embedding_mask[f"{layer_num}.weight"],
                                                                rows_to_be_swapped)
                modified_embedding_bias = self._swap_columns(self.embedding_mask[f"{layer_num}.bias"],
                                                             rows_to_be_swapped) \
                    if current_bias_matrix is not None else None
                modified_embedding_column_matrix = (
                    self._swap_columns(self.embedding_mask[f"{next_layer_num}.weight"], rows_to_be_swapped))

                # Update model with modified parameters
                state_dict = self.net.state_dict()
                state_dict[f"{layer_num}.weight"].data = modified_row_matrix
                state_dict[f"{layer_num}.bias"].data = modified_bias
                state_dict[f"{next_layer_num}.weight"].data = modified_column_matrix
                self.net.load_state_dict(state_dict)

                # Update embedding mask
                self.embedding_mask[f"{layer_num}.weight"] = modified_embedding_row_matrix
                self.embedding_mask[f"{layer_num}.bias"] = modified_embedding_bias
                self.embedding_mask[f"{next_layer_num}.weight"] = modified_embedding_column_matrix

        self.embedded_subnetwork_bitvector = self._get_embedded_subnetwork_bitvector()

    def show_embedded_subnetwork(self, save_path=None):
        subnet = DrawSubnetwork(self.NN_architecture, self.embedded_subnetwork_bitvector, save_path)
        subnet.draw()

    def _get_embedded_subnetwork_bitvector(self):
        bit_vector = []
        for name, mask in self.embedding_mask.items():
            if name[2:] == "weight":
                for j in range(mask.shape[1]):
                    for i in range(mask.shape[0]):
                        bit_vector.append(int(mask[i][j]))

        return bit_vector

    def _initialize_embedding_mask(self):
        embedding_mask = {}
        for name, param in self.net.named_parameters():
            if param.ndim == 2:
                embedding_mask[name] = np.zeros((param.shape[0], param.shape[1]))
            else:
                embedding_mask[name] = np.zeros(param.shape[0])

        return embedding_mask


    def _swap_rows(self, param, rows_to_be_swapped):
        for (first, second) in rows_to_be_swapped:
            temp = copy.deepcopy(param[first])
            param[first] = copy.deepcopy(param[second])
            param[second] = temp

        return param

    def _swap_columns(self, param, rows_to_be_swapped):
        if param.ndim == 2:
            for (first, second) in rows_to_be_swapped:
                temp = copy.deepcopy(param[:, first])
                param[:, first] = copy.deepcopy(param[:, second])
                param[:, second] = temp
        else:
            self._swap_rows(param, rows_to_be_swapped)

        return param
