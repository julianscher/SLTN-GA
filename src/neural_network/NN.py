import numpy as np
import torch
from src.neural_network.helper_functions import path
from src.neural_network.base_networks import NN_with_custom_init
from src.neural_network.embedded_networks import embed_two_four_one, embed_two_two_six_two_one_lth, \
    embed_two_twenty_one_pruned
from src.neural_network.helper_functions import determine_large_nn_architecture, get_network_architecture
from src.neural_network.reinitialization_methods import reinitialize_parameters2


def get_model(NN_architecture, seed=None, param_reinitialization_method=reinitialize_parameters2, embedded=None, large_NN_param=None):
    if not embedded:
        if large_NN_param:
            C, epsilon, delta = large_NN_param[0], large_NN_param[1], large_NN_param[2]
            # Architecture of the large NN that should approximate the target network
            NN_architecture = determine_large_nn_architecture(NN_architecture, C, epsilon, delta)

        # We need a seed in all cases, to ensure that the large NN is the same for all subnetworks
        if not seed:
            seed = np.random.randint(0, 2**32 - 1, dtype=np.int64)

        # Create large NN
        return NN_with_custom_init(NN_architecture, seed, param_reinitialization_method)
    else:
        # Load NN with corresponding hidden model
        if embedded[0] == "fixed":
            return embedded[1].load_state_dict(
                torch.load(f"{path}/model_parameters_{str(NN_architecture)}_embedded_{str(embedded[2])}.pth"))
        else:
            if NN_architecture == [2, 12, 1]:
                model = embed_two_four_one(save=False)
            elif NN_architecture == [2, 23, 6, 62, 1]:
                model = embed_two_two_six_two_one_lth(save=False)
            else:
                model = embed_two_twenty_one_pruned(save=False)

            return model
