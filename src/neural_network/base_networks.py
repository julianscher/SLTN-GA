import torch
from torch import nn
import torch.nn.functional as F
from src.neural_network.helper_functions import path

class NN(nn.Module):
    """ Create network with ReLU activation functions """

    def __init__(self, NN_architecture):
        super(NN, self).__init__()
        self.NN_architecture = NN_architecture

        for l in range(len(NN_architecture) - 1):
            # Applies a linear transformation to the incoming data: y=xA^T+b
            self._modules[f"layer{l + 1}"] = nn.Linear(NN_architecture[l], NN_architecture[l + 1])

    def forward(self, x):
        for l in range(len(self.NN_architecture) - 2):
            x = F.relu(self._modules[f"layer{l + 1}"](x))
        return self._modules[f"layer{len(self.NN_architecture) - 1}"](x)


class NN_with_custom_init(nn.Module):
    """ Large NN in which to find the subnet that has good accuracy compared to the trained target network """

    def __init__(self, NN_architecture, seed, param_reinitialization_method):
        super(NN_with_custom_init, self).__init__()
        self.NN_architecture = NN_architecture
        self.seed = seed
        self.gainProd = 0.06
        torch.manual_seed(self.seed)
        for l in range(len(NN_architecture) - 1):
            self._modules[f"layer{l + 1}"] = nn.Linear(NN_architecture[l], NN_architecture[l + 1])
        param_reinitialization_method(self._modules)

    def save(self):
        # save model parameters
        torch.save(self.state_dict(), f"{path}/model_parameters_{self.NN_architecture}_lth_seed_{self.seed}.pth")

    def forward(self, x):
        for l in range(len(self.NN_architecture) - 2):
            x = F.relu(self._modules[f"layer{l + 1}"](x))
        return self._modules[f"layer{len(self.NN_architecture) - 1}"](x)