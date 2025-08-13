from models.fc.mlp import MLP
from models.fc.mlp_embedded import MLPEmbedded
from models.subnetworks.base_subnetwork import BaseSubnetwork


class SubnetMLP(BaseSubnetwork):
    def __init__(self, nn_architecture, init, prune_rate, activation, mask_except=None, device="cpu"):
        super().__init__(MLP(nn_architecture, init, prune_rate, activation, device), mask_except, device)


class SubnetMLPEmbedded(BaseSubnetwork):
    def __init__(self, nn_architecture, init, prune_rate, trained_model, swap_ratio, activation, mask_except=None, device="cpu"):
        super().__init__(MLPEmbedded(nn_architecture, init, prune_rate, trained_model, swap_ratio, activation, device), mask_except, device)

