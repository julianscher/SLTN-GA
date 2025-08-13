import math

import torch
from torch import nn


class Builder(object):
    def __init__(self):
        pass

    def linear(self, in_features, out_features, bias, init, prune_rate):
        lin = nn.Linear(in_features, out_features, bias=bias)
        self._init_linear(lin, init, prune_rate)
        return lin

    def conv(self):
        pass

    def _init_linear(self, lin, init, prune_rate):
        if init == "default":
            pass

        elif init == "uniform":
            nn.init.uniform_(lin.weight, -2.5, 2.5)
            nn.init.uniform_(lin.bias, -2.5, 2.5)

        elif init == "normal":
            nn.init.normal_(lin.weight, std=1.5)
            nn.init.normal_(lin.bias, std=1.5)

        elif init == "zeros":
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

        elif init == "scaled_kaiming_normal":
            fan = nn.init._calculate_correct_fan(lin.weight, "fan_in")
            fan = fan * (1 - prune_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                lin.weight.data.normal_(0, std)

        elif init == "scaled_signed_constant":
            fan = nn.init._calculate_correct_fan(lin.weight, "fan_in")
            fan = fan * (1 - prune_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            lin.weight.data = lin.weight.data.sign() * std

    def activation_fn(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)



