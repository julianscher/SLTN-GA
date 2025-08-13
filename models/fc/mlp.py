from torch import nn
from utilities.builder import Builder


class MLP(nn.Module):
    """ Create network with ReLU activation functions """
    def __init__(self, NN_architecture, init, prune_rate, activation, device):
        super(MLP, self).__init__()
        self.NN_architecture = NN_architecture
        builder = Builder()
        self.net = nn.Sequential()

        for l in range(len(NN_architecture) - 1):
            self.net.append(builder.linear(NN_architecture[l], NN_architecture[l + 1], bias=True, init=init,
                                           prune_rate=prune_rate))
            if l < len(NN_architecture) - 2:
                self.net.append(builder.activation_fn(activation))

        self.net.to(device)

    def forward(self, x):
        out = self.net(x)
        return out
