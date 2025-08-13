import math
import torch
from torch import nn

# ========== Different methods for parameter initialization for nn.Linear =========#
@torch.no_grad()
def reinitialize_parameters1(modules):
    for mod in modules:
        m = modules.get(mod)
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -1.0, 1.0)
            nn.init.zeros_(m.bias)


@torch.no_grad()
def reinitialize_parameters2(modules):
    for mod in modules:
        m = modules.get(mod)
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -1.0, 1.0)
            nn.init.uniform_(m.bias, -1.0, 1.0)


@torch.no_grad()
def reinitialize_parameters3(modules):
    # cf. https://github.com/RelationalML/UniversalLT/blob/eab7461f30b01211808dc1216f2807ff4052439c/edge-popup/main.py#L305, reason in paper
    gainProd = 0.06
    for mod in modules:
        m = modules.get(mod)
        if isinstance(m, nn.Linear):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = math.sqrt(2 / fan_in)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            gainProd = gainProd * std * 0.5
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=gainProd)


@torch.no_grad()
def reinitialize_parameters4(modules):
    # cf. https://github.com/gcastex/PruNet/blob/4da07e4901c26374082c2d7025b8994ffddd7d2b/prunet_models.py#L14 def init_weights, no reason
    for mod in modules:
        m = modules.get(mod)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


@torch.no_grad()
def reinitialize_parameters5(modules):
    # cf. https://github.com/acnagle/optimal-lottery-tickets/blob/f09c0d2d21b25d15b58253be8f455f733ecc5fad/pruning/utils/red_fc.py#L9_
    # -> look at https://github.com/acnagle/optimal-lottery-tickets/blob/f09c0d2d21b25d15b58253be8f455f733ecc5fad/subset-sum/main.py#L165 for potential other method
    for mod in modules:
        m = modules.get(mod)
        if isinstance(m, nn.Linear):
            std1 = math.sqrt(2. / 1)
            std2 = math.sqrt(2. / m.weight.data.size()[1])
            nn.init.normal_(m.weight, std=std1)  # it's not really clear if std=std1 or std2
            nn.init.uniform_(m.bias, a=-std2, b=std2)


@torch.no_grad()
def reinitialize_parameters6(modules):
    # cf. https://github.com/luuyin/Lottery-pools/blob/78fc43b0e26c5333ae1bd8f068a5c14b7dd0097f/ImageNet/models/initializers.py#L9 def binary, no reason
    for mod in modules:
        m = modules.get(mod)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            sigma = m.weight.data.std()
            m.weight.data = torch.sign(m.weight.data) * sigma

@torch.no_grad()
def reinitialize_parameters7(modules):
    # cf. https://github.com/gcastex/PruNet/blob/4da07e4901c26374082c2d7025b8994ffddd7d2b/prunet_models.py#L14 def init_weights, no reason
    for mod in modules:
        m = modules.get(mod)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if len(m.bias.data.shape) == 1:
                nn.init.uniform_(m.bias, -1.0, 1.0)
            else:
                nn.init.xavier_uniform_(m.bias)

@torch.no_grad()
def reinitialize_parameters8(modules):
    """ kaiming normal initialization from Ramanujan et al. with scale_fan=True, Used their default values """
    # cf. https://github.com/allenai/hidden-networks/blob/master/utils/builder.py#L110
    for mod in modules:
        m = modules.get(mod)
        if isinstance(m, nn.Linear):
            fan = nn.init._calculate_correct_fan(m.weight, "fan_in")
            fan = fan * (1 - 0.5)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            m.weight.data.normal_(0, std)

@torch.no_grad()
def reinitialize_parameters9(modules):
    """ signed constant initialization from Ramanujan et al. with scale_fan=True, Used their default values """
    # cf. https://github.com/allenai/hidden-networks/blob/master/utils/builder.py#L191
    for mod in modules:
        m = modules.get(mod)
        if isinstance(m, nn.Linear):
            fan = nn.init._calculate_correct_fan(m.weight, "fan_in")
            fan = fan * (1 - 0.5)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            m.weight.data = m.weight.data.sign() * std

@torch.no_grad()
def reinitialize_parameters10(modules):
    # cf. https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch/blob/34a8c9678406a1c7dd0fec4c9f0d25d017be55fb/main.py#L325 def weight_init
    # doesn't consistently find subnetworks with accuracy >= 0.7, even for large architectures like [2, 23, 6, 62, 1]
    for mod in modules:
        m = modules.get(mod)
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.normal_(m.bias.data)