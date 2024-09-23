import torch
from src.neural_network.helper_functions import path
from src.neural_network.base_networks import NN


# Get model for Experiment1
def embed_two_four_one(save):
    # Construct model
    net = NN([2, 12, 1])

    # Load model parameters for trained model [2, 4, 1]
    subnet_param = torch.load(f"{path}/model_parameters_[2, 4, 1].pth")

    # Retrieve subnet parameters
    W1 = subnet_param['fc1.weight']
    b1 = subnet_param['fc1.bias']
    W2 = subnet_param['fc2.weight']
    b2 = subnet_param['fc2.bias']

    # embed subnetwork in model
    for i in range(4):
        net.layer1.weight.data[i + 4][0] = W1[i][0]
        net.layer1.weight.data[i + 4][1] = W1[i][1]
        net.layer1.bias.data[i + 4] = b1[i]
        net.layer2.weight.data[0][i + 4] = W2[0][i]
    net.layer2.bias.data[0] = b2[0]

    if save:
        torch.save(net.state_dict(), f"{path}/model_parameters_[2, 12, 1]_embedded_[2, 4, 1].pth")
    return net


def embed_two_twenty_one_pruned(save):
    # Construct model
    net = NN([2, 30, 1])

    # Load model parameters for trained model [2, 20, 1]
    subnet_param = torch.load(f"{path}/model_parameters_[2, 20, 1]_pruned.pth")

    # Retrieve subnet parameters
    W1 = subnet_param['layer1.weight']
    b1 = subnet_param['layer1.bias']
    W2 = subnet_param['layer2.weight']
    b2 = subnet_param['layer2.bias']

    # embed subnetwork in model
    for i in range(20):
        net.layer1.weight.data[i] = W1[i]
        net.layer1.bias.data[i] = b1[i]
        net.layer2.weight.data[0][i] = W2[0][i]
    net.layer2.bias.data = b2
    if save:
        torch.save(net.state_dict(), f"{path}/model_parameters_[2, 30, 1]_embedded_[2, 20, 1].pth")
    return net


def embed_two_two_six_two_one_lth(save):
    # Construct model
    net = NN([2, 23, 6, 62, 1])

    # Load model parameters for trained model [2, 4, 1]
    subnet_param = torch.load(f"{path}/model_parameters_[2, 2, 6, 2, 1]_lth_0.99975acc.pth")

    # Retrieve subnet parameters
    W1 = subnet_param['layer1.weight']
    b1 = subnet_param['layer1.bias']
    W2 = subnet_param['layer2.weight']
    b2 = subnet_param['layer2.bias']
    W3 = subnet_param['layer3.weight']
    b3 = subnet_param['layer3.bias']
    W4 = subnet_param['layer4.weight']
    b4 = subnet_param['layer4.bias']

    # embed subnetwork in model
    net.layer1.weight.data[10] = W1[0]
    net.layer1.weight.data[11] = W1[1]
    net.layer1.bias.data[10] = b1[0]
    net.layer1.bias.data[11] = b1[1]
    for i in range(len(net.layer2.weight.data)):
        net.layer2.weight.data[i][10] = W2[i][0]
        net.layer2.weight.data[i][11] = W2[i][1]
    net.layer2.bias.data = b2
    net.layer3.weight.data[30] = W3[0]
    net.layer3.weight.data[31] = W3[1]
    net.layer3.bias.data[30] = b3[0]
    net.layer3.bias.data[31] = b3[1]
    net.layer4.weight.data[0][30] = W4[0][0]
    net.layer4.weight.data[0][31] = W4[0][1]
    net.layer4.bias.data = b4

    if save:
        torch.save(net.state_dict(), f"{path}/model_parameters_[2, 23, 6, 62, 1]_embedded_[2, 2, 6, 2, 1].pth")
    return net