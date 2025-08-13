import copy

import numpy as np
import torch
from torch import nn


def get_number_of_parameters(model):
    params = []
    for name, parameter in model.named_parameters():
        if "score" not in name:
            params.append(parameter)
    return sum(p.numel() for p in params)


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def get_all_parameters(model):
    all_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            all_params.append(param.detach().cpu().numpy().flatten())

    # Flatten all parameters into a single array
    all_params = np.concatenate(all_params)
    return all_params


def get_all_parameters_with_position(model):
    parameter_tensors = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            parameter_tensors.append((name, param.detach().cpu().numpy()))

    all_params = []
    for ele in parameter_tensors:
        if len(ele[1].shape) == 1:
            for i in range(ele[1].shape[0]):
                all_params.append((ele[0], (i), ele[1][i]))
        elif len(ele[1].shape) == 2:
            for i in range(ele[1].shape[0]):
                for j in range(ele[1].shape[1]):
                    all_params.append((ele[0], (i, j), ele[1][i][j]))

    return all_params


def replace_parameter_at_position(model, parameter_name, position, value):
    if isinstance(position, int):
        model.state_dict()[parameter_name][position] = copy.deepcopy(value)
    elif isinstance(position, tuple):
        model.state_dict()[parameter_name][position[0]][position[1]] = copy.deepcopy(value)


def adapt_dtype(criterion, y_batch):
    if isinstance(criterion, (nn.CrossEntropyLoss, nn.NLLLoss, nn.BCEWithLogitsLoss)):
        return y_batch.long()
    elif isinstance(criterion, (nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss)):
        return y_batch.float()
    else:
        raise ValueError(f"Unsupported criterion type: {type(criterion)}")


def accuracy(output, target, device):
    preds = predict(output)

    preds = preds.flatten()
    target = target.flatten()

    # Calculate accuracy
    correct = preds.eq(target)
    accuracy = (correct.sum() / len(target)) * 100
    return accuracy.item()


def cross_entropy_loss(output, target, device):
    return nn.CrossEntropyLoss().to(device)(output, target.long())


def mean_squared_error(output, target, device):
    return nn.MSELoss().to(device)(output, target.float())


def predict(output, stack=False):
    with torch.no_grad():
        predictions = np.array([])
        if output.shape[1] > 1:
            # Multi-class classification
            _, preds = output.topk(1, 1, True, True) # Returns top-1 prediction for each input
        else:
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).type(torch.long)
        if stack:
            # Horizontally stacks (concatenates) the predictions array and the flattened preds array. The result is a 1D array.
            return np.hstack((predictions, preds.numpy().flatten())).flatten()
        return preds.clone().detach().requires_grad_(False) # Return tensor object