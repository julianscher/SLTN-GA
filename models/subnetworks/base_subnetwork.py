import copy

import torch
from torch import count_nonzero


class BaseSubnetwork:
    def __init__(self, model, mask_except=None, device="cpu"):
        super(BaseSubnetwork, self).__init__()
        self.net = model.to(device)
        self.mask_except = mask_except
        self.device = device

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def __call__(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def __str__(self):
        return str(self.net)

    def save_model(self, model_path):
        torch.save(self.net.state_dict(), f"{model_path}/model.pt")

    def layers_names(self):
        return [name for name, _ in self.net.named_parameters()]

    def print_layers(self):
        for name, param in self.net.named_parameters():
            print(name, param.shape)

    def parameters(self):
        return {name: param for name, param in self.net.named_parameters()}

    def parameters_flattened(self):
        return {name: param.flatten() for name, param in self.net.named_parameters()}

    def parameters_dimensions(self):
        return {name: param.size() for name, param in self.net.named_parameters()}

    def total_number_of_parameters(self):
        params = []
        for name, parameter in self.net.named_parameters():
            params.append(parameter)
        return sum(p.numel() for p in params)

    def bitvector_ranges(self):
        bit_vector_ranges = {}
        pointer = 0
        for name, param in self.net.named_parameters():
            if self.mask_except and name in self.mask_except:
                continue
            bit_vector_ranges[name] = (pointer, pointer + param.numel() - 1)
            pointer += param.numel()
        return bit_vector_ranges

    def number_of_parameters_to_be_masked(self):
        return sum(param.numel() for name, param in self.net.named_parameters() if
                   not self.mask_except or name not in self.mask_except)

    def ratio_of_zero_parameters(self):
        zero_parameters = int(sum(param.numel() - count_nonzero(param) for name, param in self.net.named_parameters() if
                              not self.mask_except or name not in self.mask_except))
        return zero_parameters / self.number_of_parameters_to_be_masked()

    def apply_mask(self, bit_vector):
        """ Applies a binary mask to the model parameters. """
        masked_model = copy.deepcopy(self)  # Copy weights

        bit_vector_ranges = self.bitvector_ranges()
        parameters_dimensions = self.parameters_dimensions()

        for name, param in masked_model.net.named_parameters():
            if self.mask_except and name in self.mask_except:
                continue
            # Get corresponding segment in bit-vector
            start_idx, end_idx = bit_vector_ranges[name]
            segment = bit_vector[start_idx:end_idx + 1]

            # Reshape segment to necessary dimensions
            mask = segment.reshape(parameters_dimensions[name])  # For nxm matrices, creates new row after every m elements in bit-vector
            param.data *= torch.tensor(mask).to(self.device)  # Apply mask

        return masked_model