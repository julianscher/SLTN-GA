import copy
import os

import dill
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from data import MOONS, BLOBS, DIGITS
from genetic_algorithm.methods.generation.fc import NegligentGeneration
from models.fc.mlp import MLP
from models.subnetworks.subnetworks import SubnetMLP
from utilities.helper_functions import get_results_path
from utilities.net_utils import predict, cross_entropy_loss, accuracy
import torch.nn.functional as F
import torch
from torch import nn
from torch.func import functional_call, vmap, hessian
import seaborn as sns


def apply_parameter_vector(model, param_vector, device, use_mask_except):
    modified_model = copy.deepcopy(model)  # Copy weights

    param_vector_ranges = model.bitvector_ranges()
    parameters_dimensions = model.parameters_dimensions()

    for name, param in modified_model.net.named_parameters():
        if use_mask_except and model.mask_except and name in model.mask_except:
            continue

        start_idx, end_idx = param_vector_ranges[name]
        segment = param_vector[start_idx:end_idx + 1]


        section = torch.tensor(segment).reshape(
            parameters_dimensions[name])
        param.data.copy_(section.to(device))

    return modified_model

def plot_loss_continuous_landscape1d(data_loader, model1, model2, alpha_min, alpha_max, alpha_num, criterion, device, seed, use_mask_except=False, log_path=None):
    X = data_loader.dataset.tensors[0]
    y = data_loader.dataset.tensors[1]
    torch.random.manual_seed(seed)

    theta_s = model1.parameters_flattened()
    theta_l = model2.parameters_flattened()

    def f(theta_s, theta_l, alpha):
        param_vector = []
        for name, param in theta_l.items():
            if use_mask_except and model1.mask_except and name in model1.mask_except:
                continue
            modified_params = theta_s[name] + alpha * (param - theta_s[name])
            param_vector.extend(list(modified_params.detach().numpy()))
        return np.array(param_vector)

    alphas = np.linspace(alpha_min, alpha_max, alpha_num)

    losses = []
    accuracies = []
    for alpha in alphas:
        modified_model = apply_parameter_vector(model1, f(theta_s, theta_l, alpha), device, use_mask_except)
        with torch.no_grad():
            loss = criterion(modified_model(X), y, device)
            losses.append(loss.item())
            acc = accuracy(modified_model(X), y, device)
            accuracies.append(acc)

    print(accuracies)
    print(losses)

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(alphas, losses, color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax1.axvline(x=0.0, color='black', linestyle='dashed', alpha=0.7)  # Vertical line for theta_s
    ax1.axvline(x=1.0, color='black', linestyle='dashed', alpha=0.7)  # Vertical line for theta_l

    ax1.text(0.0, ax1.get_ylim()[0] - 0.07 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
             r'$\theta_s$', ha='center', va='top', fontsize=12, color='black')
    ax1.text(1.0, ax1.get_ylim()[0] - 0.07 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
             r'$\theta_l$', ha='center', va='top', fontsize=12, color='black')

    ax2 = ax1.twinx()  # Second y-axis for accuracy
    color = 'tab:red'
    ax2.set_ylabel('accuracy', color=color)
    ax2.plot(alphas, accuracies, color=color, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()

    if log_path is not None:
        plt.savefig(os.path.join(log_path, 'loss_landscape1d.pdf'))
    else:
        plt.show()
    plt.clf()


def plot_metric_continuous_landscape2d(maximizing, data_loader, model, scale, clabel, delta_min, delta_max, eta_min, eta_max, delta_num, eta_num, levels, metric, device, log_path, seed, use_mask_except=False):
    X = data_loader.dataset.tensors[0]
    y = data_loader.dataset.tensors[1]
    torch.random.manual_seed(seed)

    # Create two random direction vectors
    parameters_flattened = model.parameters_flattened()
    d1 = {}
    d2 = {}
    for name, param in parameters_flattened.items():
        d1[name] = torch.rand(size=param.shape) * scale
        d2[name] = torch.rand(size=param.shape) * scale

    def f(theta_s, delta, d1, eta, d2):
        param_vector = []
        for name, param in d1.items():
            if use_mask_except and model.mask_except and name in model.mask_except:
                continue
            modified_params = theta_s[name] + delta * param + eta * d2[name]
            param_vector.extend(list(modified_params.detach().numpy()))
        return np.array(param_vector)

    deltas, etas = np.meshgrid(np.linspace(delta_min, delta_max, delta_num), np.linspace(eta_min, eta_max, eta_num))

    coefficients = np.c_[deltas.ravel(), etas.ravel()]

    metric_values = []
    iterator = tqdm(coefficients, desc="coefficients")
    for delta, eta in iterator:
        modified_model = apply_parameter_vector(model, f(parameters_flattened, delta, d1, eta, d2), device, use_mask_except)
        with torch.no_grad():
            value = metric(modified_model(X), y, device)
            metric_values.append(value.item() if isinstance(value, torch.Tensor) else value)

    print(metric_values)

    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 5.0
    fig, ax = plt.subplots(figsize=(18, 13))  # Set size here
    CS = ax.contour(deltas, etas, np.array(metric_values).reshape(deltas.shape), levels=levels, cmap='viridis' if maximizing else 'viridis_r', linewidths=4)
    if clabel:
        ax.clabel(CS, fontsize=15)
    plt.xlabel(r'$\delta$', fontsize=50)
    plt.ylabel(r'$\eta$', fontsize=50)
    ax.tick_params(axis='both', labelsize=40)
    ax.tick_params(width=5, length=10)
    plt.savefig(os.path.join(log_path, f'{metric}_landscape_scale{scale}.pdf')) # , dpi=300
    plt.show()

def plot_metric_discrete_landscape2d(plot_type, maximizing, data_loader, base_model, mask0, delta_min, delta_max, eta_min, eta_max, delta_num, eta_num, levels, metric, device, log_path, seed):
    X = data_loader.dataset.tensors[0]
    y = data_loader.dataset.tensors[1]
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Randomly sample two bit-masks
    negligent_generation = NegligentGeneration(base_model, 0.5)
    mask1 = negligent_generation.generate()
    mask2 = negligent_generation.generate()

    def AND(a, b):
        return min(a, b)

    def OR(a, b):
        return (a+b) - (AND(a, b))

    def f1(mask0, delta, mask1, eta, mask2):
        delta_is_negative = np.sign(delta) == -1
        eta_is_negative = np.sign(eta) == -1
        delta = np.abs(delta)
        eta = np.abs(eta)

        mask_s = []
        for idx in range(len(mask0)):
            bit = mask0[idx]
            bit0 = mask0[idx]
            bit1 = mask1[idx]
            bit2 = mask2[idx]
            if idx <= delta:
                if not delta_is_negative:
                    bit = OR(bit0, bit1)
                else:
                    bit = AND(bit0, bit1)

            if idx <= eta:
                if not eta_is_negative:
                    bit = OR(bit, bit2)
                else:
                    bit = AND(bit, bit2)

            mask_s.append(bit)

        return np.array(mask_s)

    def f2(mask0, delta, mask1, eta, mask2):
        delta_is_negative = np.sign(delta) == -1
        eta_is_negative = np.sign(eta) == -1
        delta = np.abs(delta)
        eta = np.abs(eta)

        def bit_flip(bit):
            return np.abs(bit - 1)

        mask_s = []
        for idx in range(len(mask0)):
            bit = mask0[idx]
            bit0 = mask0[idx]
            bit1 = mask1[idx]
            bit2 = mask2[idx]
            if idx <= delta:
                if not delta_is_negative:
                    bit = OR(bit0, bit1)
                else:
                    bit = OR(bit0, bit_flip(bit1))

            if idx <= eta:
                if not eta_is_negative:
                    bit = OR(bit, bit2)
                else:
                    bit = OR(bit0, bit_flip(bit2))

            mask_s.append(bit)

        return np.array(mask_s)

    deltas, etas = np.meshgrid(np.linspace(delta_min, delta_max, delta_num, dtype=int), np.linspace(eta_min, eta_max,  eta_num, dtype=int))


    coefficients = np.c_[deltas.ravel(), etas.ravel()]
    """masks = []
    for delta, eta in coefficients:
        print(delta, eta, f(mask0, delta, mask1, eta, mask2))
        masks.append(f(mask0, delta, mask1, eta, mask2))
    
    print(np.unique(masks, axis=0))"""

    metric_values = []
    iterator = tqdm(coefficients, desc="coefficients")
    for delta, eta in iterator:
        mask_s = f2(mask0, delta, mask1, eta, mask2)
        modified_model = model.apply_mask(mask_s)
        with torch.no_grad():
            value = metric(modified_model(X), y, device)
            metric_values.append(value.item() if isinstance(value, torch.Tensor) else value)

    deltas_standardized = 2 * (deltas - delta_min) / (delta_max - delta_min) - 1
    etas_standardized = 2 * (etas - eta_min) / (eta_max - eta_min) - 1

    if plot_type == "contour":
        plt.figure(figsize=(30, 25))
        fig, ax = plt.subplots()
        CS = ax.contour(deltas_standardized, etas_standardized, np.array(metric_values).reshape(deltas.shape), levels=levels, cmap='viridis' if maximizing else 'viridis_r')
        ax.clabel(CS, fontsize=10)
        plt.savefig(os.path.join(log_path, f'{metric}_landscape.pdf'), dpi=300)
        plt.show()
    elif plot_type == "heatmap":
        metric_values_grid = np.array(metric_values).reshape(deltas.shape)

        plt.figure(figsize=(30, 25))
        plt.imshow(metric_values_grid, cmap='viridis' if maximizing else 'viridis_r',  interpolation='nearest', aspect='auto', origin='lower',
                   extent=[deltas_standardized.min(), deltas_standardized.max(), etas_standardized.min(), etas_standardized.max()])
        plt.colorbar(label='Metric Value')
        plt.xlabel('Delta')
        plt.ylabel('Eta')

        plt.title(f'Metric Landscape Heatmap')

        plt.savefig(os.path.join(log_path, f'{metric}_landscape_heatmap.pdf'), dpi=300)

        plt.show()

def hessian_analysis(model, loss_fn, data_loader, device, log_path, model_name):
    """
    The Hessian of the loss function represents the second-order derivatives and can provide insight into the curvature of the loss landscape.

    Eigenvalues of the Hessian: The Hessian's eigenvalues describe the curvature of the loss function in different directions.
    A "harder" loss landscape typically has a larger proportion of eigenvalues that are large (indicating sharp curvatures)
    or negative (indicating saddle points).

    Condition Number of Hessian: A high condition number of the Hessian (ratio of the largest to the smallest eigenvalue)
    suggests that the optimization problem is more ill-conditioned and harder to solve.
    cf. https://discuss.pytorch.org/t/efficient-computation-of-hessian-with-respect-to-network-weights-using-autograd-grad-and-symmetry-of-hessian-matrix/156222/8
    """
    def compute_hessian(model, loss_fn, data_loader, device):
        """Simple Hessian computation using finite differences"""
        X = data_loader.dataset.tensors[0]
        y = data_loader.dataset.tensors[1]

        #X, y = next(iter(data_loader))

        # Compute second derivatives (Hessian approximation) of loss with respect to model parameters

        params = dict(model.net.named_parameters())

        def fcall(params, inputs):
            return functional_call(model.net, params, inputs)

        def compute_loss(params, inputs, targets):
            outputs = vmap(fcall, in_dims=(None, 0))(params, inputs)
            return loss_fn(outputs, targets, device)

        """
        hess = {
    'net.0.weight': {
        'net.0.weight': Hessian matrix for 'net.0.weight' w.r.t 'net.0.weight',
        'net.0.bias': Hessian matrix for 'net.0.weight' w.r.t 'net.0.bias',
        'net.2.weight': Hessian matrix for 'net.0.weight' w.r.t 'net.2.weight',
        'net.2.bias': Hessian matrix for 'net.0.weight' w.r.t 'net.2.bias',
    },
    'net.0.bias': {
        'net.0.weight': Hessian matrix for 'net.0.bias' w.r.t 'net.0.weight',
        'net.0.bias': Hessian matrix for 'net.0.bias' w.r.t 'net.0.bias',
        'net.2.weight': Hessian matrix for 'net.0.bias' w.r.t 'net.2.weight',
        'net.2.bias': Hessian matrix for 'net.0.bias' w.r.t 'net.2.bias',
    },
    'net.2.weight': {
        'net.0.weight': Hessian matrix for 'net.2.weight' w.r.t 'net.0.weight',
        'net.0.bias': Hessian matrix for 'net.2.weight' w.r.t 'net.0.bias',
        'net.2.weight': Hessian matrix for 'net.2.weight' w.r.t 'net.2.weight',
        'net.2.bias': Hessian matrix for 'net.2.weight' w.r.t 'net.2.bias',
    },
    'net.2.bias': {
        'net.0.weight': Hessian matrix for 'net.2.bias' w.r.t 'net.0.weight',
        'net.0.bias': Hessian matrix for 'net.2.bias' w.r.t 'net.0.bias',
        'net.2.weight': Hessian matrix for 'net.2.bias' w.r.t 'net.2.weight',
        'net.2.bias': Hessian matrix for 'net.2.bias' w.r.t 'net.2.bias',
    },
}"""
        # Each Hessian element (hess[param_i][param_j]) corresponds to the second-order derivative (Hessian) of the loss function with respect to the parameters param_i and param_j.
        hess = hessian(compute_loss, argnums=0)(params, X, y) # dictionary of format {params1: {params1: values1, ... paramsn: valuesn}, ..., paramsn: {params1: values1, ... paramsn: valuesn}}. corresponds to NxN matrix
        """print(hess.keys())
        key = list(params.keys())[0]  # take weight in first layer as example key
        # Only the elements on the diagonal are interesting for analysis
        print(hess[key][key].shape)  # Hessian of loss w.r.t first weight (shape [75, 64, 75, 64])"""
        return hess

    hess = compute_hessian(model, loss_fn, data_loader, device)

    def extract_diagonal(hessian_matrix):
        diagonal = {}
        for key in hessian_matrix.keys():
            diagonal[key] = {key: hessian_matrix[key][key]}
        return diagonal

    diagonal = extract_diagonal(hess)

    def analyze_hessian_blocks(hessian_dict):
        eigenvalues_dict = {}

        for param_name, sub_dict in hessian_dict.items():
            if param_name in sub_dict:  # Ensure it's a valid Hessian block
                hessian_matrix = sub_dict[param_name]  # Extract Hessian block for the parameter
                if hessian_matrix.ndim == 4:
                    N = hessian_matrix.shape[0]*hessian_matrix.shape[1]
                    hessian_matrix = hessian_matrix.reshape(N, N)

                hessian_matrix = hessian_matrix.detach().cpu().numpy()
                hessian_matrix = torch.from_numpy(hessian_matrix).float()

                # For validation purposes
                """if param_name ==  "net.2.weight":
                    print(hessian_matrix[-1][-1].item())"""

                try:
                    # Compute eigenvalues
                    #eigenvalues = np.linalg.eigvals(hessian_matrix)  # Use symmetric eigendecomposition
                    eigenvalues = torch.linalg.eigvalsh(hessian_matrix)
                    eigenvalues_dict[param_name] = eigenvalues

                    # Analyze eigenvalues
                    max_eigenvalue = eigenvalues.max().item()
                    min_eigenvalue = eigenvalues.min().item()
                    condition_number = abs(max_eigenvalue / max(abs(min_eigenvalue), 1e-12))  # Avoid division by zero
                    spectral_radius = abs(max_eigenvalue)

                    """print(f"Analysis for {param_name}:")
                    print(f"  Max Eigenvalue: {max_eigenvalue}")
                    print(f"  Min Eigenvalue: {min_eigenvalue}")
                    print(f"  Condition Number: {condition_number}")
                    print(f"  Spectral Radius: {spectral_radius}\n")"""

                    with open(f"{log_path}/hessian_{model_name}.txt", "a") as f:
                        print(f"  Max Eigenvalue: {max_eigenvalue}", file=f)
                        print(f"  Min Eigenvalue: {min_eigenvalue}", file=f)
                        print(f"  Condition Number: {condition_number}", file=f)
                        print(f"  Spectral Radius: {spectral_radius}\n", file=f)

                except RuntimeError as e:
                    print(f"Skipping {param_name} due to error: {e}")

        return eigenvalues_dict

    eigenvalues = analyze_hessian_blocks(diagonal)

    with open(f"{log_path}/hessian_{model_name}.txt", "a") as f:
        f.write(f"Eigenvalues\n: {eigenvalues}")

    return eigenvalues


def compute_hessian_entry(model, loss_fn,  data_loader, param_name, i, j, device):
    """
    Computes a single Hessian entry ∂²Loss / ∂(param[i]) ∂(param[j]) for validation.

    Args:
        model: The neural network model.
        loss_fn: The loss function.
        X: Input data.
        y: Target labels.
        param_name: The name of the parameter to differentiate.
        i, j: Indices of the parameter tensor for Hessian entry.

    Returns:
        The Hessian entry value as a scalar.
    """
    X = data_loader.dataset.tensors[0]
    y = data_loader.dataset.tensors[1]

    param = dict(model.net.named_parameters())[param_name]

    param.requires_grad = True

    # Compute first derivative (Gradient)
    loss = loss_fn(model(X), y, device)
    grad1 = torch.autograd.grad(loss, param, create_graph=True)[0]  # Gradient w.r.t param

    # Extract the specific element of gradient to differentiate
    grad1_ij = grad1.view(-1)[i]  # Convert to 1D and pick index i

    # Compute second derivative (Hessian entry)
    hessian_entry = torch.autograd.grad(grad1_ij, param, retain_graph=True)[0]

    return hessian_entry.view(-1)[j].item()  # Convert to 1D and pick index j


def plot_hessian_eigenvalues(eigenvalues_dict, log_path):
    """
    Plots the eigenvalue distribution of Hessian matrices for different model parameters.

    Args:
        eigenvalues_dict: Dictionary of eigenvalues per parameter block.
    """
    plt.figure(figsize=(12, 6))

    for param_name, eigenvalues in eigenvalues_dict.items():
        eigenvalues = eigenvalues.cpu().numpy()  # Convert to NumPy if necessary

        sns.histplot(eigenvalues, bins=50, kde=True, label=param_name, alpha=0.6)

    plt.axvline(x=0, color='k', linestyle='--', label="Zero Eigenvalue")  # Mark zero for reference
    plt.xlabel("Eigenvalues")
    plt.ylabel("Density")
    plt.title("Hessian Eigenvalue Distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{log_path}/hessian_eigenvalues_distribution.png")
    plt.show()
