import numpy as np
import torch
from matplotlib import pyplot as plt

from models.fc.mlp import MLP
from sklearn.manifold import TSNE


def plot_parameter_distribution(model, log_path=None, model_name=None):
    all_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            all_params.append(param.detach().cpu().numpy().flatten())

    # Flatten all parameters into a single array
    all_params = np.concatenate(all_params)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_params, bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.ylim(0, 100)
    plt.title("Distribution of Model Parameters")
    plt.xlabel("Parameter Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    if log_path:
        plt.savefig(f"{log_path}/parameter_distribution_{model_name}.png")
    else:
        plt.show()

def plot_overlaying_parameter_distributions(model1, model2):
    all_params_model1 = []
    for name, param in model1.named_parameters():
        if param.requires_grad:
            all_params_model1.append(param.detach().cpu().numpy().flatten())

    # Flatten all parameters into a single array
    all_params_model1 = np.concatenate(all_params_model1)

    all_params_model2 = []
    for name, param in model2.named_parameters():
        if param.requires_grad:
            all_params_model2.append(param.detach().cpu().numpy().flatten())

    # Flatten all parameters into a single array
    all_params_model2 = np.concatenate(all_params_model2)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_params_model1, bins=100, alpha=0.7, color='blue', edgecolor='black', label='Model 1')
    plt.hist(all_params_model2, bins=100, alpha=0.7, color='red', edgecolor='black', label='Model 2')
    plt.ylim(0, 100)
    plt.title("Distribution of Model Parameters")
    plt.xlabel("Parameter Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend()
    plt.show()


def visualize_sequential_latent_space(model, inputs, layer_idx):
    activations = []

    # Hook to capture activations
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())

    # Register the hook on the desired layer
    layer = model.net[layer_idx]
    hook = layer.register_forward_hook(hook_fn)

    # Forward pass for all inputs
    with torch.no_grad():
        for input_tensor in inputs:
            model(input_tensor.unsqueeze(0))

    hook.remove()  # Clean up the hook

    # Flatten activations and reduce dimensionality with t-SNE
    activations = np.concatenate(activations, axis=0)
    tsne_results = TSNE(n_components=2).fit_transform(activations)

    # Plot t-SNE results
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
    plt.title(f"t-SNE of Layer {layer_idx} Representations")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()


def saliency_map_sequential(model, input_tensor):
    model.eval()
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor.requires_grad_()
    output = model(input_tensor)
    output.backward()  # Compute gradients

    saliency = input_tensor.grad.abs().squeeze().cpu()  # Get absolute gradient
    return saliency


if __name__ == '__main__':
    model1_state_dict = torch.load(
        "/Users/julianwork/Coding/PyCharmProjects/genetic-neuron-selection/study_out/ga_performance_results/make_blobs/make_ten_blobs/job_0/model_max_pruned.pt")
    model1 = MLP([2, 100, 10], "uniform", 0.5, "relu", "cpu")
    model1.load_state_dict(model1_state_dict)

    """model2_state_dict = torch.load(
        "/Users/julian/Coding/Projects/PyCharmProjects/genetic-neuron-selection/study_out/run2024-12-15 11:18:03/run0/model.pt")
    model2 = MLP([1, 100, 1], "default", 0.5, "relu", "cpu")
    model2.load_state_dict(model2_state_dict)"""

    plot_parameter_distribution(model1, "/Users/julianwork/Coding/PyCharmProjects/genetic-neuron-selection/study_out/ga_performance_results/make_blobs/make_ten_blobs/job_0", "model_max_pruned")
    #plot_overlaying_parameter_distributions(model1, model2)

    """state_dict = torch.load("/Users/julian/Coding/Projects/PyCharmProjects/genetic-neuron-selection/study_out/run2024-12-02 18:19:56/run0/model.pt")
    model = MLP([1, 50, 100, 50, 1], "default", 0.5)
    model.load_state_dict(state_dict)
    print(model)"""
    #plot_parameter_distribution(model)

    # Example usage latent-space
    """data = QUADRATIC(True, "cpu", instantiate="make_sinusoid_quadratic", noisy=False)
    inputs = data.test_set.X
    #visualize_sequential_latent_space(model, inputs, layer_idx=0)  # Visualize second Linear layer

    # Example usage
    input_tensor = inputs[0]
    #saliency = saliency_map_sequential(model, input_tensor)"""

    #print(f"Saliency: {saliency.item()}")  # Prints saliency value for the input feature

    """import torch

    # Define tensors
    tensor1 = torch.randn(100, 1)  # Shape: (100, 1)
    tensor2 = torch.randn(100)  # Shape: (100)

    # Perform addition
    result = tensor1 + tensor2

    # Check the shape of the result
    print(result.shape)  # Output: torch.Size([100, 100])"""