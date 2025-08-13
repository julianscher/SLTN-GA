import math

import numpy as np
import scipy.stats as stats
import torch
from scipy.stats import t, ttest_ind, ttest_rel, entropy
from scipy.spatial.distance import jensenshannon
from torch import cosine_similarity
import torch.nn.functional as F

from data import MOONS
from models.fc.mlp import MLP
from utilities.helper_functions import get_results_path
from utilities.net_utils import get_all_parameters


def get_mean(values):
    return np.mean(values)

def get_population_standard_deviation(values):
    """ divides by population size N """
    return np.std(values)

def get_sample_standard_deviation(values):
    """ divides by population size N - 1"""
    return np.std(values, ddof=1)

def get_population_standard_error(values):
    return stats.sem(values, axis=None, ddof=0)

def get_sample_standard_error(values):
    return stats.sem(values, axis=None)

def get_sample_standard_error_estimate(values):
    # cf. https://www.statology.org/standard-error-of-mean-python/
    standard_deviation = get_sample_standard_deviation(values)
    return standard_deviation / np.sqrt(np.size(values))

def get_number_of_required_runs(values, precision):
    """ the precision defines the level of uncertainty of the true value you find acceptable """
    standard_error = get_sample_standard_error_estimate(values)
    print(math.pow((standard_error / precision), 2))

def get_mean_confidence_interval(values, confidence=0.95):
    """ cf. https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data """
    a = 1.0 * np.array(values)
    n = np.size(a)
    m, se = get_mean(a), get_sample_standard_error(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h

def get_mean_confidence_interval_short(values, confidence=0.95):
    """ loc is the mean, scale is the standard error of the mean """
    return t.interval(confidence, df=np.size(values) - 1, loc=get_mean(values), scale=get_sample_standard_error(values))

def perform_independent_ttest(first_values, second_values):
    """ first_values: baseline, second_values: competitor """
    t_stat, p_value = ttest_ind(first_values, second_values, equal_var=False) # Assuming unequal variances
    return t_stat, p_value

def perform_paired_ttest(first_values, second_values):
    t_stat_paired, p_value_paired = ttest_rel(first_values, second_values)
    return t_stat_paired, p_value_paired

def get_effect_size(first_values, second_values):
    # Effect Size: Cohen's d
    mean_diff = get_mean(first_values) - get_mean(second_values)
    pooled_std = np.sqrt(((get_sample_standard_deviation(first_values)**2) + (get_sample_standard_deviation(second_values)**2)) / 2)
    effect_size = mean_diff / pooled_std
    return effect_size


# Parameter-based Model Comparison

def calculate_parameters_distance_of_models(model1_parameters, model2_parameters):
    return sum(torch.norm(p1 - p2).item() for p1, p2 in zip(torch.tensor(model1_parameters), torch.tensor(model2_parameters)))


def rank_models_parameters_according_to_frobenius_norm(model1_parameters_with_position, model2_parameters_with_position, reverse):
    parameters_distances = [(p2, torch.norm(torch.tensor(p1[-1]) - torch.tensor(p2[-1])).item()) for p1, p2 in
                            zip(model1_parameters_with_position, model2_parameters_with_position)]
    return sorted(parameters_distances, key=lambda x: x[1], reverse=reverse)


def rank_models_parameters_according_to_cosine_similarity(model1_parameters_with_position, model2_parameters_with_position, reverse):
    parameters_distances = [(p2, cosine_similarity(torch.tensor(p1[-1]).view(-1), torch.tensor(p2[-1]).view(-1), dim=0).item()) for p1, p2 in
                            zip(model1_parameters_with_position, model2_parameters_with_position)]
    return sorted(parameters_distances, key=lambda x: x[1], reverse=reverse)


def rank_models_parameters_according_to_euclidean_distance(model1_parameters_with_position, model2_parameters_with_position, reverse):
    parameters_distances = [(p2, torch.sqrt(torch.sum((torch.tensor(p1[-1]) - torch.tensor(p1[-2])) ** 2)).item()) for p1, p2 in
                            zip(model1_parameters_with_position, model2_parameters_with_position)]
    return sorted(parameters_distances, key=lambda x: x[1], reverse=reverse)


def rank_models_parameters_according_to_manhattan_distance(model1_parameters_with_position, model2_parameters_with_position, reverse):
    parameters_distances = [(p2, torch.sum(torch.abs((torch.tensor(p1[-1]) - torch.tensor(p1[-2])))).item()) for p1, p2 in
                            zip(model1_parameters_with_position, model2_parameters_with_position)]
    return sorted(parameters_distances, key=lambda x: x[1], reverse=reverse)


def rank_models_parameters_according_to_chebyshev_distance(model1_parameters_with_position, model2_parameters_with_position, reverse):
    parameters_distances = [(p2, torch.max(torch.abs((torch.tensor(p1[-1]) - torch.tensor(p1[-2])))).item()) for p1, p2 in
                            zip(model1_parameters_with_position, model2_parameters_with_position)]
    return sorted(parameters_distances, key=lambda x: x[1], reverse=reverse)


def calculate_parameters_cosine_similarity_of_models(model1_parameters, model2_parameters):
    similarities = []
    for p1, p2 in zip(torch.tensor(model1_parameters), torch.tensor(model2_parameters)):
        similarities.append(cosine_similarity(p1.view(-1), p2.view(-1), dim=0).item())
    return sum(similarities) / len(similarities)


def calculate_kl_divergence_of_models(model1_parameter_distribution, model2_parameter_distribution):
    # Calculate KL divergence
    kld = entropy(model1_parameter_distribution, model2_parameter_distribution)
    return kld

def calculate_jensenshannon_distance_of_models(model1_parameter_distribution, model2_parameter_distribution):
    jsd = jensenshannon(model1_parameter_distribution, model2_parameter_distribution)
    return jsd


# Output-based Model Comparison
def calculate_activation_correlation_of_models(model1, model2, inputs):
    outputs1 = model1(inputs)
    outputs2 = model2(inputs)
    correlation = torch.corrcoef(torch.cat([outputs1.flatten(), outputs2.flatten()])).item()
    return correlation


def calculate_output_divergence_of_models(model1, model2, inputs):
    outputs1 = model1(inputs)
    outputs2 = model2(inputs)

    # Compute distance and similarity
    l2_distance = torch.norm(outputs1 - outputs2).item()
    mse = F.mse_loss(outputs1, outputs2).item()
    cosine_sim = F.cosine_similarity(outputs1.flatten(), outputs2.flatten(), dim=0).item()

    return l2_distance, mse, cosine_sim


def compare_representations(model1, model2, inputs):
    def hook_fn(module, input, output):
        activations.append(output.detach())

    activations = []
    hooks = []

    # Register hooks for model1 and model2 networks
    for module in model1.net.children():
        hooks.append(module.register_forward_hook(hook_fn))

    _ = model1(inputs)  # Run model1

    model1_activations = activations[:]
    activations.clear()  # Clear for next network


    for module in model2.net.children():
        hooks.append(module.register_forward_hook(hook_fn))

    _ = model2(inputs)  # Run model2
    model2_activations = activations[:]

    # Remove hooks
    for hook in hooks:
        hook.remove()

    padded_activations = []
    for model1_activation, model2_activation in zip(model1_activations, model2_activations):
        model1_activation_padded, model2_activation_padded = pad_ndarray(np.array(model1_activation), np.array(model2_activation))
        padded_activations.append((model1_activation_padded, model2_activation_padded))


    # Compare activations
    similarities = [F.cosine_similarity(torch.tensor(source).flatten(), torch.tensor(target).flatten(), dim=0).item()
                    for source, target in padded_activations]
    return similarities


def prepare_parameter_distribution(model):
    all_params = get_all_parameters(model)

    # Convert to numpy arrays (if not already)
    all_params = np.array(all_params)

    # Shift and normalize to ensure non-negativity
    all_params_shifted = all_params - np.min(all_params) + 1e-10

    # Normalize to make them valid probability distributions
    parameter_distribution = all_params_shifted / np.sum(all_params_shifted)

    return parameter_distribution


def pad_ndarray(ndarray_1, ndarray_2):
    ndarray_1 = ndarray_1.flatten()
    ndarray_2 = ndarray_2.flatten()
    size = max(len(ndarray_1), len(ndarray_2))
    # Pad with 0s
    ndarray_1 = np.pad(ndarray_1,
                       (0, size - len(ndarray_1)), mode='constant')
    ndarray_2 = np.pad(ndarray_2,
                       (0, size - len(ndarray_2)), mode='constant')
    return ndarray_1, ndarray_2

def count_missing_parameter_elements(model1, model2):
    # Get the state_dicts (parameter dictionaries) of the models
    model1_params = model1.state_dict()
    model2_params = model2.state_dict()

    missing_elements_count = 0

    # Check each parameter in model1
    for name, param1 in model1_params.items():
        if name not in model2_params:
            # If the parameter name is not in model2, count all elements in param1 as missing
            missing_elements_count += param1.numel()
        else:
            param2 = model2_params[name]

            # Handle size differences
            min_size = torch.min(torch.tensor(param1.size()), torch.tensor(param2.size()))
            overlapping_param1 = param1.view(-1)[:min_size.prod()]
            overlapping_param2 = param2.view(-1)[:min_size.prod()]

            # Count mismatched elements in the overlapping region
            mismatched_elements = (overlapping_param1 != overlapping_param2).sum().item()
            missing_elements_count += mismatched_elements

            # Count additional elements in param1 (extra elements in model1)
            missing_elements_count += param1.numel() - overlapping_param1.numel()

    return missing_elements_count

def measure_confidence1(model, inputs):
    """If the maximum probability is close to 0.5, this indicates uncertainty in the model’s prediction, meaning
    it is not very confident. If it's closer to 1 (e.g., 0.9 or higher), the model is more confident."""
    probs = F.softmax(model(inputs), dim=-1)  # Softmax probabilities
    return torch.max(probs) # Confidence (max probability)

def measure_confidence2(model, inputs):
    """Entropy measures the uncertainty in a probability distribution. Higher entropy indicates higher uncertainty
    (i.e., the classifier is less confident in its predictions). Lower entropy indicates lower uncertainty
    (i.e., the classifier is more confident).
    """
    probs = F.softmax(model(inputs), dim=-1)
    entropy = -np.sum(probs * np.log(probs + 1e-9))  # Add a small constant to avoid log(0)
    return entropy


if __name__ == '__main__':
    model1_state_dict = torch.load(
        f"{get_results_path()}/study_out/run2024-12-07 17:24:51/run0/model.pt")
    model1 = MLP([1, 100, 1], "default", 0.5, "sine", "cpu")
    model1.load_state_dict(model1_state_dict)

    model2_state_dict = torch.load(
        f"{get_results_path()}/study_out/run2024-12-11 17:21:42/run0/model.pt")
    model2 = MLP([1, 100, 1], "default", 0.5, "sine", "cpu")
    model2.load_state_dict(model2_state_dict)

    """model3_state_dict = torch.load(
        f"{get_results_path()}/study_out/run2024-12-01 11:47:34/run0/model.pt")
    model3 = MLP([1, 100, 1], "default", 0.5)
    model3.load_state_dict(model3_state_dict)"""

    model1_parameter_distribution = prepare_parameter_distribution(model2)
    model2_parameter_distribution = prepare_parameter_distribution(model1)
    #model1_parameter_distribution, model2_parameter_distribution = pad_ndarray(model1_parameter_distribution, model2_parameter_distribution)
    #model3_parameter_distribution = prepare_parameter_distribution(model3)
    model1_all_parameters = get_all_parameters(model1)
    model2_all_parameters = get_all_parameters(model2)
    #model3_all_parameters = get_all_parameters(model3)

    data = MOONS(norm=True, device="cpu")
    input = data.test_set.X[:1]

    print(calculate_parameters_distance_of_models(model1_all_parameters, model2_all_parameters))
    #print(calculate_parameters_distance_of_models(model1_all_parameters, model3_all_parameters))
    print(calculate_parameters_cosine_similarity_of_models(model1_all_parameters, model2_all_parameters))
    #print(calculate_parameters_cosine_similarity_of_models(model1_all_parameters, model3_all_parameters))
    print(calculate_kl_divergence_of_models(model1_parameter_distribution, model2_parameter_distribution))
    #print(calculate_kl_divergence_of_models(model1_parameter_distribution, model3_parameter_distribution))
    print(calculate_jensenshannon_distance_of_models(model1_parameter_distribution, model2_parameter_distribution))
    #print(calculate_jensenshannon_distance_of_models(model1_parameter_distribution, model3_parameter_distribution))

    print(calculate_activation_correlation_of_models(model1, model2, input))
    #print(calculate_activation_correlation_of_models(model1, model3, input))
    print(calculate_output_divergence_of_models(model1, model2, input))
    #print(calculate_output_divergence_of_models(model1, model3, input))
    print(compare_representations(model1, model2, input))
    #print(compare_representations(model1, model3, input))








