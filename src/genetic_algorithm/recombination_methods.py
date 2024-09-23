import random
import numpy as np
from src.genetic_algorithm.bit_vector_operations import get_input_output_paths, get_bit_position, \
    check_for_input_output_path, get_neuron_connections, activate_connections, remove_floating_connections, \
    apply_connectivity_strictness_level
from src.genetic_algorithm.graph_operations import get_list_of_all_vertices
from src.genetic_algorithm.helper_functions import calculate_dimensionality


# ============ Recombination methods ============#

def crossover_recombination_path_fixed(genome1, genome2, recombine_args):
    NN_architecture = recombine_args.get("environment_args").get("NN_architecture")
    connectivity_scheme = recombine_args.get("recombine_args").get("connectivity_scheme", "lax")

    # Choose random parent to inherit input-output-path from
    basis = [genome1, genome2][np.random.choice([0, 1])].copy()

    # Get positions of bits of one input-output-path
    paths = get_input_output_paths(NN_architecture, basis)
    random_idx = np.random.choice(len(paths))
    path = paths[random_idx]
    pos = []
    for (layer, conn) in path:
        pos.append(get_bit_position(NN_architecture, layer, conn))

    child = [np.random.choice([genome1[idx], genome2[idx]]) if idx not in pos
             else basis[idx] for idx in range(len(basis))]

    return apply_connectivity_strictness_level(child, NN_architecture, connectivity_scheme)


def crossover_recombination(genome1, genome2, recombine_args):
    NN_architecture = recombine_args.get("environment_args").get("NN_architecture")
    connectivity_scheme = recombine_args.get("recombine_args").get("connectivity_scheme", "lax")

    while True:
        # Generate random child
        child = [np.random.choice([genome1[i], genome2[i]]) for i in range(len(genome1))]

        # Test if child contains input-output-path
        if check_for_input_output_path(NN_architecture, child):
            return apply_connectivity_strictness_level(child, NN_architecture, connectivity_scheme)


def crossover_recombination_negligent(genome1, genome2, recombine_args):
    child = [np.random.choice([genome1[i], genome2[i]]) for i in range(len(genome1))]
    return child


def neuron_recombination(genome1, genome2, recombine_args):
    NN_architecture = recombine_args.get("environment_args").get("NN_architecture")
    connectivity_scheme = recombine_args.get("recombine_args").get("connectivity_scheme", "lax")
    neurons = get_list_of_all_vertices(NN_architecture)
    valid_neurons = [neuron for neuron in neurons if neuron[0] % 2 != 0]
    child = [0 for _ in range(calculate_dimensionality(NN_architecture))]

    while True:
        # Pick randomly valid neuron from one of the parents
        for neuron in valid_neurons:
            neuron_giver = random.choice((genome1, genome2))

            # Get all activated incoming and outgoing connections of neuron
            connections = get_neuron_connections(NN_architecture, neuron_giver, neuron)

            # Activate these connections in child genome
            child = activate_connections(child, NN_architecture, connections)

        if check_for_input_output_path(NN_architecture, child):
            return apply_connectivity_strictness_level(child, NN_architecture, connectivity_scheme)


def input_output_path_recombination(genome1, genome2, recombine_args):
    NN_architecture = recombine_args.get("environment_args").get("NN_architecture")
    connectivity_scheme = recombine_args.get("recombine_args").get("connectivity_scheme", "lax")
    child = ["?" for _ in range(calculate_dimensionality(NN_architecture))]

    # For both parent genomes, get all their input_output_paths
    input_output_paths_genome1 = get_input_output_paths(NN_architecture, genome1)
    input_output_paths_genome2 = get_input_output_paths(NN_architecture, genome2)

    # Choose how many of the input_output_paths to inherit from each parent
    # input_output_paths_genome1 and input_output_paths_genome2 contain at least 1 path each
    count1 = np.random.randint(0, len(input_output_paths_genome1))
    if count1 == 0:
        if len(input_output_paths_genome2) == 1:
            count2 = 1
        else:
            count2 = np.random.randint(1, len(input_output_paths_genome2))
    else:
        count2 = np.random.randint(0, len(input_output_paths_genome2))

    # Get random indices
    random_indices1 = np.random.choice(np.arange(len(input_output_paths_genome1)), count1, replace=False)
    random_indices2 = np.random.choice(np.arange(len(input_output_paths_genome2)), count2, replace=False)

    # Get selected input_output_paths
    input_output_paths_child = [input_output_paths_genome1[idx] for idx in random_indices1] + \
                               [input_output_paths_genome2[idx] for idx in random_indices2]

    # Activate the input_output_paths in child
    for input_output_path in input_output_paths_child:
        child = activate_connections(child, NN_architecture, input_output_path)

    # Randomly assign values from parent genomes to remaining child bits that are not part of an input_output_path
    for idx, bit in enumerate(child):
        if bit == "?":
            child[idx] = np.random.choice((genome1[idx], genome2[idx]))

    return apply_connectivity_strictness_level(child, NN_architecture, connectivity_scheme)
