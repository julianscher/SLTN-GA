import numpy as np
from src.genetic_algorithm.bit_vector_operations import get_valid_positions, get_connections, check_for_input_output_path, \
    get_bit_position, get_random_path_through_network, activate_connections, \
    deactivate_connections, get_neuron_connections, remove_floating_connections, remove_non_input_output_paths, \
    apply_connectivity_strictness_level
from src.genetic_algorithm.graph_operations import get_list_of_all_vertices
from src.genetic_algorithm.helper_functions import calculate_dimensionality


# ============ Mutation methods ============#

def single_point_mutation_path_fixed(genome, mutate_args):
    """ Fix one input_output_path and deny mutating bits of connections on this path. All other bits can be mutated """
    NN_architecture = mutate_args.get("environment_args").get("NN_architecture")
    connectivity_scheme = mutate_args.get("mutate_args").get("connectivity_scheme", "lax")

    connections = get_valid_positions(NN_architecture, genome)
    mutant = apply_mutation(genome.copy(), NN_architecture, connections)
    return apply_connectivity_strictness_level(mutant, NN_architecture, connectivity_scheme)


def single_point_mutation(genome, mutate_args):
    """ Mutate random bit, then check if there is still an input_output_path """
    NN_architecture = mutate_args.get("environment_args").get("NN_architecture")
    connectivity_scheme = mutate_args.get("mutate_args").get("connectivity_scheme", "lax")
    connections = get_connections(NN_architecture, genome, True)

    while True:
        mutant = apply_mutation(genome.copy(), NN_architecture, connections)

        # Check if resulting bit_vector has an input_output_path
        if check_for_input_output_path(NN_architecture, mutant):
            return apply_connectivity_strictness_level(mutant, NN_architecture, connectivity_scheme)


def single_point_mutation_negligent_optimized(genome, mutate_args):
    """ Mutate random bit, then check if there is still an input_output_path """
    random_idx = np.random.choice(len(genome))
    bit = genome[random_idx]
    mutant = genome.copy()
    if bit:
        mutant[random_idx] = 0
    else:
        mutant[random_idx] = 1
    return mutant


def path_flip_mutation(genome, mutate_args):
    NN_architecture = mutate_args.get("environment_args").get("NN_architecture")
    connectivity_scheme = mutate_args.get("mutate_args").get("connectivity_scheme", "lax")
    while True:
        mutant = genome.copy()

        # Get random path through network (can contain deactivated connections)
        random_path = get_random_path_through_network(NN_architecture)

        # Perform bit-flip mutation on path
        for layer, conn in random_path:
            mutant = perform_bit_flip(mutant, NN_architecture, layer, conn)

        # Check if resulting bit_vector has an input_output_path
        if check_for_input_output_path(NN_architecture, mutant):
            return apply_connectivity_strictness_level(mutant, NN_architecture, connectivity_scheme)


def path_mutation(genome, mutate_args):
    NN_architecture = mutate_args.get("environment_args").get("NN_architecture")
    connectivity_scheme = mutate_args.get("mutate_args").get("connectivity_scheme", "lax")
    alpha = mutate_args.get("mutate_args").get("alpha")
    mutant = genome.copy()
    while True:
        r = np.random.random()

        # Get random path through network (can contain deactivated connections)
        random_path = get_random_path_through_network(NN_architecture)

        # Activate or deactivate all connections on random_path
        if r < alpha:
            mutant = activate_connections(mutant, NN_architecture, random_path)
            return apply_connectivity_strictness_level(mutant, NN_architecture, connectivity_scheme)
        else:
            mutant = deactivate_connections(mutant, NN_architecture, random_path)
            if check_for_input_output_path(NN_architecture, mutant):
                return apply_connectivity_strictness_level(mutant, NN_architecture, connectivity_scheme)


def neuron_mutation(genome, mutate_args):
    NN_architecture = mutate_args.get("environment_args").get("NN_architecture")
    connectivity_scheme = mutate_args.get("mutate_args").get("connectivity_scheme", "lax")
    alpha = mutate_args.get("mutate_args").get("alpha", 0.5)
    mutant = genome.copy()

    # a neuron is defined by (layer, neuron_number)
    neurons = get_list_of_all_vertices(NN_architecture)
    valid_neurons = [neuron for neuron in neurons if neuron[0] % 2 != 0]

    while True:
        # Pick random neuron from valid_neurons
        random_idx = np.random.choice(range(len(valid_neurons)))
        neuron = valid_neurons[random_idx]

        # Get all incoming and outgoing connections
        connections = get_neuron_connections(NN_architecture,
                                             [1 for _ in range(calculate_dimensionality(NN_architecture))], neuron)

        r = np.random.random()

        # Activate or deactivate all incoming or outgoing connections of the neuron
        if r < alpha:
            mutant = activate_connections(mutant, NN_architecture, connections)
            if check_for_input_output_path(NN_architecture, mutant):
                return apply_connectivity_strictness_level(mutant, NN_architecture, connectivity_scheme)
        else:
            mutant = deactivate_connections(mutant, NN_architecture, connections)
            if check_for_input_output_path(NN_architecture, mutant):
                return apply_connectivity_strictness_level(mutant, NN_architecture, connectivity_scheme)


# ================= Helper functions ================= #


def apply_mutation(genome, NN_architecture, connections):

    # Choose random connection from connections
    layer = np.random.choice(range(1, len(connections) + 1))
    random_idx = np.random.choice(len(connections[layer - 1][1]))
    conn = connections[layer - 1][1][random_idx]

    return perform_bit_flip(genome, NN_architecture, layer, conn)


def perform_bit_flip(genome, NN_architecture, layer, conn):
    # Get the corresponding bit position in bit vector
    pos = get_bit_position(NN_architecture, layer, conn)

    # Apply mutation
    if genome[pos] == 1:
        genome[pos] = 0
    else:
        genome[pos] = 1

    return genome