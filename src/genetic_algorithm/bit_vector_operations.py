from copy import copy
import numpy as np
from src.genetic_algorithm.evaluate_methods import evaluate_accuracy_goal
from src.genetic_algorithm.graph_operations import get_list_of_all_vertices, get_subnetwork_graph, find_all_paths, \
    find_all_parents, check_for_path, get_first_path, get_random_path
from src.genetic_algorithm.helper_functions import get_number_of_connections_between_layers, calculate_dimensionality


def get_valid_positions(NN_architecture, genome):
    input_output_paths = get_input_output_paths(NN_architecture, genome)

    # Select one input-output-path which can't be altered, to ensure there is no layer collapse
    random_idx = np.random.choice(len(input_output_paths))
    locked_input_output_path = input_output_paths[random_idx]

    valid_positions = get_connections(NN_architecture, genome, True)

    for idx, l in enumerate(valid_positions):
        l[1].remove(locked_input_output_path[idx][1])

    return valid_positions


def get_input_output_paths(NN_architecture, bit_vector):
    vertices = get_list_of_all_vertices(NN_architecture)
    connections = get_connections(NN_architecture, bit_vector)
    G = get_subnetwork_graph(connections, copy(vertices))
    destination_vertices = vertices[:NN_architecture[0]]
    source_vertices = vertices[-NN_architecture[-1]:]

    input_output_paths = []
    for src in source_vertices:
        for dest in destination_vertices:
            paths_to_dest = find_all_paths(find_all_parents(G, src), src, dest)
            paths_to_dest = [[(path[i], path[i + 1]) for i in range(len(path)) if i % 2 == 0]
                             for path in paths_to_dest]
            for paths in paths_to_dest:
                paths.sort()
            # Extract connections from path
            for paths in paths_to_dest:
                input_output_path = extract_connections(paths)
                input_output_paths.append(input_output_path)
    return input_output_paths


def extract_connections(path):
    connections = []
    for idx in range(len(path) - 1):
        x = path[idx]
        y = path[idx + 1]
        connections.append((x[0] + 1, (y[1], x[1])))
    return connections


def get_first_input_output_path(NN_architecture, bit_vector):
    vertices = get_list_of_all_vertices(NN_architecture)
    connections = get_connections(NN_architecture, bit_vector)
    G = get_subnetwork_graph(connections, copy(vertices))
    destination_vertices = vertices[:NN_architecture[0]]
    source_vertices = vertices[-NN_architecture[-1]:]

    # Shuffle source and destination vertices, to get random input_output_path
    np.random.shuffle(destination_vertices)
    np.random.shuffle(source_vertices)

    for src in source_vertices:
        for dest in destination_vertices:
            first_path = get_first_path(G, src, dest)
            if first_path:
                first_path.sort()
                return extract_connections(first_path)


def get_random_path_through_network(NN_architecture):
    vertices = get_list_of_all_vertices(NN_architecture)
    connections = get_connections(NN_architecture, [1 for _ in range(calculate_dimensionality(NN_architecture))])
    G = get_subnetwork_graph(connections, copy(vertices))
    source_vertices = vertices[-NN_architecture[-1]:]
    # Choose one random neuron from the output layer
    src_idx = np.random.choice(range(len(source_vertices)))
    random_path = get_random_path(G, source_vertices[src_idx])
    random_path.sort()
    return extract_connections(random_path)


def check_for_input_output_path(NN_architecture, bit_vector):
    vertices = get_list_of_all_vertices(NN_architecture)
    connections = get_connections(NN_architecture, bit_vector)
    G = get_subnetwork_graph(connections, copy(vertices))
    destination_vertices = vertices[:NN_architecture[0]]
    source_vertices = vertices[-NN_architecture[-1]:]

    for src in source_vertices:
        for dest in destination_vertices:
            if check_for_path(G, vertices, src, dest):
                return True
    return False


def get_neuron_connections(NN_architecture, bit_vector, neuron):
    """ Returns all the connections that go in and out (if output neuron, only ingoing connections) of the neuron.
     Only works for hidden and output neurons """
    assert neuron[0] != 0  # Assert neuron is not input neuron
    vertices = get_list_of_all_vertices(NN_architecture)
    connections = get_connections(NN_architecture, bit_vector)
    G = get_subnetwork_graph(connections, copy(vertices))
    parents = G[neuron]
    # The children are all the nodes that have the neuron as parent (in layer l+1, if neuron is in layer l)
    children = [node for node in G.keys() if neuron in G[node]]

    # Logic-wise it should be [neuron, parent] and [child, neuron], but then we would have to sort it, so we just
    # sort it right away
    ingoing_connections = [[parent, neuron] for parent in parents]
    outgoing_connections = [[neuron, child] for child in children]

    # Extract connections and store them together
    connections = []
    for connection in ingoing_connections:
        connections.append(extract_connections(connection))
    for connection in outgoing_connections:
        connections.append(extract_connections(connection))
    # Unpack connections into one list
    connections = [connection[0] for connection in connections]

    return connections


def remove_floating_connections(NN_architecture, bit_vector):
    """ floating connections are those connections, that are not connected to the input or the output layer
    floating connections only exist for NN_architectures with > 2 layers """
    non_connected_neurons = get_non_connected_hidden_neurons(NN_architecture, bit_vector)

    # Deactivate all the connections of the non_connected_neurons
    for neuron in non_connected_neurons:
        bit_vector = deactivate_connections(bit_vector, NN_architecture,
                                            get_neuron_connections(NN_architecture, bit_vector, neuron))

    return bit_vector


def get_non_connected_hidden_neurons(NN_architecture, bit_vector):
    """ The non_connected_neurons are those that might have floating connections """
    vertices = get_list_of_all_vertices(NN_architecture)
    connections = get_connections(NN_architecture, bit_vector)
    G = get_subnetwork_graph(connections, copy(vertices), back=False)
    G_back = get_subnetwork_graph(connections, copy(vertices))
    destination_vertices = vertices[NN_architecture[0]:-NN_architecture[-1]]  # Only hidden neurons can have floating connections
    output_neurons = vertices[-NN_architecture[-1]:]
    input_neurons = vertices[:NN_architecture[0]]

    # First check for path outgoing from the output layer
    non_connected_neurons_from_back = []
    for dest in destination_vertices:
        not_reachable = True
        for src in output_neurons:
            # Check for path outgoing from the input and from the output layer
            if check_for_path(G_back, vertices, src, dest):
                not_reachable = False
        if not_reachable:
            non_connected_neurons_from_back.append(dest)

    # Now check for path for those that are not connected to the output layer outgoing from the input layer
    non_connected_neurons = []
    for dest in non_connected_neurons_from_back:
        not_reachable = True
        for src in input_neurons:
            # Check for path outgoing from the input and from the output layer
            if check_for_path(G, vertices, src, dest, back=False):
                not_reachable = False
        if not_reachable:
            non_connected_neurons.append(dest)

    return non_connected_neurons


def check_for_floating_connections(NN_architecture, bit_vector):
    # Floating connections, only exist for networks with at least 3 layers
    if len(NN_architecture) <= 3:
        return False
    non_connected_neurons = get_non_connected_hidden_neurons(NN_architecture, bit_vector)
    for neuron in non_connected_neurons:
        if get_neuron_connections(NN_architecture, bit_vector, neuron):
            return True
    return False


def remove_non_input_output_paths(genome, NN_architecture):
    input_output_paths = get_input_output_paths(NN_architecture, genome)
    bit_vector = [0 for _ in range(calculate_dimensionality(NN_architecture))]

    # Deactivate all connections, that are not part of an input_output_path
    for input_output_path in input_output_paths:
        bit_vector = activate_connections(bit_vector, NN_architecture, input_output_path)
    return bit_vector


def check_if_all_connections_are_part_of_input_output_paths(NN_architecture, genome):
    input_output_paths = get_input_output_paths(NN_architecture, genome)
    activated_connections = get_connections(NN_architecture, genome)
    input_output_paths_set = set([conn for connection in input_output_paths for conn in connection])
    activated_connections_set = []
    for layer_connections in activated_connections:
        activated_connections_set.extend([(layer_connections[0], conn) for conn in layer_connections[1]])
    activated_connections_set = set(activated_connections_set)

    if any([True if conn not in input_output_paths_set else False for conn in activated_connections_set]):
        return False
    return True


def apply_connectivity_strictness_level(bit_vector, NN_architecture, connectivity_scheme):
    if connectivity_scheme == "strict":
        return remove_non_input_output_paths(bit_vector, NN_architecture)
    elif connectivity_scheme == "soft":
        return remove_floating_connections(NN_architecture, bit_vector)
    else:
        return bit_vector


def activate_connections(genome, NN_architecture, connections):
    for layer, conn in connections:
        # Get the corresponding bit position in bit vector
        pos = get_bit_position(NN_architecture, layer, conn)

        genome[pos] = 1
    return genome


def deactivate_connections(genome, NN_architecture, connections):
    for layer, conn in connections:
        # Get the corresponding bit position in bit vector
        pos = get_bit_position(NN_architecture, layer, conn)

        genome[pos] = 0
    return genome


def check_for_layer_collapse(NN_architecture, genome):
    number_of_connections_between_layers = get_number_of_connections_between_layers(NN_architecture)
    for idx, ele in enumerate(number_of_connections_between_layers):
        layer_range = (sum(number_of_connections_between_layers[:idx]),
                       sum(number_of_connections_between_layers[:idx]) +
                       number_of_connections_between_layers[idx])

        # Check if all connections were deactivated in this layer
        if all([c == 0 for c in genome[layer_range[0]:layer_range[1]]]):
            return True


def get_connections(NN_architecture, genome, all=False):
    """ If all=True, then this method returns all the connections (activated and deactivated) of the NN
    If genome is max(), then the returned value is the same for all=True and all=False """
    connections = []
    number_of_connections_between_layers = get_number_of_connections_between_layers(NN_architecture)
    for l in range(len(number_of_connections_between_layers)):
        connections.append((l + 1, []))

    for idx, ele in enumerate(number_of_connections_between_layers):
        layer_range = (sum(number_of_connections_between_layers[:idx]),
                        sum(number_of_connections_between_layers[:idx]) +
                        number_of_connections_between_layers[idx])

        # Split layer_range
        connections_sub_np = np.array_split(genome[layer_range[0]:layer_range[1]], NN_architecture[idx])
        connections_sub = []
        for array in connections_sub_np:
            connections_sub.append(list(array))

        for i, conn_list in enumerate(connections_sub):
            for j, bit in enumerate(conn_list):
                if all:
                    # Get all connections between layers
                    connections[idx][1].append((j, i))
                else:
                    # Get only activated connections
                    if bit == 1:
                        connections[idx][1].append((j, i))

    return connections


def unpack_connections(connections):
    connections_flattened = []
    for (l, conn) in connections:
        for connection in conn:
            connections_flattened.append((l, connection))
    return connections_flattened


def retrieve_bit_vector(NN_architecture, selected_connections):
    dimensionality = calculate_dimensionality(NN_architecture)
    bit_vector = [0 for _ in range(dimensionality)]
    for (layer, selections) in selected_connections:
        for connection in selections:
            bit_vector[get_bit_position(NN_architecture, layer, connection)] = 1
    return bit_vector


def get_bit_position(NN_architecture, layer, connection):
    number_of_connections_between_layers = get_number_of_connections_between_layers(NN_architecture)
    (j, i) = connection
    num_conn = [0]
    num_conn = num_conn + number_of_connections_between_layers
    return sum(num_conn[:layer]) + j + i * NN_architecture[layer]


def get_connection(NN_architecture, bit_vector, bit_position):
    """ Notice: The bit_position has to be in range(0, len(bit_vector)).
    An alternative method would be to use get_connections() unpack all connections into one list and use bit_position
    as index """
    number_of_connections_between_layers = get_number_of_connections_between_layers(NN_architecture)
    for idx, ele in enumerate(number_of_connections_between_layers):
        layer_range = (sum(number_of_connections_between_layers[:idx]),
                        sum(number_of_connections_between_layers[:idx]) +
                        number_of_connections_between_layers[idx])

        if bit_position in range(layer_range[0], layer_range[1]):
            connections_sub_np = np.array_split(bit_vector[layer_range[0]:layer_range[1]], NN_architecture[idx])
            number_of_sub_ranges = len(connections_sub_np)
            index_intervals = np.array_split(np.arange(layer_range[0], layer_range[1]), number_of_sub_ranges)
            (i, sublist) = [(idx, sublist) for idx, sublist in enumerate(index_intervals) if bit_position in sublist][0]
            j = sublist.tolist().index(bit_position)
            return idx+1, (j, i)


def get_continuing_connections(selections_layer_l_minus_1, n_neurons_layer_l):
    """
    selections_layer_l_minus_1: contains all the selected connections that lead into layer l-1
    n_neurons_layer_l: the number of neurons in layer l

    returns: all valid incoming connections from layer l-1 without duplicates
    """
    return list(set([(j, i) for j in range(n_neurons_layer_l)
                     for i in [cell for (cell, pos) in selections_layer_l_minus_1]]))


def check_if_bit_vector_passes_accuracy_filter(bit_vector, lower_bound, args):
    NN_architecture = args.get("environment_args").get("NN_architecture")
    model = args.get("environment_args").get("model")
    X_for_accuracy = args.get("environment_args").get("X_for_accuracy")
    Y_for_accuracy = args.get("environment_args").get("Y_for_accuracy")
    if evaluate_accuracy_goal(bit_vector, NN_architecture, model, X_for_accuracy, Y_for_accuracy) >= lower_bound:
        return True
    return False


def check_if_bit_vector_has_accuracy_in_bound(bit_vector, lower_bound, upper_bound, args):
    NN_architecture = args.get("environment_args").get("NN_architecture")
    model = args.get("environment_args").get("model")
    X_for_accuracy = args.get("environment_args").get("X_for_accuracy")
    Y_for_accuracy = args.get("environment_args").get("Y_for_accuracy")
    if lower_bound <= evaluate_accuracy_goal(bit_vector, NN_architecture, model, X_for_accuracy, Y_for_accuracy) \
            < upper_bound:
        return True
    return False


if __name__ == "__main__":
    import itertools
    network_architecture = [2, 4, 1]
    list_of_bit_vectors = itertools.product([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    list_of_valid_bit_vectors = []
    for bit_v in list_of_bit_vectors:
        if check_for_input_output_path(network_architecture, bit_v):
            list_of_valid_bit_vectors.append(bit_v)
    print(len(list_of_valid_bit_vectors))
    """for bit_v in list_of_valid_bit_vectors:
        print(bit_v)"""
    count = 0
    for bit_v in list_of_valid_bit_vectors:
        if bit_v.count(1) == 2:
            count += 1
    print(count)