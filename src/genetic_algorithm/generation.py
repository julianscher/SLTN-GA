import math
import time
import numpy as np
from typing import Union, List
from src.genetic_algorithm.evaluate_methods import evaluate_bit_vector_accuracy
from src.genetic_algorithm.helper_functions import get_number_of_connections_between_layers, calculate_dimensionality
from src.genetic_algorithm.bit_vector_operations import get_continuing_connections, retrieve_bit_vector, get_connections, \
    check_for_input_output_path, activate_connections, remove_floating_connections, \
    check_if_bit_vector_passes_accuracy_filter, check_if_bit_vector_has_accuracy_in_bound, get_input_output_paths


def generate_bit_vector(generate_args) -> List[int]:
    """
    Potentially throws IndexError, because of sample sizes if population is of size 0 but 1 element should be selected
    """
    NN_architecture = generate_args.get("environment_args").get("NN_architecture")
    arg = generate_args.get("generate_args").get("arg")
    number_of_layers = len(NN_architecture)
    selected_connections = []
    number_of_connections_between_layers = get_number_of_connections_between_layers(NN_architecture)
    dimensionality = calculate_dimensionality(NN_architecture)

    if arg == "max":
        for idx in range(len(NN_architecture) - 1):
            # Save selected connections for debugging purpose
            selected_connections.append((idx + 1, [(j, i) for j in range(NN_architecture[idx + 1])
                                                   for i in range(NN_architecture[idx])]))
        return [1 for _ in range(dimensionality)]
    elif arg == "min" or arg == "emph1":
        # emph1 ("emphasize 1s") generates bit_vector that is near max with higher probability
        # To construct input-output-paths, repeat for all layers:
        # select 1 connection between layer l and layer l+1 from the list of continuing connections
        # dependent on that selection, determine the continuing connections between layer l and layer l+1
        if arg == "min":
            # select one random connection between layer 0 and layer 1
            connections = [(j, i) for j in range(NN_architecture[1])
                           for i in range(NN_architecture[0])]
            random_idx = np.random.choice(len(connections))
            selections_l_1 = [connections[random_idx]]
        else:
            # select random connections between layer 0 and layer 1 without repetition
            connections = [(j, i) for j in range(NN_architecture[1])
                           for i in range(NN_architecture[0])]
            random_size = np.random.choice(range(1, number_of_connections_between_layers[0] + 1))
            random_idxs = np.random.choice(len(connections), size=random_size, replace=False)
            selections_l_1 = [connections[i] for i in random_idxs]
        selected_connections.append((1, selections_l_1))  # selected connections leading into layer 1

        selections = selections_l_1
        for l in range(2, number_of_layers):
            continuing_connections = get_continuing_connections(selections, NN_architecture[l])
            if arg == "min":
                random_idx = np.random.choice(len(continuing_connections))
                selections = [continuing_connections[random_idx]]
            else:
                random_size = np.random.choice(range(1, len(continuing_connections) + 1))
                random_idxs = np.random.choice(len(continuing_connections), random_size, replace=False)
                selections = [continuing_connections[i] for i in random_idxs]
            selected_connections.append((l, selections))

        # At this point the bit-vector has only fully connected paths
        bit_vector = retrieve_bit_vector(NN_architecture, selected_connections)

        if arg != "min":
            # Now arbitrarily add further connections, since there is at least one established input-output-path
            all_connections = get_connections(NN_architecture, bit_vector, True)

            # Get connections that are not in selected_connections
            deactivated_connections = get_connections(NN_architecture, bit_vector, True)
            for idx, l in enumerate(all_connections):
                for t in l[1]:
                    # Remove all connections that are part of selected_connections
                    if selected_connections[idx][1].count(t) == 1:
                        deactivated_connections[idx][1].remove(t)

            for idx, l in enumerate(selected_connections):
                # It should also be possible that no further addition is made, therefore the range starts at 0
                if len(deactivated_connections[idx][1]) > 0:
                    random_size = np.random.choice(range(len(deactivated_connections[idx][1]) + 1))
                    random_idxs = np.random.choice(len(deactivated_connections[idx][1]), random_size, replace=False)
                    # Activate random number of formerly deactivated connections
                    l[1].extend([deactivated_connections[idx][1][i] for i in random_idxs])

        return retrieve_bit_vector(NN_architecture, selected_connections)
    elif type(arg) is float:
        accuracy_bound_not_met = True
        while accuracy_bound_not_met:
            # Create random bit_vector
            random_bit_vector = [np.random.choice([0, 1]) for _ in range(dimensionality)]

            # Check if bit_vector's accuracy has at least the boundary value
            if evaluate_bit_vector_accuracy(random_bit_vector, generate_args) >= arg:
                return random_bit_vector
    else:
        return generate_lax_connectivity(generate_args)


def generate_strict_connectivity(generate_args):
    NN_architecture = generate_args.get("environment_args").get("NN_architecture")
    # Get input_output_paths of max()
    input_output_paths = generate_args.get("generate_args").get("input_output_paths", None)
    if not input_output_paths:
        max_bit_vector = generate_bit_vector({"environment_args": {"NN_architecture": NN_architecture},
                                              "generate_args": {"arg": "max"}})
        input_output_paths = get_input_output_paths(NN_architecture, max_bit_vector)
        generate_args.get("generate_args")["input_output_paths"] = input_output_paths
    dimensionality = calculate_dimensionality(NN_architecture)
    max_connections_between_layers = max(get_number_of_connections_between_layers(NN_architecture))
    bit_vector = [0 for _ in range(dimensionality)]

    # Choose how many input_output_paths the bit_vector should have
    count = np.random.randint(1, max_connections_between_layers + 1)

    # Select which input_output_paths the bit_vector should have
    random_indices = np.random.choice(np.arange(len(input_output_paths)), count, replace=False)
    input_output_paths_bit_vector = [input_output_paths[idx] for idx in random_indices]

    # Activate the input_output_paths in bit_vector
    for input_output_path in input_output_paths_bit_vector:
        bit_vector = activate_connections(bit_vector, NN_architecture, input_output_path)
    return bit_vector


def generate_soft_connectivity(generate_args):
    NN_architecture = generate_args.get("environment_args").get("NN_architecture")
    dimensionality = calculate_dimensionality(NN_architecture)
    ones_count_min = len(NN_architecture) - 1  # the number of 1s in a min()

    while True:
        bit_vector = [1 for _ in range(dimensionality)]
        # Prune random weights
        count = np.random.randint(0, dimensionality - ones_count_min)
        random_indices = np.random.choice(np.arange(dimensionality), count, replace=False)
        bit_vector = [0 if idx in random_indices else bit for idx, bit in enumerate(bit_vector)]

        # Check if input-output-path still exists -> Instead, we could also use evaluate_bit_vector_accuracy
        if check_for_input_output_path(NN_architecture, bit_vector):
            break

    # Remove floating connections from bit_vector
    bit_vector = remove_floating_connections(NN_architecture, bit_vector)

    return bit_vector


def generate_lax_connectivity(generate_args):
    NN_architecture = generate_args.get("environment_args").get("NN_architecture")
    dimensionality = calculate_dimensionality(NN_architecture)
    not_input_output_path_exists = True
    while not_input_output_path_exists:
        # Create random bit_vector
        random_bit_vector = [np.random.choice([0, 1]) for _ in range(dimensionality)]

        # Check if input-output-path exists
        if check_for_input_output_path(NN_architecture, random_bit_vector):
            return random_bit_vector


def generate_lax_connectivity_alternative(generate_args):
    dimensionality = calculate_dimensionality(generate_args.get("environment_args").get("NN_architecture"))
    bound = generate_args.get("generate_args").get("bound", 0.51)
    while True:
        # Create random bit_vector
        random_bit_vector = [np.random.choice([0, 1]) for _ in range(dimensionality)]

        # Check if bit_vector's accuracy has at least the boundary value
        if check_if_bit_vector_passes_accuracy_filter(random_bit_vector, bound, generate_args):
            return random_bit_vector


def generate_lax_connectivity_alternative_with_upper_bound(generate_args):
    dimensionality = calculate_dimensionality(generate_args.get("environment_args").get("NN_architecture"))
    lower_bound = generate_args.get("generate_args").get("lower_bound")  # inclusive
    upper_bound = generate_args.get("generate_args").get("upper_bound")  # exclusive
    while True:
        # Create random bit_vector
        random_bit_vector = [np.random.choice([0, 1]) for _ in range(dimensionality)]

        # Check if bit_vector's accuracy is in bound of lower and upper bound
        if check_if_bit_vector_has_accuracy_in_bound(random_bit_vector, lower_bound, upper_bound, generate_args):
            return random_bit_vector


def generate_lax_connectivity_alternative_faster(generate_args):
    NN_architecture = generate_args.get("environment_args").get("NN_architecture")
    dimensionality = calculate_dimensionality(generate_args.get("environment_args").get("NN_architecture"))
    bound = generate_args.get("generate_args").get("bound", 0.51)
    number_of_unsuccessful_tries = 0
    while True:
        # Create random bit_vector
        random_bit_vector = [np.random.choice([0, 1]) for _ in range(dimensionality)]

        # Check if bit_vector's accuracy has at least the boundary value
        if number_of_unsuccessful_tries <= 500 \
                and check_if_bit_vector_passes_accuracy_filter(random_bit_vector, bound, generate_args):
            return random_bit_vector
        elif 500 < number_of_unsuccessful_tries <= 1000 and bound - 0.1 >= 0.6 and \
                check_if_bit_vector_passes_accuracy_filter(random_bit_vector, 0.6, generate_args):
            return random_bit_vector
        elif 1000 < number_of_unsuccessful_tries <= 1500 and bound >= 0.51 and \
                check_if_bit_vector_passes_accuracy_filter(random_bit_vector, 0.51, generate_args):
            return random_bit_vector
        elif number_of_unsuccessful_tries > 1500 and check_for_input_output_path(NN_architecture, random_bit_vector):
            return random_bit_vector

        number_of_unsuccessful_tries += 1


def accuracy_bound_change(dist):
    return 1/10 * (0.051 + 0.93 * math.exp(-9.36 * dist))


def generate_lax_connectivity_learned_accuracy_filter(generate_args):
    """
    Two notes on this method:
    - Sometimes the algorithm takes some time with evaluating the accuracy of the subnetwork in NN
    - If not generation_arg for iteration_time is given, the algorithm won't be deterministic
    - In its current form its only applicable to binary classification problems
    """
    NN_architecture = generate_args.get("environment_args").get("NN_architecture")
    dimensionality = calculate_dimensionality(NN_architecture)
    # If there is no value, that means this is the first individual or the generated individuals before were
    # successfully created with 0.85 bound
    bound = generate_args.get("generate_args").get("bound", 0.85)

    # measure time for creating one bit_vector
    start_time = time.time()
    # Create random bit_vector
    random_bit_vector = [np.random.choice([0, 1]) for _ in range(dimensionality)]

    # If bound was decreased so much, that it passed 0.51 accept every bit_vector that has an input_output_path
    if bound == 0.51:
        return generate_lax_connectivity(generate_args)

    # Check if bit_vector's accuracy has at least the boundary value
    if check_if_bit_vector_passes_accuracy_filter(random_bit_vector, bound, generate_args):
        return random_bit_vector
    else:
        iteration_time = time.time() - start_time
        if generate_args.get("generate_args").get("iteration_time"):
            iteration_time = generate_args.get("generate_args").get("iteration_time")

        maximum_reasonable_generation_time = generate_args.get("generate_args").get("max_reas_gen_time",  0.5)  # 0.5 seconds
        maximum_given_generation_time = generate_args.get("generate_args").get("max_giv_gen_time",  1.5)  # 1.5 seconds
        unsuccessful_tries_upper_bound = math.floor(maximum_reasonable_generation_time/iteration_time)
        max_unsuccessful_tries_upper_bound = math.floor(maximum_given_generation_time/iteration_time)
        upper_bound_distance = max_unsuccessful_tries_upper_bound - unsuccessful_tries_upper_bound

        # Handle edge cases
        if unsuccessful_tries_upper_bound < 1 or max_unsuccessful_tries_upper_bound < 1 or upper_bound_distance == 0:
            return generate_lax_connectivity(generate_args)

        number_of_unsuccessful_tries = 0
        while number_of_unsuccessful_tries <= max_unsuccessful_tries_upper_bound:
            # Create random bit_vector
            random_bit_vector = [np.random.choice([0, 1]) for _ in range(dimensionality)]

            # Check if bit_vector's accuracy has at least the boundary value
            if check_if_bit_vector_passes_accuracy_filter(random_bit_vector, bound, generate_args):
                if number_of_unsuccessful_tries > unsuccessful_tries_upper_bound:
                    # standardized distance of number_of_unsuccessful_tries to max_unsuccessful_tries_upper_bound
                    unsuccessful_tries_distance = number_of_unsuccessful_tries/upper_bound_distance

                    # Decrease accuracy bound while not passing the minimum accuracy
                    # to ensure there are input_output_paths
                    decreased_bound = max(bound - accuracy_bound_change(unsuccessful_tries_distance), 0.51)

                    # Update bound for successive generations
                    generate_args.get("generate_args")["bound"] = decreased_bound

                return random_bit_vector

            number_of_unsuccessful_tries += 1

        # If no valid bit_vector could be generated in max_unsuccessful_tries_upper_bound, decrease accuracy bound
        # with the highest change and call generate_lax_connectivity_learned_accuracy_filter recursively with the new
        # decreased_bound

        # num_rec is a value that slows down the change with increasing number of recursive calls
        if not generate_args.get("generate_args").get("num_rec"):
            generate_args.get("generate_args")["num_rec"] = 1

        num_rec = generate_args.get("generate_args").get("num_rec")

        decreased_bound = max(bound - math.pow(0.1, num_rec), 0.51)

        # Update bound for recursive call and update num_rec
        generate_args.get("generate_args")["bound"] = decreased_bound
        generate_args.get("generate_args")["num_rec"] += 0.15

        return generate_lax_connectivity_learned_accuracy_filter(generate_args)


def generate_negligent(generate_args):
    NN_architecture = generate_args.get("environment_args").get("NN_architecture")
    dimensionality = calculate_dimensionality(NN_architecture)

    # Create random bit_vector
    return [np.random.choice([0, 1]) for _ in range(dimensionality)]


