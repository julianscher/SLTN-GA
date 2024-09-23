import numpy as np
import scipy.spatial.distance as dist

# ============ random value functions ============#

def randi(**kwargs):
    """ draws random int (0 or 1) ( or in 'min'-'max' range if specified in kwargs) """
    dim = ('dim' in kwargs) and kwargs['dim'] or 1
    min_val = ('min' in kwargs) and kwargs['min'] or 0
    max_val = ('max' in kwargs) and kwargs['max'] or 1
    choice = np.random.randint(min_val, max_val)
    return choice


def randf(**kwargs):
    """ draws random float in [0;1] (or in 'min'-'max' range if specified in kwargs) """
    dim = ('dim' in kwargs) and kwargs['dim'] or 1
    min_val = ('min' in kwargs) and kwargs['min'] or 0
    max_val = ('max' in kwargs) and kwargs['max'] or 1
    choice = np.random.random_sample()
    return (max_val - min_val) * choice + min_val


def rando(option1, option2, chance=0.5):
    """ returns one of the options depending on the chance """
    if randb(chance):
        return option1
    else:
        return option2


def randb(chance=0.5):
    """ returns a random boolean """
    return randf() < chance


# ============ NN architecture related methods ============#

def get_number_of_connections_between_layers(NN_architecture):
    if len(NN_architecture) > 1:
        return [NN_architecture[i] * NN_architecture[i + 1]
                for i in range(len(NN_architecture) - 1)]
    elif len(NN_architecture) == 1:
        return NN_architecture[0]
    else:
        return 0


def calculate_dimensionality(NN_architecture):
    num = get_number_of_connections_between_layers(NN_architecture)
    if type(num) == int:
        return num
    else:
        return sum(num)


# ============ diversity related methods ============#

def calculate_diversity_less_efficient(population):
    # the distance between two bit_vectors is the sum of differing bits
    total_distance = 0
    for i, current_individual in enumerate(population.individuals):
        for j, individual in enumerate(population.individuals):
            if i != j:
                total_distance += calculate_distance_between_individuals_genomes(current_individual.genome,
                                                                                 individual.genome)
    return total_distance // 2


def calculate_diversity(population):
    genomes = np.array([individual.genome for individual in population.individuals])
    return int(np.sum(dist.pdist(genomes, 'cityblock')))


def calculate_max_diversity(population_size, min, max):
    cluster1_size = population_size // 2
    cluster2_size = population_size - cluster1_size
    genomes = [min for _ in range(cluster1_size)] + [max for _ in range(cluster2_size)]
    return int(np.sum(dist.pdist(genomes, 'cityblock')))


def calculate_relative_diversity(population, max_diversity):
    return calculate_diversity(population) / max_diversity


def calculate_distance_between_individuals_genomes(genome1, genome2):
    distance = 0
    for idx, bit in enumerate(genome1):
        if bit != genome2[idx]:
            distance += 1
    return distance


# ============ other methods ============#

def get_method_args(environment_args, method_args, method_args_name):
    blacklist = ["mutate_args", "recombine_args", "generate_args", "evaluate_individual_args",
                 "evaluate_for_selection_args", "evaluate_for_recombination_args", "selection",
                 "selection_args"]
    return {"environment_args": {arg: environment_args.get(arg) for arg in environment_args.keys()
                                 if str(arg) not in blacklist}, method_args_name: method_args}


def winning_ticket_included(population):
    for individual in population.individuals:
        if individual.genome in [
            [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
             1, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
                                       0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]]:
            return True
    return False


def get_parents_collection_dfs(individual, population_age, collection):
    """ Because we know the order of DFS, one can build a tree from collection (currently, by hand) """
    if individual.parents is None:
        return collection
    for parent in individual.parents:  # matches DFS
        collection.append((population_age - 1, parent))
        get_parents_collection_dfs(parent, population_age - 1, collection)
    return collection


def get_genealogy_bfs(individual, population_age):
    """
    This method returns a dictionary with all the parents the submitted individual is based on
    !Note: Because of the structure of parents_dict, in case of equal subtrees, only the first occurrence of this
    subtree is stored in the dictionary. The missing subtrees must be derived manually
    """
    parents_dict = {individual: None}
    visited = [(population_age, individual)]  # List for visited parents.
    queue = [(population_age, individual)]  # Initialize a queue

    while queue:  # Creating loop to visit each parent
        ind = queue.pop(0)

        if ind[1].parents:
            parents_dict[ind[1]] = ind[1].parents
            for parent in ind[1].parents:
                if (ind[0] - 1, parent) not in visited:
                    visited.append((ind[0] - 1, parent))
                    queue.append((ind[0] - 1, parent))
    return parents_dict


def get_leaves_dfs(individual, population_age):
    parents_collection = get_parents_collection_dfs(individual, population_age, [(population_age, individual)])
    leaves = []
    for i in range(len(parents_collection)):
        if i < len(parents_collection) - 1:
            # If in the list the adjacent node isn't smaller, that means that item at i is a leave
            if parents_collection[i + 1][0] >= parents_collection[i][0]:
                leaves.append(parents_collection[i])
        else:
            leaves.append(parents_collection[i])
    # without the following line, leaves contains more information than which can be gathered by using get_leaves_bfs
    leaves = set([ind for (age, ind) in leaves])  # uses set operation, to remove leaves from equal subtrees
    return leaves


def get_leaves_bfs(individual, population_age):
    all_parents = get_genealogy_bfs(individual, population_age)
    leaves = []
    for parents in all_parents.values():
        if parents:
            for parent in parents:
                # If node has no parents, that means that node is a leave
                if parent not in all_parents.keys():
                    leaves.append(parent)
        else:
            leaves.append(list(all_parents.keys())[0])
    return leaves


def get_mean_population_fitness(population):
    population.evaluate_by_accuracy()
    fitness = []
    for individual in population.individuals:
        fitness.append(individual.fitness)
    return np.mean(fitness)

