from utilities.fc.fc_bit_vector_operations import get_bit_position
from utilities.ga_utils import randi


def apply_mutation(genome, NN_architecture, connections):

    # Choose random connection from connections
    layer = randi(min=1, max=len(connections) + 1)
    random_idx = randi(min=0, max=len(connections[layer - 1][1]))
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
