from collections import defaultdict
from copy import deepcopy
import numpy as np


class Graph:
    def __init__(self, edges, vertices):

        self.adjList = {v: [] for v in vertices}

        # allocate memory for the adjacency list
        for i in range(len(vertices)):
            self.adjList[i] = []

        # add edges to the directed graph
        for (src, dest) in edges:
            # allocate node in adjacency list from src to dest
            self.adjList[src].append(dest)


# Function to find all parents of a node
def getParent(graph, node, vertices):
    # collect parents as a set in order to avoid duplicate entries
    parents = set()
    for src in vertices:
        # Iterate over all edges from src node
        for dest in graph.adjList[src]:
            # if destination is our needed child add source as parent
            if dest == node:
                parents.add(src)

    # Return all parents as a list
    return list(parents)


def get_edges(connections):
    edges = []
    for (layer, layer_connections) in connections:
        for (y, x) in layer_connections:
            edges.append(((layer - 1, x), (layer, y)))
    return edges


def get_edges_back(connections):
    connections.sort(reverse=True)
    edges = []
    for (layer, layer_connections) in connections:
        for (y, x) in layer_connections:
            edges.append(((layer, y), (layer - 1, x)))
    return edges


def get_list_of_all_vertices(NN_architecture):
    vertices = []
    for l, n in enumerate(NN_architecture):
        for i in range(n):
            vertices.append((l, i))
    return vertices


def get_layer_from_vertex(vertex):
    return vertex[0]


def get_neuron_number_from_vertex(vertex):
    return vertex[1]


def get_subnetwork_graph(connections, vertices, back=True):
    """
    Notice, if back=True, then the graph is constructed from the output layer back to the input layer,
    therefore a parent is a node that is further down in the direction of the input layer than the considered node.
    Otherwise, the graph is constructed from the input layer to the output layer
    """
    vertices = deepcopy(vertices)
    G = {}
    if back:
        edges = get_edges(connections)
        vertices.sort(reverse=True)
        v_0 = 0
    else:
        edges = get_edges_back(connections)
        v_0 = vertices[-1][0]
    graph = Graph(edges, vertices)
    for v in vertices:
        if v[0] != v_0:
            G[v] = getParent(graph, v, vertices)
    return G


# modified BFS
def find_all_parents(G, s):
    Q = [s]
    parents = defaultdict(set)
    while len(Q) != 0:
        v = Q[0]
        Q.pop(0)
        for w in G.get(v, []):
            parents[w].add(v)
            Q.append(w)
    return parents


# recursive path-finding function (assumes that there exists a path in G from a to b)
def find_all_paths(parents, a, b):
    return [a] if a == b else [y + b for x in list(parents[b]) for y in find_all_paths(parents, a, x)]


def check_for_path(G, vertices, src, dest, back=True):
    # Mark all the vertices as not visited
    visited = {v: False for v in vertices}

    # Create a queue for BFS
    queue = []

    # Mark the source node as visited and enqueue it
    queue.append(src)
    visited[src] = True

    while queue:

        # Dequeue a vertex from queue
        n = queue.pop(0)

        # If this adjacent node is the destination node,
        # then return true
        if n == dest:
            return True

        if back:
            # if we check backwards through the network, then we don't consider the input neurons
            v_0 = 0
        else:
            # if we check backwards through the network, then we don't consider the output neurons
            v_0 = vertices[-1][0]
        # If layer is not v_0, continue to do BFS
        if n[0] != v_0:
            for i in G[n]:
                if not visited[i]:
                    queue.append(i)
                    visited[i] = True

    # If BFS is complete without visited dest
    return False


def get_first_path(G, src, dest):
    if src == dest:
        return [dest]
    else:
        if src[0] != 0:
            for i in G[src]:
                sub_path = get_first_path(G, i, dest)
                if sub_path is not None:
                    return [src] + sub_path


def get_random_path(G, src):
    """ G is a fully connected graph """
    parents = G[src]
    path = [src]
    while parents:
        random_idx = np.random.choice(range(len(parents)))
        n = parents[random_idx]
        path.append(n)
        if n[0] != 0:
            parents = G[n]
        else:
            return path
