#============================================================================== #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                           #
# All rights reserved.                                                          #
#                                                                               #
# This source code and the accompanying materials are made available under      #
# the terms of the Apache License 2.0 which accompanies this distribution.      #
# The QAOA-GPT implementation in CUDA-Q is based on this paper:                 #
# https://arxiv.org/pdf/2504.16350                                              #
# Usage or reference of this code or algorithms requires citation of the paper: #
# Ilya Tyagin, Marwa Farag, Kyle Sherbert, Karunya Shirali, Yuri Alexeev,       #
# Ilya Safro "QAOA-GPT: Efficient Generation of Adaptive and Regular Quantum    #
# Approximate Optimization Algorithm Circuits", IEEE International Conference   #
# on Quantum Computing and Engineering (QCE), 2025.                             #
# ============================================================================= #

import networkx as nx
import numpy as np
import random

def graph_to_adj_m(g):
    """Convert a NetworkX graph to an adjacency matrix (numpy array)."""
    return nx.to_numpy_array(g)

def graph_to_edgelist(g):
    """Return a weighted edge list: (src, dst, weight) for all edges."""
    return [(u, v, d.get('weight', 1.0)) for u, v, d in g.edges(data=True)]

def edgelist_to_graph(edgelist, num_vertices=0):
    """Create a weighted undirected graph from an edge list."""
    if num_vertices == 0:
        num_vertices = max(max(src, dst) for src, dst, _ in edgelist) + 1
    g = nx.Graph()
    g.add_nodes_from(range(num_vertices))
    for src, dst, w in edgelist:
        g.add_edge(src, dst, weight=w)
    return g

def generate_random_graph(n, methods=None):
    """
    Generate a connected random graph using specified methods with random parameters.
    Returns: (graph, method)
    """
    if methods is None:
        methods = [
            "erdos_renyi",
            "barabasi_albert",
            "watts_strogatz",
            "random_regular",
            "bipartite"
        ]
    method = random.choice(methods)
    while True:
        if method == "erdos_renyi":
            p = random.uniform(0.3, 0.9)
            G = nx.erdos_renyi_graph(n, p)
        elif method == "barabasi_albert":
            m = random.randint(1, n - 1)
            G = nx.barabasi_albert_graph(n, m)
        elif method == "watts_strogatz":
            k = random.randint(2, n - 1)
            p = random.uniform(0.1, 1.0)
            G = nx.watts_strogatz_graph(n, k, p)
        elif method == "random_regular":
            d = random.randint(2, n - 1)
            if n * d % 2 == 0:
                G = nx.random_regular_graph(d, n)
            else:
                continue
        elif method == "bipartite":
            n1 = random.randint(2, n - 1)
            n2 = n - n1
            G = nx.complete_bipartite_graph(n1, n2)
        else:
            raise ValueError(f"Unknown method: {method}")

        if G.number_of_edges() > 0 and nx.is_connected(G):
            break
    return G, method


def add_rand_weights_to_graph(g, neg_weights=False):
    """
    Add random weights to all edges in the graph.
    Returns a new weighted graph.
    """
    g_weighted = nx.Graph()
    g_weighted.add_nodes_from(g.nodes())
    for u, v in g.edges():
        w = round(random.random(), 2)
        while w == 0.0:
            w = round(random.random(), 2)
        if neg_weights:
            w *= random.choice([-1, 1])
        g_weighted.add_edge(u, v, weight=w)
    return g_weighted

def norm_elist_weights(e_list):
    """
    Normalize edge weights so their sum of absolute values is 1.
    Returns: (normalized_edge_list, total_weight)
    """
    total_weight = sum(abs(weight) for _, _, weight in e_list)
    scaled_weighted_edge_list = [
        (u, v, weight / total_weight) for u, v, weight in e_list
    ]
    return scaled_weighted_edge_list, total_weight