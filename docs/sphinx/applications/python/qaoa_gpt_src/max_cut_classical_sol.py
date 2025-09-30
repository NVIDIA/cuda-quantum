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

import random
import numpy as np
import networkx as nx


# Compute the max cut of a graph and its value using brute force, simulated annealing, or one-exchange algorithm.
# The brute force method is exact but computationally expensive for large graphs.
# Simulated annealing provides a probabilistic approach that can yield good results in reasonable time.
# One-exchange is a heuristic method provided by NetworkX.
def brute_force_max_cut(graph):
    """
    Computes the Max-Cut of a weighted graph using a brute-force approach.

    Args:
        graph (nx.Graph): The input weighted graph.

    Returns:
        tuple: A tuple containing:
            - float: The value of the Max-Cut.
            - tuple: A tuple of two sets representing the partition of nodes 
                     that achieves the Max-Cut.
    """
    nodes = list(graph.nodes)
    max_cut_value = -1
    best_partition = (set(), set())

    for i in range(1, 2**len(nodes) - 1):
        binary_representation = bin(i)[2:].zfill(len(nodes))
        group1 = {
            nodes[j]
            for j, bit in enumerate(binary_representation)
            if bit == '1'
        }
        group2 = set(nodes) - group1

        cut_value = 0
        for u, v, data in graph.edges(data=True):
            if (u in group1 and v in group2) or (u in group2 and v in group1):
                cut_value += data.get(
                    'weight', 1)  # Use 1 as default weight if not specified

        if cut_value > max_cut_value:
            max_cut_value = cut_value
            best_partition = (group1, group2)

        # convert to binary representation
        binary_vector_1 = [0] * len(nodes)
        for node in best_partition[0]:
            binary_vector_1[node] = 1

        binary_vector_2 = [0] * len(nodes)
        for node in best_partition[1]:
            binary_vector_2[node] = 1

        binary_vector = (''.join(str(bit) for bit in binary_vector_1),
                         ''.join(str(bit) for bit in binary_vector_2))

    return (-1 * max_cut_value), best_partition, binary_vector


def simulated_annealing_maxcut(graph,
                               initial_temp=1000,
                               cooling_rate=0.95,
                               iterations=1000):
    nodes = list(graph.nodes)
    current_solution = {node: random.choice([0, 1]) for node in nodes}

    def cut_value(solution):
        return sum(data['weight']
                   for u, v, data in graph.edges(data=True)
                   if solution[u] != solution[v])

    current_value = cut_value(current_solution)
    best_solution = current_solution.copy()
    best_value = current_value
    temp = initial_temp

    for _ in range(iterations):
        node = random.choice(nodes)
        new_solution = current_solution.copy()
        new_solution[node] = 1 - new_solution[node]  # flip side
        new_value = cut_value(new_solution)

        delta = new_value - current_value
        if delta > 0 or random.random() < np.exp(delta / temp):
            current_solution = new_solution
            current_value = new_value
            if new_value > best_value:
                best_solution = new_solution
                best_value = new_value

        temp *= cooling_rate

    set1 = [node for node in best_solution if best_solution[node] == 0]
    set2 = [node for node in best_solution if best_solution[node] == 1]

    binary_vector_1 = [0] * len(nodes)
    for node in set1:
        binary_vector_1[node] = 1

    binary_vector_2 = [0] * len(nodes)
    for node in set2:
        binary_vector_2[node] = 1

    binary_vector = (''.join(str(bit) for bit in binary_vector_1),
                     ''.join(str(bit) for bit in binary_vector_2))

    return (set(set1), set(set2)), (-1 * best_value), binary_vector


def one_exchange(graph):

    curr_cut_size, partition = nx.approximation.one_exchange(graph,
                                                             weight='weight')
    # convert to binary representation
    binary_vector_1 = [0] * len(graph.nodes)
    for node in partition[0]:
        binary_vector_1[node] = 1
    binary_vector_2 = [0] * len(graph.nodes)
    for node in partition[1]:
        binary_vector_2[node] = 1
    binary_vector = (''.join(str(bit) for bit in binary_vector_1),
                     ''.join(str(bit) for bit in binary_vector_2))

    return (curr_cut_size * -1), partition, binary_vector
