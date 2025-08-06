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

import os
import json
import socket
import time
import numpy as np
import pandas as pd
from datetime import datetime

import cudaq
from qaoa_gpt_src.adapt_qaoa import adapt_qaoa_run
from qaoa_gpt_src.hamiltonian_graph import term_coefficients, term_words, max_cut_ham
from qaoa_gpt_src.graph_functions import generate_random_graph, graph_to_edgelist, edgelist_to_graph, graph_to_adj_m, add_rand_weights_to_graph, norm_elist_weights

from qaoa_gpt_src.max_cut_classical_sol import brute_force_max_cut, one_exchange, simulated_annealing_maxcut

# Set the target to NVIDIA GPU
#######################################################
# Set the target to NVIDIA GPU with double precision support
# cudaq.set_target("nvidia")  # Uncomment this line if you want to use single precision
# For double precision, use the following line:
cudaq.set_target("nvidia", option="fp64")
#######################################################


def ensure_dirs(output_dir):
    for sub in ['hams', 'res', 'graphs', 'traces']:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)


#######################################################


# Scale the weights of the edges in the edge list by a given coefficient
def scale_elist_weights(e_list, coef):
    """
    Scales the weights of each edge in the edge list by the given coefficient.

    Args:
        e_list (list of tuples): Each tuple is (node1, node2, weight).
        coef (float): The scaling coefficient.

    Returns:
        list of tuples: New edge list with scaled weights.
    """
    return [(node1, node2, weight * coef) for (node1, node2, weight) in e_list]


######################################################
def generate_data_max_cut(output_dir='adapt_results',
                          graphs_number=1,
                          graphs_input_json="N/A",
                          n_nodes=8,
                          weighted=True,
                          use_negative_weights=False,
                          use_brute_force=True,
                          use_simulated_annealing=True,
                          use_one_exchange=True,
                          op_pool='all_pool',
                          init_gamma: list[float] = [0.01],
                          scaling_coef=1.0,
                          norm_weights=False,
                          norm_coef=1.0,
                          trials_per_graph=1,
                          optimizer='BFGS',
                          approx_ratio=0.97,
                          max_iter=10,
                          norm_threshold=1e-3,
                          energy_threshold=1e-9,
                          multi_gamma=False,
                          verbose=True):
    """
    Generates data for ADAPT-QAOA on the Max-Cut problem.
    Args:
        output_dir (str): Directory to save the results.
        graphs_number (int): Number of graphs to generate or process.
        graphs_input_json (str): Path to a JSON file containing graph data. If "N/A", generates random graphs.
        n_nodes (int): Number of nodes in each graph.
        weighted (bool): Whether to use weighted edges.
        use_negative_weights (bool): Whether to allow negative weights in the graph.
        use_brute_force (bool): Whether to compute the brute force solution.
        use_simulated_annealing (bool): Whether to compute the simulated annealing solution.
        use_one_exchange (bool): Whether to compute the one exchange solution.
        op_pool (str): Type of operator pool to use ('all_pool' or 'qaoa_mixer' or 'qaoa_single_x' or 'qaoa_double_ops').
        init_gamma (list): Initial gamma values for ADAPT-QAOA trials.
        scaling_coef (float): Coefficient to scale edge weights.
        norm_weights (bool): Whether to normalize edge weights.
        norm_coef (float): Coefficient for normalizing edge weights if norm_weights is True.
        trials_per_graph (int): Number of trials per graph for ADAPT-QAOA.
        optimizer (str): Optimizer to use for ADAPT-QAOA ('BFGS', 'L-BFGS-B', 'COBYLA').
        approx_ratio (float): Approximation ratio threshold for early stopping.
        max_iter (int): Maximum number of iterations for the adapt-qaoa iteration.
        norm_threshold (float): Threshold for gradients norm in ADAPT-QAOA for early stopping.
        energy_threshold (float): Threshold for energy convergence in ADAPT-QAOA.
        multi_gamma (bool): Whether to run multiple trials with different initial gamma values to generate multiple validated circuit. 
        Useful if you want to check for more validated circuits after approx ratio achieved with one gamma.
        verbose (bool): Whether to print detailed logs during execution.
        
    Returns:
        None: The function saves results to the specified output directory.
    
    """

    ensure_dirs(output_dir)

    pid = os.getpid()
    hostname = socket.gethostname()
    ts_string = datetime.now().strftime("%y-%m-%d__%H_%M")

    results_df = pd.DataFrame()
    hams_df = pd.DataFrame()
    graphs_df = pd.DataFrame(
        columns=['graph_num', 'g_method', 'edgelist_json', 'H_frob_norm'])
    traces_df = pd.DataFrame()

    # Load graphs from JSON if provided
    if graphs_input_json != "N/A":
        with open(graphs_input_json, 'r') as f:
            json_graphs_dict = json.load(f)
        graphs_number = len(json_graphs_dict)
        graph_names_list = list(json_graphs_dict.keys())
    else:
        graphs_number = graphs_number

    # Loop through the number of graphs to generate or process
    if verbose:
        print(f"Generating or processing {graphs_number} graphs...")

    graph_rows = []
    result_rows = []

    for graph_num in range(graphs_number):

        if graphs_input_json == "N/A":
            cur_graph_name = f"Graph_{graph_num+1}"
            g_unweighted, g_method = generate_random_graph(
                n_nodes, methods=["erdos_renyi"])
            if weighted:
                g = add_rand_weights_to_graph(g_unweighted,
                                              neg_weights=use_negative_weights)
            else:
                g = g_unweighted
        else:
            cur_graph_name = graph_names_list[graph_num]
            cur_graph_elist = json_graphs_dict[cur_graph_name]["elist"]
            n_nodes = json_graphs_dict[cur_graph_name]["n_nodes"]
            g = edgelist_to_graph(cur_graph_elist, num_vertices=n_nodes)
            g_method = "input_file"

        if verbose:
            print(f"Processing {cur_graph_name}...")
        if verbose:
            print(f"Graph method: {g_method}")
        if verbose:
            print(f"Graph edgelist: {graph_to_edgelist(g)}")

        e_list = graph_to_edgelist(g)

        # update e_list to change index to 1 based for tokenization later.
        e_list_mod = [
            (node1 + 1, node2 + 1, weight) for (node1, node2, weight) in e_list
        ]
        edgelist_json = json.dumps(e_list_mod)

        if scaling_coef != 1.0:
            e_list = scale_elist_weights(e_list, scaling_coef)

        if norm_weights:
            e_list, norm_coef = norm_elist_weights(e_list)

        ###############################################
        # Build up the problem hamiltonian

        spin_ham = max_cut_ham(e_list)
        #if verbose: print(f"Problem Hamiltonian: {spin_ham}")

        h_frob_norm = np.linalg.norm(spin_ham.to_matrix())
        if verbose:
            print(f"Frobenius norm of the Hamiltonian: {h_frob_norm}")

        # Store the graph data
        graph_rows.append({
            'graph_num': graph_num + 1,
            'g_method': g_method,
            'edgelist_json': edgelist_json,
            'H_frob_norm': h_frob_norm
        })

        # After the loop:
        graphs_df = pd.DataFrame(
            graph_rows,
            columns=['graph_num', 'g_method', 'edgelist_json', 'H_frob_norm'])

        ############################################
        # Classical solutions of max-cut problem
        if use_brute_force:
            # Brute Force
            brute_force_cut_value, partition, binary_vector = brute_force_max_cut(
                g)

        if use_simulated_annealing:
            # Simulated Annealing
            sa_partition, sa_cut_value, sa_binary_vector = simulated_annealing_maxcut(
                g)

        if use_one_exchange:
            # one_exchange
            one_exchange_cut_value, one_exchange_partition, one_exchange_binary_vector = one_exchange(
                g)
        ################################################

        # Quantum solutions of max-cut problem using ADAPT-QAOA
        if verbose:
            print(f"Preparing to run ADAPT-QAOA for graph {graph_num+1}...")

        for gamma in init_gamma:
            for trial_num in range(trials_per_graph):

                if verbose:
                    print(
                        f"Running ADAPT-QAOA for graph {graph_num+1}, trial {trial_num+1}..."
                    )
                if verbose:
                    print(f"Using initial gamma: {gamma}")

                # Run ADAPT-QAOA
                if verbose:
                    print("Running ADAPT-QAOA...")

                # Run the ADAPT-QAOA algorithm
                qubits_num = len(g.nodes)
                pool = op_pool
                g0 = gamma

                if use_simulated_annealing:
                    true_energy = sa_cut_value  # Use the simulated annealing cut value as the true energy
                    classical_cut = sa_binary_vector
                elif use_one_exchange:
                    true_energy = one_exchange_cut_value
                    classical_cut = one_exchange_binary_vector
                elif use_brute_force:
                    true_energy = brute_force_cut_value
                    classical_cut = binary_vector
                else:
                    true_energy = -999.0
                    classical_cut = "N/A"

                start_time = time.time()

                # Run the adapt_qaoa function
                adapt_qaoa_result = adapt_qaoa_run(
                    spin_ham,
                    qubits_num,
                    pool=pool,
                    gamma_0=g0,
                    norm_threshold=norm_threshold,
                    energy_threshold=energy_threshold,
                    approx_ratio=approx_ratio,
                    true_energy=true_energy,
                    optimizer=optimizer,
                    max_iter=max_iter,
                    verbose=verbose)

                end_time = time.time()
                elapsed_time = end_time - start_time

                if isinstance(adapt_qaoa_result, tuple):
                    adapt_qaoa_result = list(
                        adapt_qaoa_result)  # Convert to list
                    adapt_qaoa_result[2] = [
                        int(i) + 1 for i in adapt_qaoa_result[2]
                    ]  # Modify the third element indexes to be 1-based
                    adapt_qaoa_result = tuple(
                        adapt_qaoa_result
                    )  # Convert back to tuple (if required)
                else:
                    adapt_qaoa_result[2] = [
                        int(i) + 1 for i in adapt_qaoa_result[2]
                    ]  # Modify directly if not a tuple

                if verbose:
                    print(
                        f"ADAPT-QAOA completed in {elapsed_time:.2f} seconds.")
                    print('Energy list: ', adapt_qaoa_result[0])
                    print('Mixer pool as pauli word: ', adapt_qaoa_result[1])
                    print('Mixer pool as index: ', adapt_qaoa_result[2])
                    print('gamma list: ', adapt_qaoa_result[3])
                    print('beta list: ', adapt_qaoa_result[4])
                    print('Approx. ratio: ', adapt_qaoa_result[5])
                    print('Max cut: ', adapt_qaoa_result[6])
                    print('Number of layers: ', adapt_qaoa_result[7])
                    print('Optimizer success flag: ', adapt_qaoa_result[8])
                    print('\n')

                # Prepare the results for saving
                result_rows.append({
                    'method': 'ADAPT-QAOA',
                    'graph_name': cur_graph_name,
                    'graph_num': graph_num + 1,
                    'trial_num': trial_num + 1,
                    'n_nodes': n_nodes,
                    'init_gamma': gamma,
                    'optimizer': optimizer,
                    'pool_type': op_pool,
                    'edge_weight_scaling_coef': scaling_coef,
                    'edge_weight_norm_coef': norm_coef,
                    'energy_list': adapt_qaoa_result[0],
                    'true_energy': true_energy,
                    'mixer_pool_pauli_word': adapt_qaoa_result[1],
                    'mixer_pool_index': adapt_qaoa_result[2],
                    'gamma_coef': adapt_qaoa_result[3],
                    'beta_coef': adapt_qaoa_result[4],
                    'approx_ratio': adapt_qaoa_result[5],
                    'cut_adapt': adapt_qaoa_result[6],
                    'cut_classical': classical_cut,
                    'num_layers': adapt_qaoa_result[7],
                    'optimizer_success_flag': adapt_qaoa_result[8],
                    'elapsed_time': elapsed_time
                })

                results_df = pd.DataFrame(
                    result_rows,
                    columns=[
                        'method', 'graph_name', 'graph_num', 'trial_num',
                        'n_nodes', 'init_gamma', 'energy_list', 'true_energy',
                        'optimizer', 'pool_type', 'edge_weight_scaling_coef',
                        'edge_weight_norm_coef', 'mixer_pool_pauli_word',
                        'mixer_pool_index', 'gamma_coef', 'beta_coef',
                        'approx_ratio', 'cut_adapt', 'cut_classical',
                        'num_layers', 'optimizer_success_flag', 'elapsed_time'
                    ])
                # Early stopping: End of trial if approximation ratio is reached
                if adapt_qaoa_result[5] >= approx_ratio:
                    if verbose:
                        print(
                            f"Approximation ratio {adapt_qaoa_result[5]} reached, stopping early."
                        )
                    break
            # Early stopping: End of graph processing if approximation ratio is reached
            if adapt_qaoa_result[5] >= approx_ratio and not multi_gamma:
                if verbose:
                    print(
                        f"Approximation ratio {adapt_qaoa_result[5]} reached for graph {graph_num+1}, stopping further trials."
                    )
                # uncomment if you do not want to check for more validated circuits
                break

    #write results to files
    if verbose:
        print("Writing results to files...")

    # Save results DataFrame to CSV
    results_df.to_csv(os.path.join(output_dir, 'res',
                                   f'pid{pid}_{ts_string}_results.csv'),
                      index=False)

    # Save graphs DataFrame to CSV
    graphs_df.to_csv(os.path.join(output_dir, 'graphs',
                                  f'pid{pid}_{ts_string}_graphs.csv'),
                     index=False)

    if verbose:
        print("Data generation completed successfully.")

    return
