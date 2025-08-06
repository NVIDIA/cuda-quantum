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

#from karateclub.feathergraph import FeatherGraph
from qaoa_gpt_src.custom_feather import CustomFeatherGraph as FeatherGraph
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import networkx as nx
import numpy as np
from collections import Counter
import random
import argparse
from joblib import Parallel, delayed
from itertools import islice
from networkx import convert_node_labels_to_integers
import ast

tqdm.pandas()

def open_df_from_res_csv(fname):
    #df_list = []
    try:
        cur_df = pd.read_csv(fname)
        cur_df['worker_id'] = fname.stem
        #df_list.append(cur_df)
    except Exception as e:
        print(f'{e} (file: {fname})')
        cur_df = None
    return cur_df

def generate_tokens(results_fpath_str: str, save_path_str: str, config_path_temp:str, n_nodes: int, rounding_digits = 2, min_block_size = 128,
                   max_block_size = 256, val_frac = 0.1, test_frac = 0.1, max_abs_param_val=10,
                   perform_coef_mod_range = True, apply_sliding_window = True, 
                   n_workers = 1, skip_only_qaoa_circ = False, 
                   allowed_graph_generators_list = ['all'],approx_ratio_thr= 0.97, debug_limit = 0, target_val_size = 10, verbose = True):
    
    """
        Args:
        results_fpath_str. Path to the directory with ADAPT-QAOA results. Should contain 'res' and 'graphs' subdirectories.
        save_path_str. Path to the directory where the generated tokens will be saved.
        config_path_temp: path to the template config file for training the model.
        n_nodes. Number of nodes in the graphs.
        rounding_digits. Number of digits to round the coefficients to.
        min_block_size. Minimum size of the sliding window for tokenization (min sequence length in sliding window).
        max_block_size. Maximum size of the sliding window for tokenization (nanoGPT block size).
        val_frac. Fraction of the data to be used for validation.
        test_frac. Fraction of the data to be used for testing.
        max_abs_param_val. Maximum absolute value of the coefficients (gamma and beta params).
        perform_coef_mod_range. Whether to perform coefficient modulation to the range [-max_abs_param_val, max_abs_param_val].
                                (Wrap beta to [0; pi] range; true (default))
        apply_sliding_window. Whether to apply sliding window for tokenization. (Apply sliding window to generate training samples)
        n_workers. Number of workers to use for parallel processing.
        skip_only_qaoa_circ. Whether to skip circuits that only use QAOA mixer (sum_i X(i)).
        allowed_graph_generators_list. List of allowed graph generators to filter the results.
                     Default: all. Should be separated with ; . Allowed values: erdos_renyi;barabasi_albert;watts_strogatz;random_regular;bipartite
        approx_ratio_thr. Threshold for the approximation ratio to filter the results.
        debug_limit. Limit the number of results to process for debugging purposes.
        tragtet_val_size. control the target size of the validation set, 
        regardless of the overall dataset size or class distribution.
        verbose. Whether to print verbose output.
        
        Returns:
        None. The function saves the generated tokens and metadata to the specified directory.
    
    """
    
    results_fpath_list = [Path(el) for el in results_fpath_str.split(';')]
    save_path = Path(save_path_str)

    for results_fpath in results_fpath_list:
        assert results_fpath.exists() and results_fpath.is_dir(), "Results path is invalid."
    
    if verbose: print(f"Results paths: {results_fpath_list}")
    
    ##########################################
    ## ADAPT-QAOA results
    ###########################################
    if verbose: print("Opening ADAPT-QAOA results...")
    df_list = []
    df_list_all = []
    
    for cur_dataset_res_path in results_fpath_list:
        cur_dataset_res_flist = sorted(cur_dataset_res_path.joinpath('res').glob('*.csv'))
        if debug_limit:
            cur_dataset_res_flist = cur_dataset_res_flist[:debug_limit]
        # for fname in tqdm(cur_dataset_res_flist, desc='Opening ADAPT results'):
        #     try:
        #         cur_df = pd.read_csv(fname)
        #         cur_df['worker_id'] = fname.stem
        #         df_list.append(cur_df)
        #     except Exception as e:
        #         print(f'{e} (file: {fname})')
        df_list = Parallel(n_jobs=n_workers)(
            delayed(open_df_from_res_csv)(fname) for fname in tqdm(cur_dataset_res_flist, desc=f'Opening ADAPT results ({cur_dataset_res_path.stem})')
        )
        df_list_all += df_list
    df_list  = [df for df in df_list_all if df is not None]
    
    if verbose: print("df_list len:", len(df_list))

    full_run_df = pd.concat(df_list)
    full_run_df['prefix'] = full_run_df['worker_id'].apply(
        lambda x: x[:-15])
    
    ####################################
    ## Graphs
    ####################################
    if verbose: print("Opening graphs results...")
    df_list = []
    df_list_all = []
    for cur_dataset_res_path in results_fpath_list:
        cur_dataset_res_flist = sorted(cur_dataset_res_path.joinpath('graphs').glob('*.csv'))
        if debug_limit:
            cur_dataset_res_flist = cur_dataset_res_flist[:debug_limit]
        # for fname in tqdm(cur_dataset_res_flist, desc='Opening graphs'):
        #     cur_df = pd.read_csv(fname)
        #     cur_df['worker_id'] = fname.stem
        #     df_list.append(cur_df)
        df_list = Parallel(n_jobs=n_workers)(
            delayed(open_df_from_res_csv)(fname) for fname in tqdm(cur_dataset_res_flist, desc=f'Opening graphs ({cur_dataset_res_path.stem})')
        )
        df_list_all += df_list
        
        for df in df_list:
            if df is not None:
                if 'g_method' not in df.columns:
                    #print("Graphs were generated with older version of ADAPT-GPT preprocessor. Most likely, they are ER.")
                    df['g_method'] = "erdos_renyi"
    df_list  = [df for df in df_list_all if df is not None]
    if verbose: print("df_list len:", len(df_list))


    full_run_graphs_df = pd.concat(df_list)
    full_run_graphs_df['edgelist_list'] = (
        full_run_graphs_df['edgelist_json'].progress_apply(
            lambda x: json.loads(x)
        )
    )
    full_run_graphs_df['edgelist_list_len'] = (
        full_run_graphs_df['edgelist_list'].progress_apply(
            lambda x: len(x)
        )
    )
    full_run_graphs_df['num_connected_comp'] = full_run_graphs_df['edgelist_list'].progress_apply(
        lambda x: len(
            list(
                nx.connected_components(
                    nx.Graph([edge[:2] for edge in x])
                )
            )
        )
    )
    full_run_graphs_df['prefix'] = full_run_graphs_df['worker_id'].apply(
        lambda x: x[:-14]
    )
    if verbose: print("Graphs count:")
    if verbose: print(full_run_graphs_df['g_method'].value_counts())

    ###################################################
    ## Merge adapt-qaoa results and graphs  results
    ###################################################
    if verbose: print("Merging ADAPT-QAOA results and graphs results...")
    
    if verbose:
        print(f"Number of rows in full_run_df: {len(full_run_df)}")
        print("Columns in full_run_df:", full_run_df.columns)
        print("Sample rows from full_run_df:")
        print(full_run_df.head())
    
    
    combined_res_df = pd.merge(
        left = full_run_df, right = full_run_graphs_df,
        left_on=['prefix', 'graph_num'], right_on=['prefix', 'graph_num']
    )
    
    if verbose:
        print("Unique prefixes in full_run_df:", full_run_df['prefix'].unique())
        print("Unique prefixes in full_run_graphs_df:", full_run_graphs_df['prefix'].unique())

        print("Unique graph_num in full_run_df:", full_run_df['graph_num'].unique())
        print("Unique graph_num in full_run_graphs_df:", full_run_graphs_df['graph_num'].unique())
    
    
    ###############################################
    # Add derived columns
    ###############################################
    
    # convert energy_list from string to list
    combined_res_df["energy_list"] = combined_res_df["energy_list"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    
    combined_res_df["n_layers"] = combined_res_df["energy_list"].apply(len)
    
    combined_res_df['graph_id'] = (
      combined_res_df['prefix']
    + '_^_'
    + combined_res_df['graph_num'].astype(str)
    )

    # convert mixer_pool_index from string to list of ints
    combined_res_df['mixer_pool_index'] = combined_res_df['mixer_pool_index'].apply(
    lambda x: [int(i) for i in ast.literal_eval(x)] if isinstance(x, str) else x
    )
    
    
    combined_res_df['only_qaoa_circ'] = combined_res_df['mixer_pool_index'].progress_apply(
        lambda x: all(e == n_nodes+1 for e in x)
    )
    
    
    # Filter by allowed graph generators if needed
    if allowed_graph_generators_list != ['all']:
        if verbose: print(f"Filtering graphs based on allowed generators: {allowed_graph_generators_list}")
        if verbose: print(f"N circuits before: {len(combined_res_df)}")
        combined_res_df = combined_res_df[
            combined_res_df['g_method'].isin(allowed_graph_generators_list)
        ]
        if verbose: print(f"N circuits after: {len(combined_res_df)}")
    
    if verbose: print(combined_res_df['g_method'].value_counts())
    
    
    ############################################
    # Graph embedding
    ############################################
    if verbose: print("Applying FEATHER graph embedding...")

    combined_unique_graphs_df = (
        combined_res_df[['graph_id', 'edgelist_json']]
            .drop_duplicates()
    )


    def create_weighted_graph_nx(w_elist):
        G = nx.Graph()
        G.add_weighted_edges_from(w_elist)
        return G

    combined_unique_graphs_df['edgelist_py_list'] = combined_unique_graphs_df['edgelist_json'].progress_apply(
        lambda x: [
            (e[0]-1, e[1]-1, e[2]) for e in json.loads(x)
            #(e[0]-1, e[1]-1) for e in x
        ]
    )

    #combined_unique_graphs_df['graph_nx'] = (
    #    combined_unique_graphs_df['edgelist_py_list']
    #        .progress_apply(lambda x: create_weighted_graph_nx(x))
    #)
    
    combined_unique_graphs_df['graph_nx'] = (
    combined_unique_graphs_df['edgelist_py_list']
        .progress_apply(lambda x: convert_node_labels_to_integers(create_weighted_graph_nx(x)))
    )

    combined_unique_graphs_w_idx_df = combined_unique_graphs_df.set_index('graph_id')
    graphs_nx_dict = combined_unique_graphs_w_idx_df['graph_nx'].to_dict()
    graphs_nx_filt_dict = dict(
        [(name, g) for name, g in tqdm(graphs_nx_dict.items()) if g.number_of_nodes() == n_nodes]
    )
    graphs_nx_filt_names = list(graphs_nx_filt_dict.keys())
    graphs_nx_filt_list = list(graphs_nx_filt_dict.values())

    emb_graph_idx_to_id_dict = {k:v for k,v in enumerate(graphs_nx_filt_names)}
    emb_graph_id_to_idx_dict = {v:k for k,v in enumerate(graphs_nx_filt_names)}

    #def get_feather_emb(g_list):
    #    feather_model = FeatherGraph()
    #    feather_model.fit(graphs=g_list)
    #    return feather_model.get_embedding()
    def get_feather_emb(g_list):
        # Using our custom wrapper with the original FEATHER implementation
        feather_model = FeatherGraph(
            order=5,
            eval_points=25,
            theta_max=2.5,
            seed=42,
            pooling="mean"
        )
        feather_model.fit(graphs=g_list)
        return feather_model.get_embedding()
    

    def split_list(lst, n):
        it = iter(lst)
        return [list(islice(it, i)) for i in [len(lst) // n + (1 if x < len(lst) % n else 0) for x in range(n)]]

    def embed_nx_w_feather_parallel(graphs_list, n_workers=2):
        graphs_chunked_list = split_list(graphs_list, n_workers)
        
        #graphs_chunked_list=[graphs_list]
        
        emb_np_list = Parallel(n_jobs=n_workers)(
            delayed(get_feather_emb)(g_chunk) for g_chunk in graphs_chunked_list
        )
        
        return np.vstack(emb_np_list)

    feather_par_emb = embed_nx_w_feather_parallel(graphs_nx_filt_list[:], n_workers=n_workers)
    feather_par_emb = feather_par_emb.round(rounding_digits)
    #print(f"Graph embedding shape: {feather_par_emb.shape} (n_graphs × dimension)")
    #print(f"Graph embedding dimension: {feather_par_emb.shape[1]}")

    combined_res_df['has_emb'] = combined_res_df['graph_id'].apply(
        lambda x: True if x in emb_graph_id_to_idx_dict else False
    )
    
    #######################################
    # Filtering 
    #######################################
    
    if verbose: print("Filtering results...")
    if verbose: print(f"Number of rows in combined_res_df before filtering: {len(combined_res_df)}")

    # Convert gamma_coef from string to list of floats
    combined_res_df['gamma_coef'] = combined_res_df['gamma_coef'].apply(
    lambda x: [float(coef) for coef in ast.literal_eval(x)] if isinstance(x, str) else x
    )
    
    # convert approx_ratio from string to float
    combined_res_df['approx_ratio'] = pd.to_numeric(combined_res_df['approx_ratio'], errors='coerce')
    
    combined_res_filt_df = combined_res_df[
        # (
        #     combined_res_df['beta_coef'].apply(
        #         lambda x: all([abs(coef) < max_abs_param_val for coef in x])
        #     )
        # )
        # &
        (
            combined_res_df['gamma_coef'].apply(
                lambda x: all([abs(coef) < max_abs_param_val for coef in x])
            )
        )
        &
        (
            combined_res_df['approx_ratio'] > approx_ratio_thr
        )
    ].copy()
    

    if skip_only_qaoa_circ:
        if verbose: print("Filtering out only QAOA circuits...")
        n_only_qaoa_circ = combined_res_filt_df['only_qaoa_circ'].sum()
        if verbose: print(f"Removing {n_only_qaoa_circ} out of total {len(combined_res_filt_df)}")
        combined_res_filt_df = combined_res_filt_df[
            combined_res_filt_df['only_qaoa_circ'] == False
        ]
    
    
    # Convert beta_coef and gamma_coef from strings to lists of floats
    
    combined_res_filt_df['beta_coef'] = combined_res_filt_df['beta_coef'].apply(
        lambda x: [float(coef) for coef in ast.literal_eval(x)] if isinstance(x, str) else x
    )
    
    combined_res_filt_df['gamma_coef'] = combined_res_filt_df['gamma_coef'].apply(
        lambda x: [float(coef) for coef in ast.literal_eval(x)] if isinstance(x, str) else x
    )
    
    combined_res_filt_df['mixer_pool_index'] = combined_res_filt_df['mixer_pool_index'].apply(
        lambda x: [int(idx) for idx in ast.literal_eval(x)] if isinstance(x, str) else x
    )

    if verbose: print(f"Number of rows in combined_res_filt_df after filtering: {len(combined_res_filt_df)}")
    

    ########################################
    # Tokenization
    ########################################
    
    if verbose: print("Tokenizing...")
    tokens_list = []

    ## Special symbols
    special_symbols_list = [
        'pad',
        'bos',
        'eos',
        'new_layer_p',
        'end_of_graph'
    ]
    tokens_list += special_symbols_list
    
    ## Edges
    all_edges_list = []
    for g in combined_res_filt_df['edgelist_list']:
        for e in g:
            all_edges_list.append(tuple(e[:2]))
    all_edges_set = set(all_edges_list)

    if verbose: print(f"\tTotal tokens for edges: {len(all_edges_set)}")
    tokens_list += list(all_edges_set)
    
    ## Coeffs

    n_steps = int((max_abs_param_val * 2 / (10 ** -rounding_digits) ) + 1)

    all_coefs_round_set = set(
        [
            round(coef, rounding_digits) for coef in np.linspace(start=-max_abs_param_val, stop=max_abs_param_val, num=n_steps).tolist()
        ]
    )
    len(all_coefs_round_set)
    tokens_list += list(all_coefs_round_set)

    if verbose: print(f"\tTotal tokens for coefs: {len(all_coefs_round_set)}")


    ## Operator pool
    
    ops_list = []
    for l in combined_res_filt_df['mixer_pool_index']:
        
        ops_list += l

    ops_list = list(set(ops_list))
    if verbose: print(f"\tTotal tokens for operator pool: {len(ops_list)}")
    tokens_list += ops_list

    ######################################
    ## Tokenization
    #######################################
    
    int_idx_to_token_dict = dict(enumerate(tokens_list))
    token_to_int_idx_dict = {v:k for k,v in int_idx_to_token_dict.items()}

    vocab_size = len(int_idx_to_token_dict)
    if verbose: print(f"\tTotal tokens in vocab: {vocab_size}")
    
    """
    We know that beta coefficients can't exceed pi in QAOA formulation, 
    but when we were using ADAPT.jl it happens occasionally due to the optimization process. 
    Since our GPT model requires that all numerical values have distinct tokens, 
    we are limited to what value range we can represent, so we apply this function to effectively 
    return beta values into their canonical range. Beta values are pi-periodical, 
    so if a beta value exceeds the range, we simply return it back to pi-range.
    We also experimented with gamma parameters, but it turned out to be more complicated, 
    and we did not get consistent results, so we don't do gamma modulation at this time.
    
    """
    
    def arth_mod(a,b):
        result = a % b
        return result if a >= 0 else result - b
    
    def tokenize_row(row, coef_mod = True):       

        tokens_seq_list = ['bos']

        for edge in row['edgelist_list']:
            edge_tuple = tuple(edge[:2])
            edge_weight = edge[2]
            tokens_seq_list.append(edge_tuple)
            tokens_seq_list.append(edge_weight)

        tokens_seq_list.append('end_of_graph')

        for p in range(row['n_layers']):
            tokens_seq_list.append('new_layer_p')
            tokens_seq_list.append(row['mixer_pool_index'][p])

            cur_beta = row['beta_coef'][p]
                
            if coef_mod:
                # cur_beta = cur_beta % (np.pi)
                cur_beta = arth_mod(cur_beta, np.pi)
            if cur_beta > -max_abs_param_val and cur_beta < max_abs_param_val:
                cur_beta_round = round(cur_beta, rounding_digits)
                tokens_seq_list.append(cur_beta_round)
            else:
                return None

            cur_gamma = row['gamma_coef'][p]
            
            # if coef_mod:
            #     cur_gamma = cur_gamma % (2*np.pi)
            if cur_gamma > -max_abs_param_val and cur_gamma < max_abs_param_val:
                cur_gamma_round = round(cur_gamma, rounding_digits)
                tokens_seq_list.append(cur_gamma_round)
            else:
                return None
        
        tokens_seq_list.append('eos')
        
        return tokens_seq_list
    
    combined_res_filt_df[f'token_seq_round_d{rounding_digits}'] = combined_res_filt_df.progress_apply(
    lambda x: tokenize_row(x, coef_mod=perform_coef_mod_range),
    axis=1,
    )
    
    
    combined_res_tok_df = combined_res_filt_df.dropna()
    combined_res_tok_df[f'token_int_seq_round_d{rounding_digits}'] = (
        combined_res_tok_df[f'token_seq_round_d{rounding_digits}'].progress_apply(
            lambda x: [token_to_int_idx_dict[token] for token in x]
        )
    )
    
    ########################################################
    # Generating training split for nanoGPT
    ########################################################
    if verbose: print("Generating training, validation and test splits...")

    n = len(combined_res_tok_df)

    combined_res_tok_shf_df = (
        combined_res_tok_df
            .sample(frac=1)
            .reset_index(drop=True)
    )

    if verbose:
        print(f"combined_res_df shape: {combined_res_df.shape}")
        print(f"combined_res_tok_df shape: {combined_res_tok_df.shape}")
        print(f"combined_res_tok_shf_df shape: {combined_res_tok_shf_df.shape}")

    graph_ids = combined_res_tok_shf_df['graph_id'].drop_duplicates().to_list()

    # Compute the number of graphs for each set
    n_total = len(graph_ids)
    n_test = int(n_total * test_frac)  # Define test_frac for the size of the test set
    n_val = int(n_total * val_frac)  # val_frac defines the validation set size
    n_train = n_total - n_test - n_val  # Remaining will be the training set

    # Split into train, val, and test sets
    train_graph_ids_set = set(graph_ids[:n_train])
    val_graph_ids_set = set(graph_ids[n_train:n_train + n_val])
    test_graph_ids_set = set(graph_ids[n_train + n_val:])

    assert len(train_graph_ids_set.intersection(val_graph_ids_set)) == 0
    assert len(train_graph_ids_set.intersection(test_graph_ids_set)) == 0
    assert len(val_graph_ids_set.intersection(test_graph_ids_set)) == 0
    
    def pad_with_zeros(seq, target_len):
        pad_len = target_len - len(seq)
        if pad_len > 0:
            padded_seq = seq + [0] * pad_len
        else:
            padded_seq = seq
            
        if len(padded_seq) !=max_block_size:
            if verbose: print(f"padded_seq len: {len(padded_seq)}")
        return padded_seq

    def sliding_window(numbers, min_block_size, max_block_size):
        
        if min_block_size != max_block_size:
            block_size = random.randint(min_block_size, max_block_size)
        else:
            block_size = min_block_size

        if block_size >= len(numbers):
            window = numbers[:-1]
            window_shifted = numbers[1:]   
            return [
                [
                    pad_with_zeros(window, target_len=max_block_size),
                    pad_with_zeros(window_shifted, target_len=max_block_size)
                ]
            ]
        
        result_xy_list = []
        result = []
        for i in range(0, len(numbers) - block_size + 1):
            window = numbers[i:i + block_size]
            result.append(window)
            
        for x, y in zip(result, result[1:]):
            result_xy_list.append(
                [
                    pad_with_zeros(x, target_len=max_block_size),
                    pad_with_zeros(y, target_len=max_block_size)
                ]
            )
        
        return result_xy_list
    
    # Assign the 'label' column based on the split
    combined_res_tok_shf_df['label'] = 'train'
    combined_res_tok_shf_df.loc[combined_res_tok_shf_df['graph_id'].isin(val_graph_ids_set), 'label'] = 'val'
    combined_res_tok_shf_df.loc[combined_res_tok_shf_df['graph_id'].isin(test_graph_ids_set), 'label'] = 'test'

    if apply_sliding_window:
        print('Applying sliding window...')
        combined_res_tok_shf_df[f'token_int_seq_round_d{rounding_digits}_sw'] = combined_res_tok_shf_df[f'token_int_seq_round_d{rounding_digits}'].progress_apply(
            lambda x: sliding_window(
                x,
                min_block_size=min_block_size,
                max_block_size=max_block_size
            )
        )
        
    train_data = combined_res_tok_shf_df[
        combined_res_tok_shf_df['label'] == 'train'
    ]
    val_data = combined_res_tok_shf_df[
        combined_res_tok_shf_df['label'] == 'val'
    ]
    test_data = combined_res_tok_shf_df[
        combined_res_tok_shf_df['label'] == 'test'
    ]

    print(f"\tNumber of training samples: {len(train_data)}, val samples: {len(val_data)}, test samples: {len(test_data)}")


    if apply_sliding_window:

        train_data_conc = []
        train_data_graph_idx_list = []
        for cur_graph_id, l in zip(
            train_data['graph_id'],
            train_data[f'token_int_seq_round_d{rounding_digits}_sw']
        ):
            if cur_graph_id in emb_graph_id_to_idx_dict:
                train_data_conc += l
                train_data_graph_idx_list += [emb_graph_id_to_idx_dict[cur_graph_id]] * len(l)
        train_data_conc_np = np.array(train_data_conc, dtype=np.uint16)

        val_data_conc = []
        val_data_graph_idx_list = []
        for cur_graph_id, l in zip(
            val_data['graph_id'],
            val_data[f'token_int_seq_round_d{rounding_digits}_sw']
        ):
            if cur_graph_id in emb_graph_id_to_idx_dict:
                val_data_conc += l
                val_data_graph_idx_list += [emb_graph_id_to_idx_dict[cur_graph_id]] * len(l)
        val_data_conc_np = np.array(val_data_conc, dtype=np.uint16)

        test_data_conc = []
        test_data_graph_idx_list = []
        for cur_graph_id, l in zip(
            test_data['graph_id'],
            test_data[f'token_int_seq_round_d{rounding_digits}_sw']
        ):
            if cur_graph_id in emb_graph_id_to_idx_dict:
                test_data_conc += l
                test_data_graph_idx_list += [emb_graph_id_to_idx_dict[cur_graph_id]] * len(l)
        test_data_conc_np = np.array(test_data_conc, dtype=np.uint16)


        #print(f"\tTrain has {len(train_data_conc_np):,} samples")
        #print(f"\tVal has {len(val_data_conc_np):,} samples")
        #print(f"\tTest has {len(test_data_conc_np):,} samples")
    
    #######################################
    # Saving
    #######################################
    if verbose: print("Saving data...")

    save_path.mkdir(parents=True, exist_ok=True)

    if apply_sliding_window:
        
        np.save(
            save_path.joinpath('train.npy'),
            train_data_conc_np
        )
        np.save(
            save_path.joinpath('val.npy'),
            val_data_conc_np
        )
        np.save(
            save_path.joinpath('test.npy'),
            test_data_conc_np
        )

    combined_res_df.to_pickle(
        save_path.joinpath('combined_res_df.pkl')
    )

    combined_res_tok_shf_df.to_pickle(
        save_path.joinpath('combined_res_tok_shf_df.pkl')
    )

    
    sample_size_per_w_bucket = int(
        target_val_size
        / len(combined_res_df['edgelist_list_len'].drop_duplicates())
    )
    
    '''
    val_data_sampled = (
        val_data[
            (~val_data['token_seq_round_d2'].isna())
        ]
        .groupby('edgelist_list_len', group_keys=False).apply(
            lambda x: x.sample(sample_size_per_w_bucket) if len(x) > sample_size_per_w_bucket else x
        )
        .reset_index(drop=True)
    )
    '''
    
    val_data_sampled = (
    val_data[
        (~val_data[f'token_seq_round_d{rounding_digits}'].isna())
    ]
    .groupby('edgelist_list_len', group_keys=False)
    .filter(lambda group: len(group) > sample_size_per_w_bucket)
    .reset_index(drop=True)
    )
    
    #print(val_data_sampled.columns)
    #print('Checking val_data_sampled...')
    #print(val_data_sampled.groupby('edgelist_list_len').groups.keys())
    #print(val_data_sampled.groupby('edgelist_list_len').apply(lambda group: group.drop(columns=['edgelist_list_len']).sample(sample_size_per_w_bucket)))

    val_data_sampled.to_pickle(
        save_path.joinpath('combined_res_tok_shf_val_df.pkl')
    )

    emb_size = feather_par_emb.shape[1]
    np.save(
        save_path.joinpath(f'feather_emb_d{emb_size}.npy'),
        feather_par_emb
    )

    meta = {
        'vocab_size': vocab_size,
        'itos': int_idx_to_token_dict,
        'stoi': token_to_int_idx_dict,
        'train_data_graph_idx_list': train_data_graph_idx_list,
        'val_data_graph_idx_list': val_data_graph_idx_list,
        'test_data_graph_idx_list': test_data_graph_idx_list,
        'emb_graph_id_to_idx_dict': emb_graph_id_to_idx_dict,
        'emb_graph_idx_to_id_dict': emb_graph_idx_to_id_dict,
    }

    pd.to_pickle(
        meta,
        save_path.joinpath('meta.pkl')
    )

    with open(f'{config_path_temp}/train_adapt_gpt_config_template.py') as f:
        config_template_str = f.read()

    pool_type = combined_res_df['pool_type'].iloc[0]

    dataset_name = save_path.stem
    config_to_save_str = config_template_str.format(
        out_dir = f'out-{dataset_name}',
        dataset = dataset_name,
        block_size = max_block_size,
        use_graph_emb = "True",
        pool_type = pool_type,
        n_nodes = n_nodes,
        token_seq_round = f'token_seq_round_d{rounding_digits}',
    )

    with open(save_path.joinpath('train_adapt_gpt_config.py'), 'w') as f:
        f.write(config_to_save_str)

    if verbose: print(f"Data is saved to: {str(save_path.absolute())}")
    if verbose: print("Done!")


    return