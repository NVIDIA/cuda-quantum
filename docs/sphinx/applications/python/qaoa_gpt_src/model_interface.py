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

import pickle
from contextlib import nullcontext
import torch
import tiktoken
from nanoGPT.model_pad_gemb import GPTConfig as GPTConfig_gemb
from nanoGPT.model_pad_gemb import GPT as GPT_gemb

from nanoGPT.model_pad import GPTConfig as GPTConfig_nogemb
from nanoGPT.model_pad import GPT as GPT_nogemb

import pandas as pd
import json
from tqdm import tqdm
import random
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from pathlib import Path

from qaoa_gpt_src.util import (
    generate_circ_from_df, 
    eval_adapt_gpt_circ_cudaq,
    prepare_model_input
)

class QAOA_GPT():
    def __init__(
        self,
        model_ckpt,
        config_file,
        data_dir,
        device, # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        n_nodes='infer',
        temp_folder='adapt_gpt_temp_data',
    ):
        
        config_fpath = Path(config_file)
        assert config_fpath.is_file()

        print(f"Loading config from: {config_fpath}")
        config_vars = {}
        with open(config_fpath) as f:
            exec(f.read(), config_vars)

        self.pool_type = config_vars['pool_type']
        self.use_graph_emb = config_vars['use_graph_emb']

        if 'n_nodes' not in config_vars:
            if n_nodes == 'infer':
                raise AttributeError(
                    """Number of nodes is not found in the provided config.
                    You need to supply it as an argument in AdaptGPT constructor:
                    AdaptGPT(..., n_nodes=<N>,...)
                    """
                )
            else:
                assert type(n_nodes) == int
                self.n_nodes = n_nodes
        else:
            self.n_nodes = config_vars['n_nodes']

        #self.out_dir = Path(out_dir)
        self.data_dir = Path(data_dir)
        self.model_ckpt = Path(model_ckpt)
        self.temp_folder = Path(temp_folder)
        
        self.seed = 1337
        self.init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
        self.device = device
        if self.device == 'cuda':
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = 'bfloat16'
        else:
            self.dtype = 'float16'
        
        self.compile = False # use PyTorch 2.0 to compile the model to be faster
        
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)

        self.meta = pd.read_pickle(f'{data_dir}/meta.pkl')

        if self.use_graph_emb:
            self.gptconfig = GPTConfig_gemb
            self.gpt = GPT_gemb
        else:
            self.gptconfig = GPTConfig_nogemb
            self.gpt = GPT_nogemb

        self.model = self.open_model(self.model_ckpt)
            
        return None

    def open_model(
        self,
        model_fpath,
    ):
        # init from a model saved in a specific directory
        out_path = Path(model_fpath)
        checkpoint = torch.load(out_path, map_location=self.device)
        gptconf = self.gptconfig(**checkpoint['model_args'])
        model = self.gpt(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        
        return model
    
    def generate_circ_from_nx(
        self,
        graphs_container,
        calculate_classic_maxcut=True,
        n_samples_per_batch=50, # max number of distinct graphs in a batch
        num_samples=5, # number of samples to draw
        max_new_tokens=150, # number of tokens generated in each sample
        temperature=0.1, # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        top_k=200, # retain only the top_k most likely tokens, clamp others to have 0 probability
    ):
        graphs_nx_df, feather_par_emb, emb_graph_id_to_idx_dict = prepare_model_input(
            graphs_container,
            n_nodes=self.n_nodes,
            calculate_classic_maxcut=calculate_classic_maxcut,
        )

        gc_df = generate_circ_from_df(
            graphs_nx_df,
            graph_emb_np=feather_par_emb,
            emb_graph_id_to_idx_dict=emb_graph_id_to_idx_dict,
            meta=self.meta,
            model=self.model,
            device=self.device,
            ctx=self.ctx,
            n_samples_per_batch=n_samples_per_batch,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            token_seq_col='token_seq_round_d2',
            normalize_weights_flag=False,
        )

        return gc_df

    def eval_circ_df_cudaq(
        self,
        qaoa_gpt_circ_df,
        adapt_gpt_path='.'
    ):
        qaoa_gpt_circ_eval_df = eval_adapt_gpt_circ_cudaq(
            qaoa_gpt_circ_df,
            n_nodes=self.n_nodes,
            temp_folder=self.temp_folder,
            pool_type=self.pool_type
        )

        output_columns_list =[
            "graph_prefix",
            "graph",
            "n_edges",
            "q_circuits",
            "adapt_gpt_energies"
        ]

        if "true_energy" in qaoa_gpt_circ_df.columns:
            output_columns_list.append("true_energy")

        if "energy_gurobi" in qaoa_gpt_circ_df.columns:
            output_columns_list.append("energy_gurobi")
        
        return qaoa_gpt_circ_eval_df[output_columns_list]
    
        
        
        