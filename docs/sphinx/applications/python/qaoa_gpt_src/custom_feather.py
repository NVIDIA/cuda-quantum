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

import numpy as np
import networkx as nx
import sys
import os
import math
import glob

# Add the FEATHER module path
#feather_path = '/home/cudaq/qaoa-gpt-cudaq-feather'
# Change from absolute path to current directory
feather_path = os.path.dirname(os.path.abspath(__file__))

if feather_path not in sys.path:
    sys.path.append(feather_path)

# Import the original FEATHER implementation
try:
    from FEATHER.src.feather import FEATHER as OrigFeatherNode  # For node-level embedding
    from FEATHER.src.feather import FEATHERG as OrigFeatherGraph  # For graph-level embedding
except ImportError as e:
    print(f"Import error: {e}")
    print("Current sys.path:", sys.path)
    print("Checking if files exist:")
    print(glob.glob(os.path.join(feather_path, 'FEATHER/src/*')))

class CustomFeatherGraph:
    """Custom wrapper for the original FEATHER implementation with multiple node features"""
    
    def __init__(
        self,
        theta_max=2.5,      # Default from original implementation
        eval_points=25,     # Default from original implementation
        order=5,            # Default from original implementation
        pooling="mean",     # Default from original implementation
        seed=42             # Add seed parameter with default value
    ):
        self.theta_max = theta_max
        self.eval_points = eval_points
        self.order = order
        self.pooling = pooling
        self.seed = seed
        self.embedding = None
    
    def _extract_features(self, graph):
        """
        Extract node features:
        1. Log degree
        2. Clustering coefficient
        """
        # Number of nodes in the graph
        num_nodes = graph.number_of_nodes()
        
        # Initialize feature matrix
        features = np.zeros((num_nodes, 2))  # 2 features per node
        
        # Calculate clustering coefficients for all nodes at once
        clustering_coeffs = nx.clustering(graph)
        
        # Extract log degree and clustering coefficient for each node
        for node in range(num_nodes):
            # Log of degree + 1 (to avoid log(0))
            features[node, 0] = math.log(graph.degree(node) + 1.0)
            
            # Clustering coefficient (default to 0 if not available)
            features[node, 1] = clustering_coeffs.get(node, 0.0)
            
        return features
    
    def fit(self, graphs):
        """
        Fit the model and generate embeddings for a list of graphs
        
        Args:
            graphs: List of networkx graphs to embed
        """
        # Set random seed
        np.random.seed(self.seed)
        
        # Process each graph to ensure it meets FEATHER requirements
        processed_graphs = []
        for graph in graphs:
            # Ensure the graph has self-loops and integer node labels
            graph = nx.convert_node_labels_to_integers(graph)
            graph = graph.copy()
            for node in graph.nodes():
                if not graph.has_edge(node, node):
                    graph.add_edge(node, node)
            processed_graphs.append(graph)
        
        # Initialize embedding list
        all_embeddings = []
        
        # Process each graph individually to use multiple features
        for graph in processed_graphs:
            # Extract node features (log degree and clustering coefficient)
            node_features = self._extract_features(graph)
            
            # Create edge list for FEATHER
            edge_list = [(int(u), int(v)) for u, v in graph.edges()]
            
            # Create FEATHER node-level embedding model
            feather_node = OrigFeatherNode(
                theta_max=self.theta_max,
                eval_points=self.eval_points,
                order=self.order
            )
            
            # Fit and transform to get node embeddings
            feather_node.fit(graph, node_features)
            node_embeddings = feather_node.get_embedding()
            
            # Pool node embeddings to get graph embedding
            if self.pooling == "mean":
                graph_embedding = np.mean(node_embeddings, axis=0)
            elif self.pooling == "max":
                graph_embedding = np.max(node_embeddings, axis=0)
            elif self.pooling == "min":
                graph_embedding = np.min(node_embeddings, axis=0)
            else:
                graph_embedding = np.mean(node_embeddings, axis=0)  # Default to mean
            
            all_embeddings.append(graph_embedding)
        
        # Convert to numpy array
        self.embedding = np.array(all_embeddings)
        
        # Print embedding dimension for verification
        if len(processed_graphs) > 0:
            print(f"Generated embedding dimension: {self.embedding.shape[1]}")
            
            # Expected dimension calculation
            expected_dim = 2 * self.eval_points * 2 * self.order  # 2 features * eval_points * 2 (sin/cos) * order
            print(f"Expected dimension: {expected_dim}")
        
        return self
    
    def get_embedding(self):
        """Return the graph embeddings"""
        return self.embedding