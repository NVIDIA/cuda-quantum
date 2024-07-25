import argparse
import os
import sys
import time
import warnings
from typing import Callable

import cudaq
import networkx as nx
import numpy as np
import pandas as pd
from cudaq import spin
from mpi4py import MPI

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from divisive_clustering import Coreset, DivisiveClustering

warnings.filterwarnings("ignore")

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "-t",
    "--target",
    type=str,
    choices=["qpp-cpu", "nvidia", "nvidia-mgpu"],
    help=
    "Quantum simulator backend. Default is qpp-cpu. See https://nvidia.github.io/cuda-quantum/0.6.0/using/simulators.html for more options.",
)
argparser.add_argument(
    "-d",
    "--depth",
    type=int,
    default=1,
    help="Depth of the QAOA circuit. Default is 1.",
)
argparser.add_argument("-i",
                       "--max_iterations",
                       type=int,
                       default=75,
                       help="Max iterations for the optimizer.")
argparser.add_argument("-s",
                       "--max_shots",
                       type=int,
                       default=100000,
                       help="Max shots for the simulation.")
argparser.add_argument("-m",
                       "--M",
                       type=int,
                       default=10,
                       help="Size of the coreset.")

args = argparser.parse_args()

target = args.target
coreset_size = args.M
circuit_depth = args.depth
max_iterations = args.max_iterations
max_shots = args.max_shots


class DivisiveClusteringVQA(DivisiveClustering):

    def __init__(
        self,
        circuit_depth: int,
        max_iterations: int,
        max_shots: int,
        threshold_for_max_cut: float,
        create_Hamiltonian: Callable,
        optimizer: cudaq.optimizers.optimizer,
        optimizer_function: Callable,
        create_circuit: Callable,
        normalize_vectors: bool = True,
        sort_by_descending: bool = True,
        coreset_to_graph_metric: str = "dist",
    ) -> None:
        self.circuit_depth = circuit_depth
        self.max_iterations = max_iterations
        self.max_shots = max_shots
        self.threshold_for_maxcut = threshold_for_max_cut
        self.normalize_vectors = normalize_vectors
        self.sort_by_descending = sort_by_descending
        self.coreset_to_graph_metric = coreset_to_graph_metric
        self.create_Hamiltonian = create_Hamiltonian
        self.create_circuit = create_circuit
        self.optimizer = optimizer
        self.optimizer_function = optimizer_function
        self.time_consumed = 0

    def run_divisive_clustering(
        self,
        coreset_vectors_df_for_iteration: pd.DataFrame,
    ):
        coreset_vectors_for_iteration_np, coreset_weights_for_iteration_np = (
            self._get_iteration_coreset_vectors_and_weights(
                coreset_vectors_df_for_iteration))

        G = Coreset.coreset_to_graph(
            coreset_vectors_for_iteration_np,
            coreset_weights_for_iteration_np,
            metric=self.coreset_to_graph_metric,
        )

        counts = self.get_counts_from_simulation(
            G,
            self.circuit_depth,
            self.max_iterations,
            self.max_shots,
        )

        return self._get_best_bitstring(counts, G)

    def get_counts_from_simulation(self, G, circuit_depth, max_iterations,
                                   max_shots):
        qubits = len(G.nodes)
        Hamiltonian = self.create_Hamiltonian(G)
        optimizer, parameter_count, initial_params = self.optimizer_function(
            self.optimizer,
            max_iterations,
            qubits=qubits,
            circuit_depth=circuit_depth)

        kernel = self.create_circuit(qubits, circuit_depth)

        def objective_function(parameter_vector: list[float],
                               hamiltonian=Hamiltonian,
                               kernel=kernel) -> tuple[float, list[float]]:
            get_result = lambda parameter_vector: cudaq.observe(
                kernel, hamiltonian, parameter_vector, qubits, circuit_depth
            ).expectation()

            cost = get_result(parameter_vector)

            return cost

        t0 = time.process_time()

        energy, optimal_parameters = optimizer.optimize(
            dimensions=parameter_count, function=objective_function)

        counts = cudaq.sample(
            kernel,
            optimal_parameters,
            qubits,
            circuit_depth,
            shots_count=max_shots,
        )

        tf = time.process_time()
        self.time_consumed += tf - t0

        return counts


def get_K2_Hamiltonian(G: nx.Graph) -> cudaq.SpinOperator:
    H = 0

    for i, j in G.edges():
        weight = G[i][j]["weight"]
        H += weight * (spin.z(i) * spin.z(j))

    return H


def get_QAOA_circuit(number_of_qubits, circuit_depth):

    @cudaq.kernel
    def kernel(thetas: list[float], number_of_qubits: int, circuit_depth: int):
        qubits = cudaq.qvector(number_of_qubits)

        layers = circuit_depth

        for layer in range(layers):
            for qubit in range(number_of_qubits):
                cx(qubits[qubit], qubits[(qubit + 1) % number_of_qubits])
                rz(2.0 * thetas[layer], qubits[(qubit + 1) % number_of_qubits])
                cx(qubits[qubit], qubits[(qubit + 1) % number_of_qubits])

            rx(2.0 * thetas[layer + layers], qubits)

    return kernel


def get_optimizer(optimizer: cudaq.optimizers.optimizer, max_iterations,
                  **kwargs):
    parameter_count = 4 * kwargs["circuit_depth"] * kwargs["qubits"]
    initial_params = np.random.uniform(-np.pi / 8.0, np.pi / 8.0,
                                       parameter_count)
    optimizer.initial_parameters = initial_params

    optimizer.max_iterations = max_iterations
    return optimizer, parameter_count, initial_params


def create_coreset_df(
    raw_data_size: int = 1000,
    number_of_sampling_for_centroids: int = 10,
    coreset_size: int = 10,
    number_of_coresets_to_evaluate: int = 4,
    coreset_method: str = "BFL2",
):
    raw_data = Coreset.create_dataset(raw_data_size)
    coreset = Coreset(
        raw_data=raw_data,
        number_of_sampling_for_centroids=number_of_sampling_for_centroids,
        coreset_size=coreset_size,
        number_of_coresets_to_evaluate=number_of_coresets_to_evaluate,
        coreset_method=coreset_method,
    )

    coreset_vectors, coreset_weights = coreset.get_best_coresets()

    coreset_df = pd.DataFrame({
        "X": coreset_vectors[:, 0],
        "Y": coreset_vectors[:, 1],
        "weights": coreset_weights,
    })
    coreset_df["Name"] = [chr(i + 65) for i in coreset_df.index]

    return coreset_df


if __name__ == "__main__":
    cudaq.set_target(target)

    coreset_df = create_coreset_df(
        raw_data_size=1000,
        number_of_sampling_for_centroids=10,
        coreset_size=coreset_size,
        number_of_coresets_to_evaluate=4,
        coreset_method="BFL2",
    )

    optimizer = cudaq.optimizers.COBYLA()

    divisive_clustering = DivisiveClusteringVQA(
        circuit_depth=circuit_depth,
        max_iterations=max_iterations,
        max_shots=max_shots,
        threshold_for_max_cut=0.5,
        create_Hamiltonian=get_K2_Hamiltonian,
        optimizer=optimizer,
        optimizer_function=get_optimizer,
        create_circuit=get_QAOA_circuit,
        normalize_vectors=True,
        sort_by_descending=True,
        coreset_to_graph_metric="dist",
    )

    t0 = time.process_time()

    hierarchial_clustering_sequence = divisive_clustering.get_divisive_sequence(
        coreset_df)
    tf = time.process_time()

    print(f"Total time for the execution: {tf - t0}")

    print(f"Total time spent on CUDA-Q: {divisive_clustering.time_consumed}")
