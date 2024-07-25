from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import cudaq
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.spatial import Voronoi
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from tqdm import tqdm


class Coreset:

    def __init__(
        self,
        raw_data: np.ndarray,
        number_of_sampling_for_centroids: int,
        coreset_size: int,
        number_of_coresets_to_evaluate: Optional[int] = 10,
        coreset_method: Optional[str] = "BFL2",
        k_value_for_BLK2: Optional[int] = 2,
    ) -> None:
        self._raw_data = raw_data
        self._coreset_size = coreset_size
        self._number_of_coresets_to_evaluate = number_of_coresets_to_evaluate
        self._number_of_sampling_for_centroids = number_of_sampling_for_centroids
        self._k_value_for_BLK2 = k_value_for_BLK2

        if coreset_method not in ["BFL2", "BLK2"]:
            raise ValueError("Coreset method must be either BFL2 or BLK2.")
        else:
            self._coreset_method = coreset_method

    @property
    def raw_data(self) -> np.ndarray:
        return self._raw_data

    @property
    def coreset_size(self) -> int:
        return self._coreset_size

    @property
    def number_of_coresets_to_evaluate(self) -> int:
        return self._number_of_coresets_to_evaluate

    @property
    def number_of_sampling_for_centroids(self) -> int:
        return self._number_of_sampling_for_centroids

    @property
    def coreset_method(self) -> str:
        return self._coreset_method

    @property
    def k_value_for_BLK2(self) -> int:
        return self._k_value_for_BLK2

    @raw_data.setter
    def raw_data(self, raw_data: np.ndarray) -> None:
        self._raw_data = raw_data

    @coreset_size.setter
    def coreset_size(self, coreset_size: int) -> None:
        self._coreset_size = coreset_size

    @number_of_coresets_to_evaluate.setter
    def number_of_coresets_to_evaluate(
            self, number_of_coresets_to_evaluate: int) -> None:
        self._number_of_coresets_to_evaluate = number_of_coresets_to_evaluate

    @number_of_sampling_for_centroids.setter
    def number_of_sampling_for_centroids(
            self, number_of_sampling_for_centroids: int) -> None:
        self._number_of_sampling_for_centroids = number_of_sampling_for_centroids

    @coreset_method.setter
    def coreset_method(self, coreset_method: str) -> None:
        self._coreset_method = coreset_method

    @k_value_for_BLK2.setter
    def k_value_for_BLK2(self, k_value_for_BLK2: int) -> None:
        self._k_value_for_BLK2 = k_value_for_BLK2

    def get_best_coresets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the best coreset vectors and weights for a given data.

        Returns:
            `Tuple[np.ndarray, np.ndarray]`: The coreset vectors and weights.
        """

        centroids = self.get_best_centroids()

        if self._coreset_method == "BFL2":
            print("Using BFL2 method to generate coresets")
            coreset_vectors, coreset_weights = self.get_coresets_using_BFL2(
                centroids)

        elif self._coreset_method == "BLK2":
            print("Using BLK2 method to generate coresets")
            coreset_vectors, coreset_weights = self.get_coresets_using_BLK2(
                centroids)
        else:
            raise ValueError("Coreset method must be either BFL2 or BLK2.")

        coreset_vectors, coreset_weights = self.best_coreset_using_kmeans_cost(
            coreset_vectors, coreset_weights)

        self.coreset_vectors = coreset_vectors
        self.coreset_weights = coreset_weights

        return (np.array(coreset_vectors), np.array(coreset_weights))

    def get_coresets_using_BFL2(
        self, centroids: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generates coreset vectors and weights using the BFL2 algorithm.

        Args:
            `centroids (List[np.ndarray])`: The centroids to use for the coreset generation.

        Returns:
            `Tuple[List[np.ndarray], List[np.ndarray]]`: List of coreset vectors and weights.
        """

        coreset_vectors_list = []
        coreset_weights_list = []
        for i in range(self.number_of_coresets_to_evaluate):
            coreset_vectors, coreset_weights = self.BFL2(centroids=centroids)
            coreset_vectors_list.append(coreset_vectors)
            coreset_weights_list.append(coreset_weights)

        return (coreset_vectors_list, coreset_weights_list)

    def get_best_centroids(self) -> List[np.ndarray]:
        """
        Get the best centroids using the D2 sampling algorithm.

        Returns:
            List[np.ndarray]: The best centroids.

        """

        best_centroid_coordinates, best_centroid_cost = None, np.inf

        for _ in range(self.number_of_sampling_for_centroids):
            centroids = self.D2_sampling()
            cost = self.get_cost(centroids)
            if cost < best_centroid_cost:
                best_centroid_coordinates, best_centroid_cost = centroids, cost

        return best_centroid_coordinates

    def D2_sampling(self) -> List[np.ndarray]:
        """
        Selects the centroids from the data points using the D2 sampling algorithm.

        Returns:
            List[np.ndarray]: The selected centroids as a list.
        """

        centroids = []
        data_vectors = self.raw_data

        centroids.append(data_vectors[np.random.choice(len(data_vectors))])

        for _ in range(self.coreset_size - 1):
            p = np.zeros(len(data_vectors))
            for i, x in enumerate(data_vectors):
                p[i] = self.distance_to_centroids(x, centroids)[0]**2
            p = p / sum(p)
            centroids.append(data_vectors[np.random.choice(len(data_vectors),
                                                           p=p)])

        return centroids

    def get_cost(self, centroids: Union[List[np.ndarray], np.ndarray]) -> float:
        """
        Computes the sum of between each data points and each centroids.

        Args:
            `centroids (Union[List[np.ndarray], np.ndarray])`: The centroids to evaluate.

        Returns:
            float: The cost of the centroids.

        """

        cost = 0.0
        for x in self.raw_data:
            cost += self.distance_to_centroids(x, centroids)[0]**2
        return cost

    def distance_to_centroids(
            self, data_instance: np.ndarray,
            centroids: Union[List[np.ndarray],
                             np.ndarray]) -> Tuple[float, int]:
        """
        Compute the distance between a data instance and the centroids.

        Args:
            `data_instance (np.ndarray)`: The data instance.
            `centroids (Union[List[np.ndarray], np.ndarray])`: The centroids as a list or `numpy` array.

        Returns:
            Tuple[float, int]: The minimum distance and the index of the closest centroid.
        """

        minimum_distance = np.inf
        closest_index = -1
        for i, centroid in enumerate(centroids):
            distance_between_data_instance_and_centroid = np.linalg.norm(
                data_instance - centroid)
            if distance_between_data_instance_and_centroid < minimum_distance:
                minimum_distance = distance_between_data_instance_and_centroid
                closest_index = i

        return (minimum_distance, closest_index)

    def BFL2(
        self, centroids: Union[List[np.ndarray], np.ndarray]
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Performs Algorithm 2 from https://arxiv.org/pdf/1612.00889.pdf BFL2. This will pick the coreset vectors and its corresponding weights.

        Args:
            centroids (List): The centroids to use for the coreset generation.

        Returns:
            Tuple[List, List]: The coreset vectors and coreset weights.
        """

        number_of_data_points_close_to_a_cluster = {
            i: 0 for i in range(len(centroids))
        }
        sum_distance_to_closest_cluster = 0.0
        for data_instance in self.raw_data:
            min_dist, closest_index = self.distance_to_centroids(
                data_instance, centroids)
            number_of_data_points_close_to_a_cluster[closest_index] += 1
            sum_distance_to_closest_cluster += min_dist**2

        Prob = np.zeros(len(self._raw_data))
        for i, p in enumerate(self._raw_data):
            min_dist, closest_index = self.distance_to_centroids(p, centroids)
            Prob[i] += min_dist**2 / (2 * sum_distance_to_closest_cluster)
            Prob[i] += 1 / (
                2 * len(centroids) *
                number_of_data_points_close_to_a_cluster[closest_index])

        if not (0.999 <= sum(Prob) <= 1.001):
            raise ValueError(
                "sum(Prob) = %s; the algorithm should automatically "
                "normalize Prob by construction" % sum(Prob))
        chosen_indices = np.random.choice(len(self._raw_data),
                                          size=self._coreset_size,
                                          p=Prob)
        weights = [1 / (self._coreset_size * Prob[i]) for i in chosen_indices]

        return ([self._raw_data[i] for i in chosen_indices], weights)

    def kmeans_cost(self,
                    coreset_vectors: np.ndarray,
                    sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Compute the cost of coreset vectors using k-means clustering.

        Args:
            `coreset_vectors (np.ndarray)`: The coreset vectors.
            `sample_weight (np.ndarray)`: The sample weights.

        Returns:
            float: The cost of the k-means clustering.

        """

        kmeans = KMeans(n_clusters=2).fit(coreset_vectors,
                                          sample_weight=sample_weight)
        return self.get_cost(kmeans.cluster_centers_)

    def best_coreset_using_kmeans_cost(
            self, coreset_vectors: List[np.ndarray],
            coreset_weights: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the best coreset using k-means cost.

        Args:
            `coreset_vectors (List[np.ndarray])`: The coreset vectors.
            `coreset_weights (List[np.ndarray])`: The coreset weights.

        Returns:
            Tuple: The best coreset vectors and coreset weights.
        """

        cost_coreset = [
            self.kmeans_cost(
                coreset_vectors=coreset_vectors[i],
                sample_weight=coreset_weights[i],
            ) for i in range(self._number_of_coresets_to_evaluate)
        ]

        best_index = cost_coreset.index(np.min(cost_coreset))
        return (coreset_vectors[best_index], coreset_weights[best_index])

    def get_coresets_using_BLK2(
        self, centroids: Union[List[np.ndarray], np.ndarray]
    ) -> Tuple[List[List[np.ndarray]], List[List[float]]]:
        """
        Generates coreset vectors and weights using Algorithm 2.

        Args:
            `centroids (List[np.ndarray])`: The centroids to use for the coreset generation.

        Returns:
            `Tuple[List[List[np.ndarray]], List[List[float]]]`: The coreset vectors and coreset weights.
        """

        coreset_vectors_list = []
        coreset_weights_list = []
        for i in range(self.number_of_coresets_to_evaluate):
            coreset_vectors, coreset_weights = self.BLK2(centroids=centroids)
            coreset_vectors_list.append(coreset_vectors)
            coreset_weights_list.append(coreset_weights)

        return (coreset_vectors_list, coreset_weights_list)

    def BLK2(
        self,
        centroids: Union[List[np.ndarray], np.ndarray],
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Performs Algorithm 2 from  https://arxiv.org/pdf/1703.06476.pdf.

        Args:
            `centroids (List[np.ndarray])`: The centroids to use for the coreset generation.

        Returns:
            `Tuple[List, List]`: The coreset vectors and coreset weights.
        """

        alpha = 16 * (np.log2(self._k_value_for_BLK2) + 2)

        B_i_totals = [0] * len(centroids)
        B_i = [np.empty_like(self._raw_data) for _ in range(len(centroids))]
        for data_instance in self._raw_data:
            _, closest_index = self.distance_to_centroids(
                data_instance, centroids)
            B_i[closest_index][B_i_totals[closest_index]] = data_instance
            B_i_totals[closest_index] += 1

        c_phi = sum([
            self.distance_to_centroids(data_instance, centroids)[0]**2
            for data_instance in self._raw_data
        ]) / len(self._raw_data)

        p = np.zeros(len(self._raw_data))

        sum_dist = {i: 0.0 for i in range(len(centroids))}
        for i, data_instance in enumerate(self._raw_data):
            dist, closest_index = self.distance_to_centroids(
                data_instance, centroids)
            sum_dist[closest_index] += dist**2

        for i, data_instance in enumerate(self._raw_data):
            p[i] = 2 * alpha * self.distance_to_centroids(
                data_instance, centroids)[0]**2 / c_phi

            closest_index = self.distance_to_centroids(data_instance,
                                                       centroids)[1]
            p[i] += 4 * alpha * sum_dist[closest_index] / (
                B_i_totals[closest_index] * c_phi)

            p[i] += 4 * len(self._raw_data) / B_i_totals[closest_index]
        p = p / sum(p)

        chosen_indices = np.random.choice(len(self._raw_data),
                                          size=self._coreset_size,
                                          p=p)
        weights = [1 / (self._coreset_size * p[i]) for i in chosen_indices]

        return [self._raw_data[i] for i in chosen_indices], weights

    @staticmethod
    def coreset_to_graph(
        coreset_vectors: np.ndarray,
        coreset_weights: np.ndarray,
        metric: Optional[str] = "dot",
        number_of_qubits_representing_data: Optional[int] = 1,
    ) -> nx.Graph:
        """
        Convert coreset vectors to a graph.

        Args:
            `coreset_vectors (np.ndarray)`: The coreset vectors.
            `coreset_weights (np.ndarray)`: The coreset weights.
            `metric (str, optional)`: The metric to use. Defaults to "dot".
            `number_of_qubits_representing_data (int, optional)`: The number of qubits representing the data. Defaults to 1.

        Returns:
            `nx.Graph`: The graph.
        """

        coreset = [(w, v) for w, v in zip(coreset_weights, coreset_vectors)]

        vertices = len(coreset)
        vertex_labels = [
            number_of_qubits_representing_data * int(i) for i in range(vertices)
        ]
        G = nx.Graph()
        G.add_nodes_from(vertex_labels)
        edges = [(
            number_of_qubits_representing_data * i,
            number_of_qubits_representing_data * j,
        ) for i in range(vertices) for j in range(i + 1, vertices)]

        G.add_edges_from(edges)

        for edge in G.edges():
            v_i = edge[0] // number_of_qubits_representing_data
            v_j = edge[1] // number_of_qubits_representing_data
            w_i = coreset[v_i][0]
            w_j = coreset[v_j][0]
            if metric == "dot":
                mval = np.dot(
                    coreset[v_i][1],
                    coreset[v_j][1],
                )
            elif metric == "dist":
                mval = np.linalg.norm(coreset[v_i][1] - coreset[v_j][1])
            else:
                raise Exception("Unknown metric: {}".format(metric))

            G[edge[0]][edge[1]]["weight"] = w_i * w_j * mval

        return G

    @staticmethod
    def normalize_array(vectors: np.ndarray,
                        centralize: bool = False) -> np.ndarray:
        """
        Normalize and centralize the array

        Args:
            `vectors (np.ndarray)`: The vectors to normalize
            `centralize (bool, optional)`: Centralize the array. Defaults to False.

        Returns:
            `np.ndarray`: The normalized array
        """

        if centralize:
            vectors = vectors - np.mean(vectors, axis=0)

        max_abs = np.max(np.abs(vectors), axis=0)
        vectors_norm = vectors / max_abs

        return vectors_norm

    @staticmethod
    def create_dataset(
        n_samples: float,
        covariance_values: List[float] = [-0.8, -0.8],
        n_features: Optional[int] = 2,
        number_of_samples_from_distribution: Optional[int] = 500,
        mean_array: Optional[np.ndarray] = np.array([[0, 0], [7, 1]]),
        random_seed: Optional[int] = 10,
    ) -> np.ndarray:
        """
        Create a data set with the given parameters.

        Args:
            `n_samples (float)`: The number of samples.
            `covariance_values (List[float], optional)`: The covariance values. Defaults to [-0.8, -0.8].
            `n_features (int, optional)`: The number of features. Defaults to 2.
            `number_of_samples_from_distribution (int, optional)`: The number of samples from the distribution. Defaults to 500.
            `mean_array (np.ndarray, optional)`: The mean array. Defaults to `np.array([[0, 0], [7, 1]])`.
            `random_seed (int, optional)`: The random seed. Defaults to 10.

        Returns:
            `np.ndarray`: The data set created
        """

        random_seed = random_seed

        X = np.zeros((n_samples, n_features))

        for idx, val in enumerate(covariance_values):
            covariance_matrix = np.array([[1, val], [val, 1]])

            distr = multivariate_normal(cov=covariance_matrix,
                                        mean=mean_array[idx],
                                        seed=random_seed)

            data = distr.rvs(size=number_of_samples_from_distribution)

            X[number_of_samples_from_distribution *
              idx:number_of_samples_from_distribution * (idx + 1)][:] = data

        return X


class DivisiveClustering(ABC):

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
        normalize_vectors: Optional[bool] = True,
        sort_by_descending: Optional[bool] = True,
        coreset_to_graph_metric: Optional[str] = "dot",
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

    @abstractmethod
    def run_divisive_clustering(
        self, coreset_vectors_df_for_iteration: pd.DataFrame
    ) -> Union[List[str], List[int]]:
        """
        Run the divisive clustering algorithm.

        Args:
            `coreset_vectors_df_for_iteration (pd.DataFrame)`: The coreset vectors for the iteration.

        Returns:
            `Union[List[str], List[int]]`: The bitstring or the cluster. The return will depend on the name of the data point given.
        """

        pass

    def get_hierarchical_clustering_sequence(
        self,
        coreset_vectors_df_for_iteration: np.ndarray,
        hierarchial_sequence: List,
    ) -> List:
        """
        Get the hierarchical clustering sequence.

        Args:
            `coreset_vectors_df_for_iteration (np.ndarray)`: The coreset vectors for the iteration.
            `hierarchial_sequence (List)`: The hierarchical sequence.

        """

        bitstring = self.run_divisive_clustering(
            coreset_vectors_df_for_iteration)
        return self._add_children_to_hierarchial_clustering(
            coreset_vectors_df_for_iteration, hierarchial_sequence, bitstring)

    def _get_iteration_coreset_vectors_and_weights(
        self, coreset_vectors_df_for_iteration: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the iteration coreset vectors and weights.

        Args:
            `coreset_vectors_df_for_iteration (pd.DataFrame)`: The coreset vectors for the iteration.

        Returns:
            `Tuple[np.ndarray, np.ndarray]`: The coreset vectors and weights.

        """

        coreset_vectors_for_iteration = coreset_vectors_df_for_iteration[[
            "X", "Y"
        ]].to_numpy()

        coreset_weights_for_iteration = coreset_vectors_df_for_iteration[
            "weights"].to_numpy()

        if self.normalize_vectors:
            coreset_vectors_for_iteration = Coreset.normalize_array(
                coreset_vectors_for_iteration, True)
            coreset_weights_for_iteration = Coreset.normalize_array(
                coreset_weights_for_iteration)

        return (coreset_vectors_for_iteration, coreset_weights_for_iteration)

    def brute_force_cost_maxcut(self, bitstrings: list[Union[str, int]],
                                G: nx.graph) -> Dict[str, float]:
        """
        Cost function for brute force method

        Args:
            bitstrings: list of bit strings
            G: The graph of the problem

        Returns:
            Dict: Dictionary with bitstring and cost value
        """

        cost_value = {}
        for bitstring in tqdm(bitstrings):
            c = 0
            for i, j in G.edges():
                edge_weight = G[i][j]["weight"]
                c += self._get_edge_cost(bitstring, i, j, edge_weight)

            cost_value.update({bitstring: c})

        return cost_value

    def _get_edge_cost(self, bitstring: str, i: int, j: int,
                       edge_weight: float) -> float:
        """
        Get the edge cost using MaxCut cost function.

        Args:
            bitstring: The bitstring
            i: The first node
            j: The second node
            edge_weight: The edge weight

        Returns:
            float: The edge cost
        """

        ai = int(bitstring[i])
        aj = int(bitstring[j])

        return -1 * edge_weight * (1 - ((-1)**ai) * ((-1)**aj))

    def _add_children_to_hierarchial_clustering(
        self,
        iteration_dataframe: pd.DataFrame,
        hierarchial_sequence: list,
        bitstring: str,
    ) -> List[Union[str, int]]:
        """
        Add children to the hierarchical clustering sequence.

        Args:
            `iteration_dataframe (pd.DataFrame)`: The iteration data frame.
            `hierarchial_sequence (list)`: The hierarchical sequence.
            `bitstring (str)`: The bitstring.

        Returns:
            list: The hierarchical sequence.
        """

        iteration_dataframe["cluster"] = [int(bit) for bit in bitstring]

        for j in range(2):
            idx = list(
                iteration_dataframe[iteration_dataframe["cluster"] == j].index)
            if len(idx) > 0:
                hierarchial_sequence.append(idx)

        return hierarchial_sequence

    @staticmethod
    def get_divisive_cluster_cost(hierarchical_clustering_sequence: List[Union[
        str, int]], coreset_data: pd.DataFrame) -> List[float]:
        """
        Get the cost of the divisive clustering at each iteration.

        Args:
            `hierarchical_clustering_sequence (List)`: The hierarchical clustering sequence.
            `coreset_data (pd.DataFrame)`: The coreset data.

        Returns:
            List[float]: The cost of the divisive clustering sequence.
        """

        coreset_data = coreset_data.drop(["Name", "weights"], axis=1)
        cost_at_each_iteration = []
        for parent in hierarchical_clustering_sequence:
            children_lst = Dendrogram.find_children(
                parent, hierarchical_clustering_sequence)

            if not children_lst:
                continue
            else:
                children_1, children_2 = children_lst

                parent_data_frame = coreset_data.iloc[parent]

                parent_data_frame["cluster"] = 0

                parent_data_frame.loc[children_2, "cluster"] = 1

                cost = 0

                centroid_coords = parent_data_frame.groupby("cluster").mean()[[
                    "X", "Y"
                ]]
                centroid_coords = centroid_coords.to_numpy()

                for idx, row in parent_data_frame.iterrows():
                    if row.cluster == 0:
                        cost += np.linalg.norm(row[["X", "Y"]] -
                                               centroid_coords[0])**2
                    else:
                        cost += np.linalg.norm(row[["X", "Y"]] -
                                               centroid_coords[1])**2

                cost_at_each_iteration.append(cost)

        return cost_at_each_iteration

    def _get_best_bitstring(self, counts: cudaq.SampleResult,
                            G: nx.Graph) -> str:
        """
        From the simulator output, extract the best bitstring.

        Args:
            `counts (cudaq.SampleResult)`: The counts.
            `G (nx.Graph)`: The graph.

        Returns:
            `str`: The best bitstring.
        """

        counts_pd = pd.DataFrame(counts.items(),
                                 columns=["bitstring", "counts"])
        counts_pd[
            "probability"] = counts_pd["counts"] / counts_pd["counts"].sum()
        bitstring_probability_df = counts_pd.drop(columns=["counts"])
        bitstring_probability_df = bitstring_probability_df.sort_values(
            "probability", ascending=self.sort_by_descending)

        unacceptable_bitstrings = [
            "".join("1" for _ in range(10)),
            "".join("0" for _ in range(10)),
        ]

        bitstring_probability_df = bitstring_probability_df[
            ~bitstring_probability_df["bitstring"].isin(unacceptable_bitstrings
                                                       )]

        if len(bitstring_probability_df) > 10:
            selected_rows = int(
                len(bitstring_probability_df) * self.threshold_for_maxcut)
        else:
            selected_rows = int(len(bitstring_probability_df) / 2)

        bitstring_probability_df = bitstring_probability_df.head(selected_rows)

        bitstrings = bitstring_probability_df["bitstring"].tolist()

        brute_force_cost_of_bitstrings = self.brute_force_cost_maxcut(
            bitstrings, G)

        return min(brute_force_cost_of_bitstrings,
                   key=brute_force_cost_of_bitstrings.get)

    def get_divisive_sequence(
            self, full_coreset_df: pd.DataFrame) -> List[Union[str, int]]:
        """
        Perform divisive clustering on the coreset data.

        Args:
            `full_coreset_df (pd.DataFrame)`: The full coreset data.

        Returns:
            `List[Union[str, int]]`: The hierarchical clustering sequence.
        """

        index_iteration_counter = 0
        single_clusters = 0

        index_values = list(range(len(full_coreset_df)))
        hierarchical_clustering_sequence = [index_values]

        while single_clusters < len(index_values):
            index_values_to_evaluate = hierarchical_clustering_sequence[
                index_iteration_counter]
            if len(index_values_to_evaluate) == 1:
                single_clusters += 1

            elif len(index_values_to_evaluate) == 2:
                hierarchical_clustering_sequence.append(
                    [index_values_to_evaluate[0]])
                hierarchical_clustering_sequence.append(
                    [index_values_to_evaluate[1]])

            else:
                coreset_vectors_df_for_iteration = full_coreset_df.iloc[
                    index_values_to_evaluate]

                hierarchical_clustering_sequence = self.get_hierarchical_clustering_sequence(
                    coreset_vectors_df_for_iteration,
                    hierarchical_clustering_sequence,
                )

            index_iteration_counter += 1

        return hierarchical_clustering_sequence


class Dendrogram:

    def __init__(
            self, coreset_data: pd.DataFrame,
            hierarchical_clustering_sequence: List[Union[str, int]]) -> None:
        self._coreset_data = self.__create_coreset_data(coreset_data)
        self._hierarchial_clustering_sequence = self.__convert_numbers_to_name(
            hierarchical_clustering_sequence, coreset_data)
        self.linkage_matrix = []

    @property
    def coreset_data(self) -> pd.DataFrame:
        return self._coreset_data

    @coreset_data.setter
    def coreset_data(self, coreset_data: pd.DataFrame) -> None:
        self.linkage_matrix = []
        self._coreset_data = coreset_data

    @property
    def hierarchical_clustering_sequence(self) -> List[Union[str, int]]:
        return self._hierarchial_clustering_sequence

    @hierarchical_clustering_sequence.setter
    def hierarchical_clustering_sequence(
            self, hierarchical_clustering_sequence: List[Union[str,
                                                               int]]) -> None:
        self.linkage_matrix = []
        self._hierarchial_clustering_sequence = hierarchical_clustering_sequence

    def __call__(self) -> List:
        if not self.linkage_matrix:
            self.get_linkage_matrix(self._hierarchial_clustering_sequence[0])

        return self.linkage_matrix

    def __create_coreset_data(self, coreset_data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates coreset data that can be used for plotting.

        Args:
            `coreset_data (pd.DataFrame)`: The coreset data.

        Returns:
            `pd.DataFrame`: The coreset data.
        """

        _coreset_data = coreset_data.copy()
        _coreset_data.index = _coreset_data.Name

        return _coreset_data.drop(columns=["Name", "weights"])

    def __convert_numbers_to_name(self,
                                  hierarchical_clustering_sequence: List[int],
                                  coreset_data: pd.DataFrame) -> List[str]:
        """
        Converts the int in the hierarchical sequence into the instance name. This would be used to plot the leaves of the dendrogram.

        Args:
            `hierarchical_clustering_sequence (List[int])`: The hierarchical clustering sequence.
            `coreset_data (pd.DataFrame)`: The coreset data.

        Returns:
            List[str]: The converted hierarchical clustering sequence.
        """

        converted_hc = []
        for hc in hierarchical_clustering_sequence:
            converted_hc.append([coreset_data.Name[num] for num in hc])

        return converted_hc

    def plot_dendrogram(
        self,
        plot_title: Optional[str] = "DIANA",
        orientation: Optional[str] = "top",
        color_threshold: Optional[int] = None,
        colors: Optional[List] = None,
        clusters: Optional[np.ndarray] = None,
        link_color_func: Optional[Callable] = None,
    ):
        """
        Plots the dendrogram.

        Args:
            `plot_title (str, optional)`: The plot title. Defaults to "DIANA".
            `orientation (str, optional)`: The orientation of the dendrogram. Defaults to "top".
            `color_threshold (int, optional)`: The color threshold to convert hierarchical clustering into flat clustering. Defaults to None.
            `colors (List, optional)`: The colors for the leaves. Defaults to None.
            `clusters (np.ndarray, optional)`: Flat clustering results from applying threshold. Defaults to None.
            `link_color_func (Callable, optional)`: Function to color the branches. Defaults to None.
        """

        if not self.linkage_matrix:
            self.get_linkage_matrix(self._hierarchial_clustering_sequence[0])

        if clusters is None:
            clusters = np.array([0] * len(self._coreset_data))

        fig = plt.figure(figsize=(10, 10), dpi=100)
        plt.title(plot_title)
        dn = dendrogram(
            self.linkage_matrix,
            labels=self._coreset_data.index,
            orientation=orientation,
            color_threshold=color_threshold * 100 if colors else None,
        )

        if color_threshold is not None:
            plt.axhline(y=color_threshold, color="r", linestyle="--")

        if colors is not None:
            if len(colors) < len(set(clusters)):
                raise ValueError(
                    "Number of colors should be equal to number of clusters")
            else:
                colors_dict = {
                    self._coreset_data.index[i]: colors[j]
                    for i, j in enumerate(clusters)
                }

                ax = plt.gca()
                xlbls = ax.get_xmajorticklabels()
                for lbl in xlbls:
                    lbl.set_color(colors_dict[lbl.get_text()])

        plt.show()

    def get_clusters_using_height(self, threshold: float) -> np.ndarray:
        """
        Get flat clusters from the hierarchical clustering using a threshold.

        Args:
            threshold (float): The height threshold to convert.

        Returns:
            `np.ndarray`: The flat cluster labels.
        """

        if not self.linkage_matrix:
            self.get_linkage_matrix(self._hierarchial_clustering_sequence[0])

        clusters = fcluster(self.linkage_matrix,
                            threshold,
                            criterion="distance")

        return np.array(clusters) - 1

    def get_clusters_using_k(self, k: int) -> np.ndarray:
        """
        Get flat clusters from the hierarchical cluster by defining the number of clusters.

        Args:
            k (int): The number of clusters.

        Returns:
            `np.ndarray`: The flat cluster labels.

        """
        if not self.linkage_matrix:
            self.get_linkage_matrix(self._hierarchial_clustering_sequence[0])

        clusters = fcluster(self.linkage_matrix, k, criterion="maxclust")

        return np.array(clusters) - 1

    def plot_clusters(
        self,
        clusters: np.ndarray,
        colors: List[str],
        plot_title: str,
        show_annotation: Optional[bool] = False,
    ):
        """
        Plot the flat clusters.

        Args:
            `clusters (np.ndarray)`: The flat clusters.
            `colors (List[str])`: The colors for the clusters.
            `plot_title (str)`: The plot title.
            `show_annotation (bool, optional)`: Show annotation. Defaults to False.

        """
        if len(colors) < len(set(clusters)):
            raise ValueError(
                "Number of colors should be equal to number of clusters")
        coreset_data = self._coreset_data.copy()
        coreset_data["clusters"] = clusters
        for i in range(coreset_data.clusters.nunique()):
            data = coreset_data[coreset_data.clusters == i]
            plt.scatter(data.X, data.Y, c=colors[i], label=f"Cluster {i}")
        if show_annotation:
            for _, row in coreset_data.iterrows():
                plt.annotate(row.name, (row.X, row.Y))
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title(plot_title)
        plt.show()

    def get_linkage_matrix(self, parent: List[str]) -> int:
        """
        Create the linkage matrix for the dendrogram and returns the index of the new branch.

        Args:
            parent (`List[str]`): The parent cluster.

        Returns:
            List: The linkage matrix.
        """

        if len(parent) < 2:
            index_of_parent = np.argwhere(self._coreset_data.index == parent[0])
            return index_of_parent[0][0]
        children_1, children_2 = self.find_children(
            parent, self._hierarchial_clustering_sequence)

        index1 = self.get_linkage_matrix(children_1)
        index2 = self.get_linkage_matrix(children_2)
        self.linkage_matrix.append([
            index1,
            index2,
            self.distance(index1) + self.distance(index2),
            self.cluster_len(index1) + self.cluster_len(index2),
        ])

        return len(self.linkage_matrix) - 1 + len(self.coreset_data)

    def distance(self, i: int) -> float:
        """
        Get the distance between two clusters.

        Args:
            i (int): The index of the cluster.

        Returns:
            float: The distance of the cluster.
        """

        if i >= len(self._coreset_data):
            distance = self.linkage_matrix[i - len(self._coreset_data)][2]
        else:
            distance = sum(
                self._coreset_data.iloc[i]) / (len(self.coreset_data) - 1)

        return abs(distance)

    def cluster_len(self, i: int):
        """
        Get the length of the cluster.

        Args:
            i (int): The index of the cluster.

        Returns:
            int: The length of the cluster.
        """

        if i >= len(self._coreset_data):
            return self.linkage_matrix[i - len(self._coreset_data)][3]
        else:
            return 1

    @staticmethod
    def find_children(
            parent: List[Union[str, int]],
            hierarchical_clustering_sequence: List[Union[str, int]]) -> List:
        """
        Find the children of a given parent cluster.

        Args:
            parent (List): The parent cluster.
            hierarchical_clustering_sequence (List): The hierarchical clustering sequence.

        Returns:
            List: The children of the parent cluster.
        """

        parent_position = hierarchical_clustering_sequence.index(parent)

        found = 0
        children = []
        for i in range(parent_position + 1,
                       len(hierarchical_clustering_sequence)):
            if any(item in hierarchical_clustering_sequence[i]
                   for item in parent):
                children.append(hierarchical_clustering_sequence[i])
                found += 1
                if found == 2:
                    break

        return children

    @staticmethod
    def plot_hierarchial_split(
            hierarchical_clustering_sequence: List[Union[str, int]],
            full_coreset_df: pd.DataFrame):
        """
        Plots the flat clusters at each iteration of the hierarchical clustering.

        Args:
            hierarchical_clustering_sequence (List): The hierarchical clustering sequence.
            `full_coreset_df` (`pd.DataFrame`): The full coreset data.
        """
        parent_clusters = [
            parent_cluster
            for parent_cluster in hierarchical_clustering_sequence
            if len(parent_cluster) > 1
        ]
        x_grid = int(np.sqrt(len(parent_clusters)))
        y_grid = int(np.ceil(len(parent_clusters) / x_grid))

        fig, axs = plt.subplots(x_grid, y_grid, figsize=(12, 12))

        for i, parent_cluster in enumerate(parent_clusters):
            parent_position = hierarchical_clustering_sequence.index(
                parent_cluster)
            children = Dendrogram.find_children(
                parent_cluster, hierarchical_clustering_sequence)
            coreset_for_parent_cluster = full_coreset_df.loc[parent_cluster]
            coreset_for_parent_cluster["cluster"] = 1
            coreset_for_parent_cluster.loc[children[0], "cluster"] = 0

            ax = axs[i // 3, i % 3]
            ax.scatter(
                coreset_for_parent_cluster["X"],
                coreset_for_parent_cluster["Y"],
                c=coreset_for_parent_cluster["cluster"],
            )
            for _, row in coreset_for_parent_cluster.iterrows():
                ax.annotate(row["Name"], (row["X"], row["Y"]))

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"Clustering at iteration {parent_position}")

        plt.tight_layout()
        plt.show()


class Voironi_Tessalation:

    def __init__(
        self,
        coreset_df: pd.DataFrame,
        clusters: np.ndarray,
        colors: List[str],
        tesslation_by_cluster: Optional[bool] = False,
    ) -> None:
        coreset_df["cluster"] = clusters

        if tesslation_by_cluster:
            cluster_means = coreset_df.groupby("cluster")[["X", "Y"]].mean()
            coreset_df = cluster_means.reset_index()
            coreset_df["cluster"] = [i for i in range(len(coreset_df))]

        coreset_df["color"] = [colors[i] for i in coreset_df.cluster]

        points = coreset_df[["X", "Y"]].to_numpy()

        self.coreset_df = coreset_df

        self.voronoi = Voronoi(points)

    def voronoi_finite_polygons_2d(self,
                                   radius: Optional[float] = None
                                  ) -> Tuple[List, np.ndarray]:
        """
        Creates the Voronoi regions and vertices for 2D data.

        Args:
            radius (Optional[None]): The radius from the data points to create the Voronoi regions. Defaults to None.

        Returns:
            Tuple: The regions and vertices.
        """

        if self.voronoi.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = self.voronoi.vertices.tolist()

        center = self.voronoi.points.mean(axis=0)
        if radius is None:
            radius = self.voronoi.points.ptp().max()

        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(self.voronoi.ridge_points,
                                      self.voronoi.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        for p1, region in enumerate(self.voronoi.point_region):
            vertices = self.voronoi.regions[region]

            if all(v >= 0 for v in vertices):
                new_regions.append(vertices)
                continue

            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue

                t = self.voronoi.points[p2] - self.voronoi.points[p1]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])

                midpoint = self.voronoi.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n

                far_point = self.voronoi.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)

            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)

    def plot_voironi(
        self,
        plot_title: Optional[str] = "Voronoi Tessalation",
        show_annotation: bool = False,
        show_scatters: bool = False,
    ):
        regions, vertices = self.voronoi_finite_polygons_2d()
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.tight_layout(pad=10)

        for j, region in enumerate(regions):
            polygon = vertices[region]
            color = self.coreset_df.color[j]
            breakpoint()
            plt.fill(*zip(*polygon), alpha=0.4, color=color, linewidth=0)
            if show_annotation:
                plt.annotate(
                    self.coreset_df.Name[j],
                    (self.coreset_df.X[j] + 0.2, self.coreset_df.Y[j]),
                    fontsize=10,
                )

        if show_scatters:
            plt.plot(self.coreset_df.X, self.coreset_df.Y, "ko")

        plt.xlim(min(self.coreset_df.X) - 1, max(self.coreset_df.X) + 1)
        plt.ylim(min(self.coreset_df.Y) - 1, max(self.coreset_df.Y) + 1)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(plot_title)
        plt.show()
