from typing import List, Optional, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from torch import Tensor


def get_physchem_properties(smiles: str) -> List[float]:
    """
    Calculates `physicochemical` properties for a given molecule.

    Parameters:
    - smiles (`str`): The SMILES representation of the molecule.

    Returns:
    - List[float]: A list of calculated `physicochemical` properties for the molecule.
    """

    mol = Chem.MolFromSmiles(smiles)

    properties = [
        ("MW", Descriptors.MolWt(mol)),  # type: ignore
        ("HBA", rdMolDescriptors.CalcNumHBA(mol)),
        ("HBD", rdMolDescriptors.CalcNumHBD(mol)),
        ("nRot", Descriptors.NumRotatableBonds(mol)),  # type: ignore
        ("nRing", rdMolDescriptors.CalcNumRings(mol)),
        ("nHet", rdMolDescriptors.CalcNumHeteroatoms(mol)),
        ("TPSA", Descriptors.TPSA(mol)),  # type: ignore
        ("LogP", Crippen.MolLogP(mol)),  # type: ignore
        ("StereoCenters",
         len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))),
    ]

    return [value for _, value in properties]


def scale_to_range(tensor: torch.Tensor, min_val: float,
                   max_val: float) -> torch.Tensor:
    """
    Scales a tensor to a specified range.

    Parameters:
    - tensor (torch.Tensor): The tensor to scale.
    - min_val (float): The minimum value of the range.
    - max_val (float): The maximum value of the range.

    Returns:
    - torch.Tensor: The scaled tensor.
    """

    # Prevent division by 0
    if tensor.min() == 0 and tensor.max() == 0:
        return tensor

    # Normalize tensor to [0, 1]
    normalized_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    # Scale to [min_val, max_val]
    scaled_tensor = normalized_tensor * (max_val - min_val) + min_val
    return scaled_tensor


def prepare_attention_inputs(
    tok: Tensor,
    pos: Tensor,
    query_weights: Tensor,
    key_weights: Tensor,
    physchem: Optional[Tensor] = None,
) -> Tensor:
    """
    Prepares attention inputs for batch processing with quantum circuits.

    This function organizes token, position, and key/query tensors of angles into pairwise combinations
    suitable for CUDA-Q functions. It structures input data into (batch, `seq_len`, `seq_len`, groups, `param_count`)
    format for efficient batched execution.


    Args:
        `tok` (torch.Tensor): Token `embeddings` (batch, `seq_len`, ansatz_layers * `num_tok_qubits`).
        `pos` (torch.Tensor): Position `embeddings` (batch, `seq_len`, ansatz_layers * `num_pos_qubits`).
        query_weights (torch.Tensor): Query weight tensor (batch, `seq_len`, ansatz_layers * total_working_qubits).
        key_weights (torch.Tensor): Key weight tensor (batch, `seq_len`, ansatz_layers * total_working_qubits).
        `physchem` (torch.Tensor, optional): Physicochemical `embeddings` (batch, `seq_len`, ansatz_layers * `num_physchem_qubits`), if used.

    Returns:
        torch.Tensor: A tensor of shape (batch, `seq_len`, `seq_len`, groups, `param_count`)

                      Notes:
                        - The final two dimensions are tensors [`tok1`, `pos1`, query_weights1, `tok2`, `pos2`, key_weights2]
                            or [`tok1`, `pos1`, query_weights1, `tok2`, `pos2`, key_weights2, `physchem`]
                            for all pairwise combinations

                        - Here, a "group" is a set of parameters that act on the same register (`tok`, `pos`, or `physchem`).

                        - Parameters are grouped by register (which are all of equal size in this work since `tok_size` == `pos_size`)
                          rather than by unitary, as grouping by unitaries
                          (which do not have the same number of parameters, i.e. token and position registers are 3 qubits while query and key states are all 6 working qubits)
                          would make it difficult to batch the tensors. It is more difficult work with ragged-arrayed tensors for other operations we used.

                        - These tensors we will eventually need to be flattened from shape (batch, n, n, groups, `param_count`) to
                          (batch*n*n, groups, `param_count`) to feed into the custom CUDA-Q functions.


    """

    # Get the sequence length dynamically
    seq_len = tok.size(1)

    # Expand token and position `embeddings` for pairwise interactions
    tok_i, tok_j = tok.unsqueeze(2).expand(-1, -1, seq_len,
                                           -1), tok.unsqueeze(1).expand(
                                               -1, seq_len, -1, -1)
    pos_i, pos_j = pos.unsqueeze(2).expand(-1, -1, seq_len,
                                           -1), pos.unsqueeze(1).expand(
                                               -1, seq_len, -1, -1)

    # As discussed in the `docstring`, we break the parameters for the query/key PQCs
    # into groups that act on the token register and ones that act on position register
    # (and possibly the `physchem` register)
    num_groups = 3 if physchem is not None else 2
    query_splits = torch.chunk(query_weights, num_groups, dim=2)
    key_splits = torch.chunk(key_weights, num_groups, dim=2)

    query_token_i, query_pos_i = query_splits[:2]
    key_token_j, key_pos_j = key_splits[:2]

    query_token_i = query_token_i.unsqueeze(2).expand(-1, -1, seq_len, -1)
    query_pos_i = query_pos_i.unsqueeze(2).expand(-1, -1, seq_len, -1)
    key_token_j = key_token_j.unsqueeze(1).expand(-1, seq_len, -1, -1)
    key_pos_j = key_pos_j.unsqueeze(1).expand(-1, seq_len, -1, -1)

    # Stack the base tensor groups
    input_tensors = [
        tok_i,
        pos_i,
        query_token_i,
        query_pos_i,
        tok_j,
        pos_j,
        key_token_j,
        key_pos_j,
    ]

    if physchem is not None:
        query_physchem_i, key_physchem_j = query_splits[2], key_splits[2]
        query_physchem_i = query_physchem_i.unsqueeze(2).expand(
            -1, -1, seq_len, -1)
        key_physchem_j = key_physchem_j.unsqueeze(1).expand(-1, seq_len, -1, -1)
        physchem_expanded = physchem.unsqueeze(2).expand(-1, -1, seq_len, -1)

        input_tensors.extend(
            [query_physchem_i, key_physchem_j, physchem_expanded])

    return torch.stack(input_tensors, dim=3)


def remove_redundant_circuits(
    full_batch_parameter_tensor: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Optimizes quantum circuit execution by eliminating redundant evaluations.

    Unlike classical GPUs, where calculating a full attention matrix and applying a mask is efficient,
    in this case it is more efficient to only run quantum simulations that calculate the lower triangle and diagonal explicitly.
    Additionally, duplicate circuits (identical parameter sets) are removed to minimize redundant computations.

    Args:
        full_batch_parameter_tensor (torch.Tensor): Quantum circuit `params` for each attention matrix element.
                                                    shape (batch, n, n, `num_groups`, `param_count`).

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            - `unique_lower_triangle_params` (Tensor): Unique circuit parameters to be executed.
            - unique_index_mapping (Tensor): Mapping of each non-redundant circuit back to its original index.
            - lower_triangle_indices (Tensor): Indices of circuits that belong to the lower triangle.
            - upper_triangle_indices (Tensor): Indices of circuits that belong to the upper triangle.
    """

    batch_size, n, _, groups, param_count = full_batch_parameter_tensor.shape

    # Create a lower triangular mask for the n x n attention matrix
    mask = torch.tril(
        torch.ones(n,
                   n,
                   dtype=torch.bool,
                   device=full_batch_parameter_tensor.device))
    mask = mask.unsqueeze(0).unsqueeze(-1).unsqueeze(
        -1)  # Shape: (1, n, n, 1, 1)
    mask = mask.expand(
        batch_size, n, n, groups,
        param_count)  # Shape: (batch_size, n, n, groups, `param_count`)

    # Apply the mask: Elements in the upper triangle are marked as -inf to exclude them from execution
    full_batch_parameter_tensor = full_batch_parameter_tensor.clone()
    full_batch_parameter_tensor[
        ~mask.expand(batch_size, n, n, groups, param_count)] = (float("-inf"))

    # Reshape to the final desired shape: (batch*`seq_len`^2, groups accepted by our custom cudaq functions, `param_count`)
    # to ensure proper batching of quantum circuit simulations
    flattened_tensor = full_batch_parameter_tensor.view(batch_size * n**2,
                                                        groups, param_count)

    # Identify circuits in the upper triangle by checking if all values are -inf
    is_upper_triangle = torch.isinf(flattened_tensor).all(dim=(1, 2))
    lower_triangle_indices = torch.nonzero(~is_upper_triangle, as_tuple=True)[0]
    upper_triangle_indices = torch.nonzero(is_upper_triangle, as_tuple=True)[0]

    # Extract only the necessary circuits (lower triangle)
    lower_triangle_parameters = flattened_tensor[lower_triangle_indices]

    # Up to this point, there should be batch*(`seq_len`^2 + `seq_len`)/2 circuits to run now that we have excluded the upper triangle
    # However, some parameter matrices we give to our custom cudaq functions may be the same,
    # so we want to remove these duplicates to avoid running the same circuit multiple times and instead broadcast the expectation values to the correct positions
    # This massively cuts down on computation time
    unique_lower_triangle_params, unique_index_mapping = torch.unique(
        lower_triangle_parameters, return_inverse=True, dim=0)

    return (
        unique_lower_triangle_params,
        unique_index_mapping,
        lower_triangle_indices,
        upper_triangle_indices,
    )


def repopulate_tensor(
    batch_size: int,
    seq_len: int,
    processed_results: Tensor,
    unique_index_mapping: Tensor,
    lower_triangle_indices: Tensor,
    upper_triangle_indices: Tensor,
) -> Tensor:
    """
    Repopulates the original attention tensor from processed results.

    This function reconstructs the original (batch, `seq_len`, `seq_len`) tensor by mapping
    processed expectation values back to their correct positions, restoring both unique
    and masked (-inf) values.

    Args:
        batch_size (int): Batch size.
        `seq_len` (int): Sequence length.
        processed_results (Tensor): Computed expectation values from unique circuits.
        unique_index_mapping (Tensor): Maps all matrices to their unique counterpart's index.
        lower_triangle_indices (Tensor): Indices of valid matrices in the flattened tensor.
        upper_triangle_indices (Tensor): Indices where the matrices were set to -inf.

    Returns:
        Tensor: Reconstructed tensor of shape (batch, `seq_len`, `seq_len`).
    """

    device = processed_results.device

    # Initialize the reconstructed tensor with -inf placeholders
    reconstructed_tensor = torch.full((batch_size * seq_len * seq_len,),
                                      float("-inf"),
                                      device=device)

    # Map processed results back to the appropriate positions
    reconstructed_tensor[lower_triangle_indices] = processed_results[
        unique_index_mapping]

    # Explicitly set upper triangle indices to -inf (already set, but ensures robustness)
    reconstructed_tensor[upper_triangle_indices] = float("-inf")

    # Reshape to the original (batch, `seq_len`, `seq_len`) format
    return reconstructed_tensor.view(batch_size, seq_len, seq_len)
