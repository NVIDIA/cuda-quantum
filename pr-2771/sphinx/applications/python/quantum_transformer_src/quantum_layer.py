from typing import Any, Optional, Tuple

import cudaq
import numpy as np
import torch
from cudaq import spin
from torch import Tensor, nn
from torch.autograd import Function

from quantum_transformer_src.unitary_library import (
    build_physchem_embeddings_circuit,
    build_sequence_only_circuit,
)
from quantum_transformer_src.utils import prepare_attention_inputs, remove_redundant_circuits, repopulate_tensor


class AttentionQuantumLayer(nn.Module):
    """
    Quantum attention layer for computing token `embeddings` and their pairwise attention weights.

    This layer encodes quantum states representing token `embeddings` and calculates attention
    weights using quantum circuits.


    Args:
        embed_dim (int): Embedding dimension.
        `qpu_count` (int): Number of GPUs for quantum circuit simulations.
        shift (Tensor): The shift value for the quantum gradient calculation if using parameter-shift rule.
        ansatz_layers (int, optional): Number of ansatz layers. Default is 1.
        `num_qubits` (int, optional): Number of working qubits in the quantum circuit. Default is 6.
        conditional_training (bool, optional): Whether to use `physicochemical` `embeddings`. Default is True.
        quantum_gradient_method (`str`, optional): Gradient computation method for quantum circuit training. Default is "`spsa`".
        epsilon (float, optional): Epsilon value for `SPSA`. Default is 0.01.

    Attributes:
        query_angles (`nn.Parameter`): Query circuit parameters.
        key_angles (`nn.Parameter`): Key circuit parameters.
        value (`nn.Linear`): Linear transformation for value `embeddings`.
        projection (`nn.Linear`): Projection layer for attention output.
        dropout (`nn.Dropout`): Dropout layer for regularization.

    Methods:
        forward(x, angles, _): Perform the forward pass.

    """

    def __init__(
        self,
        embed_dim: int,
        qpu_count: int,
        shift: Tensor,
        ansatz_layers: int = 1,
        num_qubits: int = 6,
        conditional_training: bool = True,
        quantum_gradient_method: str = "spsa",
        epsilon: float = 0.01,
    ):
        super(AttentionQuantumLayer, self).__init__()

        self.conditional_training = conditional_training
        self.quantum_gradient_method = quantum_gradient_method
        self.shift = shift
        self.epsilon = epsilon

        # Since this VQC is for generating query and key states the ansatz spans all working qubits
        total_VQC_params = ansatz_layers * num_qubits

        self.quantum_circuit = AttentionQuantumFunction(
            qpu_count=qpu_count,
            num_qubits=num_qubits,
            ansatz_layers=ansatz_layers,
            conditional_training=conditional_training,
        )

        # Query and key parameters (`learnable` quantum circuit parameters)
        self.query_angles = nn.Parameter(torch.zeros(1, 1, total_VQC_params))
        self.key_angles = nn.Parameter(torch.zeros(1, 1, total_VQC_params))

        # Classical components
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                x: Tensor,
                angles: Tensor,
                _: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the quantum attention layer.

        Args:
            x (Tensor): Input tensor of shape (batch, `seq_len`, embed_dim).
            angles (Tensor): Quantum parameter angles.
            _ (Optional[Tensor]): Placeholder for additional inputs (not used), present to have the same number as inputs as the classical self attention module.

        Returns:
            Tuple[Tensor, Tensor]:
                - Output tensor after classically applying the quantum-computed attention scores to the value matrix.
                - Attention weight matrix.
        """

        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # Apply linear transformation to obtain value `embeddings`
        v = self.value(x)

        # Broadcast query and key angles across all tokens in the sequence and batch.

        # We do this because each token in a sequence and batch must be transformed by the same
        # key function and query function. Classically, this is done by matrix multiplication.
        # Each embedding vector z_i in Z undergoes a linear transformation by a weight matrix to give
        # a transformed vector. I.e. f(z_i, W^Q) = q_i. and f(z_i, W^K) = k_i.
        # In the quantum case, instead of W^Q and W^K, we have U_q and U_k, which are determined by the angles below
        broadcasted_query_angles = self.query_angles.expand(
            B, T, -1)  # Expand to (B, T, `total_VQC_params`)
        broadcasted_key_angles = self.key_angles.expand(
            B, T, -1)  # Expand to (B, T, `total_VQC_params`)

        # Retrieve angles for token and position `embeddings`
        tok_angles, pos_angles = angles[:2]

        # We prepare the inputs for the quantum circuit. Each CUDA-Q function call will run a batch of circuits where the input to each circuit is
        # a parameter matrix of shape (parameter_groups, angles).
        # See the utility function 'prepare_attention_inputs' for more details on how the inputs are prepared.

        # circuit_parameters is shape (batch, block_size, block_size, parameter_groups, angles)
        if self.conditional_training:
            physchem_angles = angles[2]
            circuit_parameters = prepare_attention_inputs(
                tok_angles,
                pos_angles,
                broadcasted_query_angles,
                broadcasted_key_angles,
                physchem_angles,
            )
        else:

            circuit_parameters = prepare_attention_inputs(
                tok_angles, pos_angles, broadcasted_query_angles,
                broadcasted_key_angles)

        # Scaling factor for stable dot products
        # Note that it is not 1/`sqrt`(`dk`) and is instead `sqrt`(`dk`). We need to maintain a variance of 1 on a `hypersphere`.
        scale_factor = np.sqrt(C)

        # The output of AttentionQuantumFunction is batch of already masked attention matrices (pairwise dot products).
        attn_weight = (AttentionQuantumFunction.apply(
            circuit_parameters,
            self.shift,
            self.quantum_circuit,
            B,
            T,
            self.quantum_gradient_method,
            self.epsilon,
        ) * scale_factor)

        attn_weight = torch.softmax(attn_weight, dim=-1)

        # Apply attention weights to the value vectors
        y = self.projection(torch.matmul(attn_weight, v))

        return y, attn_weight


class AttentionQuantumFunction(Function):
    """
    Custom PyTorch `autograd` function for computing quantum attention scores.

    The forward function runs quantum circuits to compute attention weights, while
    the backward function calculates gradients using either SPSA or the parameter-shift rule.

    Args:
        `qpu_count` (int): Number of quantum processing units (`QPUs`).
        `num_qubits` (int, optional): Number of working qubits in the circuit. Default is 6.
        ansatz_layers (int, optional): Number of ansatz layers. Default is 1.
        conditional_training (bool, optional): Whether to use `physicochemical` `embeddings`. Default is True.

    Attributes:
        quantum_circuit (Callable): Function to construct and execute quantum circuits.
        `hamiltonian` (torch.Tensor): Observable for quantum measurement.
    """

    def __init__(
        self,
        qpu_count: int,
        num_qubits: int = 6,
        ansatz_layers: int = 1,
        conditional_training: bool = True,
    ):
        self.qpu_count = qpu_count
        self.ansatz_layers = ansatz_layers
        self.num_qubits = num_qubits
        self.hamiltonian = spin.z(0)
        self.conditional_training = conditional_training

        self.build_circuit = (build_physchem_embeddings_circuit
                              if conditional_training else
                              build_sequence_only_circuit)

    def run(self, parameters):
        """Runs the batch of quantum circuits and returns the expectation values."""
        device = parameters.device

        def param_splits(x):
            """Split input parameters across GPUs."""
            return np.array_split(x.cpu().numpy(), self.qpu_count)

        token_1, position_1, token_2, position_2 = map(
            param_splits,
            (
                parameters[:, 0, :],
                parameters[:, 1, :],
                parameters[:, 4, :],
                parameters[:, 5, :],
            ),
        )
        (
            query_token_register,
            query_pos_register,
            key_token_register,
            key_pos_register,
        ) = map(
            lambda x: x.cpu().numpy(),
            (
                parameters[:, 2, :],
                parameters[:, 3, :],
                parameters[:, 6, :],
                parameters[:, 7, :],
            ),
        )

        if self.conditional_training:
            query_physchem_register, key_physchem_register = (
                parameters[:, 8, :].cpu().numpy(),
                parameters[:, 9, :].cpu().numpy(),
            )
            physchem = param_splits(parameters[:, 10, :])

            query_angles = np.array_split(
                np.concatenate(
                    (query_token_register, query_pos_register,
                     query_physchem_register),
                    axis=1,
                ),
                self.qpu_count,
            )
            key_angles = np.array_split(
                np.concatenate(
                    (key_token_register, key_pos_register,
                     key_physchem_register),
                    axis=1,
                ),
                self.qpu_count,
            )
        else:
            query_angles = np.array_split(
                np.concatenate((query_token_register, query_pos_register),
                               axis=1),
                self.qpu_count,
            )
            key_angles = np.array_split(
                np.concatenate((key_token_register, key_pos_register), axis=1),
                self.qpu_count,
            )

        # Since the number of ansatz layers and number of working qubits are arguments to the quantum circuit, we must broadcast them to all circuits
        ansatz_layers = np.array_split(
            np.expand_dims(
                np.full((parameters.shape[0],), self.ansatz_layers, dtype=int),
                1),
            self.qpu_count,
        )
        num_working_qubits = np.array_split(
            np.expand_dims(
                np.full((parameters.shape[0],), self.num_qubits, dtype=int), 1),
            self.qpu_count,
        )

        asyncresults = []
        for i in range(self.qpu_count):
            for j in range(token_1[i].shape[0]):
                if self.conditional_training:
                    asyncresults.append(
                        cudaq.observe_async(
                            self.build_circuit,
                            self.hamiltonian,
                            token_1[i][j, :],
                            position_1[i][j, :],
                            physchem[i][j, :],
                            query_angles[i][j, :],
                            token_2[i][j, :],
                            position_2[i][j, :],
                            key_angles[i][j, :],
                            ansatz_layers[i][j],
                            num_working_qubits[i][j, :],
                            qpu_id=i,
                        ))
                else:
                    asyncresults.append(
                        cudaq.observe_async(
                            self.build_circuit,
                            self.hamiltonian,
                            token_1[i][j, :],
                            position_1[i][j, :],
                            query_angles[i][j, :],
                            token_2[i][j, :],
                            position_2[i][j, :],
                            key_angles[i][j, :],
                            ansatz_layers[i][j],
                            num_working_qubits[i][j, :],
                            qpu_id=i,
                        ))

        expectations = torch.tensor(
            [r.get().expectation() for r in asyncresults], device=device)
        return expectations

    @staticmethod
    def forward(
        ctx,
        parameters: Tensor,
        shift: Tensor,
        quantum_circuit: "AttentionQuantumFunction",
        batch_size: int,
        block_size: int,
        quantum_gradient_method: str = "spsa",
        epsilon: float = 0.01,
    ) -> Tensor:
        """Forward pass to compute quantum attention scores."""

        # Remove upper triangle circuits and circuits with duplicate parameters
        (
            cleaned_circuit_parameters,
            unique_index_mapping,
            lower_triangle_indices,
            upper_triangle_indices,
        ) = remove_redundant_circuits(parameters)

        # Save for backward pass
        ctx.save_for_backward(
            cleaned_circuit_parameters,
            unique_index_mapping,
            lower_triangle_indices,
            upper_triangle_indices,
        )
        ctx.shift, ctx.quantum_circuit, ctx.batch_size, ctx.block_size = (
            shift,
            quantum_circuit,
            batch_size,
            block_size,
        )
        ctx.quantum_gradient_method, ctx.epsilon = quantum_gradient_method, epsilon

        expectations = quantum_circuit.run(cleaned_circuit_parameters)
        return repopulate_tensor(
            batch_size,
            block_size,
            expectations,
            unique_index_mapping,
            lower_triangle_indices,
            upper_triangle_indices,
        ).to(cleaned_circuit_parameters.device)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Any
    ) -> Tuple[Optional[Tensor], None, None, None, None, None, None, None]:
        """Backward pass to compute gradients using SPSA or parameter-shift method."""
        (
            cleaned_circuit_parameters,
            unique_index_mapping,
            lower_triangle_indices,
            upper_triangle_indices,
        ) = ctx.saved_tensors
        shift, quantum_circuit, batch_size, block_size = (
            ctx.shift,
            ctx.quantum_circuit,
            ctx.batch_size,
            ctx.block_size,
        )
        quantum_gradient_method, epsilon = ctx.quantum_gradient_method, ctx.epsilon

        _, groups, param_count = cleaned_circuit_parameters.shape

        device = cleaned_circuit_parameters.device
        gradients = torch.zeros_like(cleaned_circuit_parameters)

        if quantum_gradient_method == "spsa":
            # Generate random perturbations (delta) for all parameters
            delta = (torch.randint(
                0, 2, cleaned_circuit_parameters.shape, device=device).float() *
                     2 - 1) * epsilon  # Random +/- epsilon

            # Perturb parameters
            params_plus, params_minus = (
                cleaned_circuit_parameters + delta,
                cleaned_circuit_parameters - delta,
            )

            # Compute expectation values at the perturbed points
            with torch.no_grad():
                # Concatenate the + and - perturbations to run them all in parallel
                exp_concat = quantum_circuit.run(
                    torch.cat((params_plus, params_minus), dim=0))
                num_circuits = params_plus.size(0)
                exp_plus = exp_concat[:num_circuits]
                exp_minus = exp_concat[num_circuits:]

            # Compute gradient per unique circuit
            exp_diff = ((exp_plus - exp_minus).unsqueeze(-1).unsqueeze(-1)
                       )  # Shape: (number_of_unique_circuits, 1, 1)
            spsa_gradient_unique = exp_diff.to(device) / (
                2 * delta
            )  # Shape: (number_of_unique_circuits, `param_group`, `param_per_group`)

            # Initialize a tensor to hold all gradients
            gradients_flat = torch.zeros(
                (
                    batch_size * block_size * block_size,
                    cleaned_circuit_parameters.shape[1],
                    cleaned_circuit_parameters.shape[2],
                ),
                device=device,
            )

            # Map gradients back to the full tensor
            gradients_flat[lower_triangle_indices] = spsa_gradient_unique[
                unique_index_mapping]

            # Reshape the gradients tensor to match the expected dimensions
            gradients = gradients_flat.view(
                batch_size * block_size * block_size,
                cleaned_circuit_parameters.shape[1],
                cleaned_circuit_parameters.shape[2],
            )

        elif quantum_gradient_method == "parameter-shift":
            # Parameter-shift implementation
            shift_right_tensors = []
            shift_left_tensors = []

            for i in range(groups):
                for j in range(param_count):
                    shift_right = cleaned_circuit_parameters.clone()
                    shift_right[:, i, j] += shift

                    shift_left = cleaned_circuit_parameters.clone()
                    shift_left[:, i, j] -= shift

                    shift_right_tensors.append(shift_right)
                    shift_left_tensors.append(shift_left)

            all_shift_right = torch.stack(shift_right_tensors).reshape(
                -1, groups, param_count)
            all_shift_left = torch.stack(shift_left_tensors).reshape(
                -1, groups, param_count)

            with torch.no_grad():
                all_grad_expectation_right = quantum_circuit.run(
                    all_shift_right).reshape(groups * param_count, -1)
                all_grad_expectation_left = quantum_circuit.run(
                    all_shift_left).reshape(groups * param_count, -1)

            index = 0
            for i in range(groups):
                for j in range(param_count):
                    # Unstack to get individual expectation values
                    exp_right = all_grad_expectation_right[index]
                    exp_left = all_grad_expectation_left[index]

                    # Repopulate tensors and remove mask for proper gradient calculation
                    repopulated_right = repopulate_tensor(
                        batch_size,
                        block_size,
                        exp_right,
                        unique_index_mapping,
                        lower_triangle_indices,
                        upper_triangle_indices,
                    )
                    repopulated_right[repopulated_right == float("-inf")] = 0

                    repopulated_left = repopulate_tensor(
                        batch_size,
                        block_size,
                        exp_left,
                        unique_index_mapping,
                        lower_triangle_indices,
                        upper_triangle_indices,
                    )
                    repopulated_left[repopulated_left == float("-inf")] = 0

                    # Compute gradients for each parameter
                    gradients[:, i, j] = 0.5 * (repopulated_right.reshape(-1) -
                                                repopulated_left.reshape(-1))

                    index += 1

        # Reshape grad_output to match the dimensions of gradients
        grad_output = grad_outputs[0]
        grad_output_expanded = grad_output.view(-1, 1, 1).expand_as(gradients)
        final_gradients = (grad_output_expanded * gradients).view(
            batch_size, block_size, block_size, groups, param_count)

        return (final_gradients, None, None, None, None, None, None, None)
