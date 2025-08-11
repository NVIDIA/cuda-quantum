import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch import Tensor, nn
from torch.utils.data import Dataset

from quantum_transformer_src.quantum_layer import AttentionQuantumLayer
from quantum_transformer_src.utils import get_physchem_properties, scale_to_range


class Transformer_Dataset(Dataset):
    """
    `Dataset` class for the Transformer model.

    Args:
        data_path (`str`): The path to the data file.
        block_size (int): The block size for padding the sequences. If set to None, the block size will be the maximum sequence length in the `dataset`.

    Attributes:
        smiles (pd.Series): The SMILES strings in the `dataset`.
        smiles_regex (re.Pattern): The regular expression for `tokenizing` SMILES strings.
        vocab (list): The vocabulary for the `dataset`.
        `stoi` (dict): A mapping of characters to their corresponding indices in the vocabulary.
        `itos` (dict): A mapping of indices to their corresponding characters in the vocabulary.
        block_size (int): The block size for padding the sequences.

    Methods:
        __len__(): Returns the length of the `dataset`.
        __getitem__(idx): Returns the item at the given index.
        `tokenize_smiles`(smiles): `Tokenizes` any SMILES string.

    """

    def __init__(self, data_path=None, block_size=22):

        # Get the SMILES strings from the data file
        data = pd.read_csv(data_path)
        smiles_col = "smiles" if "smiles" in data.columns else "SMILES"

        # Method to canonicalize the SMILES strings
        def canonicalize(smiles):
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol) if mol else None

        # Canonicalize the SMILES strings and remove any duplicates
        data[smiles_col] = data[smiles_col].apply(canonicalize)
        data = data.dropna(subset=[smiles_col]).drop_duplicates(
            subset=[smiles_col])
        data = data.reset_index(drop=True)

        self.smiles = data[smiles_col]

        # Tokenize the SMILES strings using a regular expression
        self.smiles_regex = re.compile(
            "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|<pad>|[CLS]|[EOS])"
        )
        special_tokens = {"<pad>", "[CLS]", "[EOS]"}

        # Build the vocabulary
        characters = {
            ch for smile in self.smiles
            for ch in self.smiles_regex.findall(smile.strip())
        }
        self.vocab = sorted(list(special_tokens | characters))

        # Create mappings for the vocabulary
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

        if block_size is None:
            print(
                "Warning: May need to adjust tokenization rules if using a dataset other than QM9."
            )
            max_seq_length = max(
                len(self.smiles_regex.findall("[CLS]" + smile.strip() +
                                              "[EOS]"))
                for smile in self.smiles)
            self.block_size = max_seq_length
        else:
            # Add 2 for [`CLS`] and [`EOS`]
            self.block_size = 2 + block_size

    # Method to return the length of the `dataset`
    def __len__(self):
        return len(self.smiles)

    # Method to return the item at the given index
    def __getitem__(self, idx):

        # Tokenize the SMILES string and pad it to the block size
        smiles = "[CLS]" + self.smiles[idx].strip() + "[EOS]"
        smiles_tokens = self.smiles_regex.findall(smiles)
        smiles += "<pad>" * (self.block_size - len(smiles_tokens))

        input_sequence = [
            self.stoi[s] for s in self.smiles_regex.findall(smiles)[:-1]
        ]
        target_sequence = [
            self.stoi[s] for s in self.smiles_regex.findall(smiles)[1:]
        ]

        physchem_properties = get_physchem_properties(self.smiles[idx].strip())

        return (
            torch.tensor(input_sequence, dtype=torch.long),
            torch.tensor(target_sequence, dtype=torch.long),
            torch.tensor(physchem_properties, dtype=torch.float),
        )

    def tokenize_smiles(self, smiles):
        """
        `Tokenize` any SMILES string using the same logic as the `dataset`.
        This if useful for getting at attention maps for any given SMILES string.

        Args:
            smiles (`str`): A valid SMILES string.

        Returns:
            list of int: `Tokenized` SMILES indices.
        """
        smiles = "[CLS]" + smiles.strip() + "[EOS]"
        smiles_tokens = self.smiles_regex.findall(smiles)
        smiles += "<pad>" * (self.block_size - len(smiles_tokens))
        return [
            self.stoi.get(s, self.stoi["<pad>"])
            for s in self.smiles_regex.findall(smiles)[:-1]
        ]


class Transformer_Model(nn.Module):
    """
    Transformer model for sequence generation, supporting both classical and quantum attention mechanisms.

    Args:
        `qpu_count` (int): Number of GPUs for quantum circuit simulations.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
        block_size (int): Maximum sequence length.
        `physchem_dim` (int): Dimension of `physicochemical` properties.
        classical_attention (bool): Use classical attention if True, else quantum attention.
        `num_qubits` (int): Number of working qubits in the quantum circuit.
        ansatz_layers (int): Layers in the quantum ansatz.
        conditional_training (bool): Include `physicochemical` `embeddings` in training.
        `use_spsa` (bool): Method for optimizing quantum circuits (default: '`spsa`').
        epsilon (float): Epsilon value for `SPSA` (if used).
        classical_parameter_reduction (bool): Reduce classical parameters to match quantum parameter count.

    Attributes:
        token_embed_quantum_parameters (`nn.Embedding`): Token `embedding` quantum parameters (angles in the PQC).
        position_embed_quantum_parameters (`nn.Parameter`): Position `embedding` quantum parameters (angles in the PQC).
        `physchem_embed_quantum_parameters` (`nn.Linear`): `Physicochemical` `embedding` quantum parameters (angles in the PQC).
        token_embed (`nn.Embedding`): Classical token `embeddings`.
        position_embed (`nn.Parameter`): Classical position `embeddings`.
        `physchem_embed` (`nn.Linear`, optional): Classical `physicochemical` `embeddings`.
        reduced_token_embed (`nn.Embedding`): The classical token embedding with a number of parameters that is equivalent to the number of quantum embedding parameters that represent the token `embeddings`.
        reduced_position_embed (`nn.Parameter`): The classical positional embedding with a number of parameters that is equivalent to the number of quantum embedding parameters that represent the positional `embeddings`.
        `reduced_physchem_embed` (`nn.Linear`): The classical `physicochemical` embedding with a number of parameters that is equivalent to the number of quantum embedding parameters that represent the `physicochemical` `embeddings`.
        dropout (`nn.Dropout`): Dropout layer.
        block (`nn.Module`): Transformer block.
        layer_norm (`nn.LayerNorm`): Layer normalization.
        output (`nn.Linear`): Final output projection layer.

    Methods:
        forward(`idx`, `physchem_props`): Forward pass of the model.

    """

    def __init__(
        self,
        qpu_count: int,
        vocab_size: int,
        embed_dim: int = 64,
        block_size: int = 22,
        physchem_dim: int = 9,
        classical_attention: bool = True,
        num_qubits: int = 6,
        ansatz_layers: int = 1,
        conditional_training: bool = False,
        quantum_gradient_method: str = "spsa",
        epsilon: float = 0.01,
        classical_parameter_reduction: bool = False,
    ):
        super().__init__()

        self.classical_attention = classical_attention
        self.conditional_training = conditional_training
        self.classical_parameter_reduction = (
            False if not classical_attention else classical_parameter_reduction)

        # Currently, to best align with the classical transformer method where the embedding dimension of
        # token and position `embeddings` are the same, we set the number of qubits for token and position `embeddings` to be the same as well
        # Thus if only learning sequences (no additional `embeddings`; only token and position), the number of qubits in each register is the number of working qubits is divided by 2
        # If we are using conditional training, we divide the number of qubits by 3 to account for the additional `physicochemical` `embeddings`
        num_qubits_per_register = (num_qubits // 2 if not conditional_training
                                   else num_qubits // 3)

        if not self.classical_attention:
            self._initialize_quantum_embeddings(
                vocab_size,
                ansatz_layers,
                num_qubits_per_register,
                block_size,
                physchem_dim,
            )

        # Classical `embeddings` (always present, regardless of attention type for use in the value matrix)
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Parameter(torch.zeros(1, block_size,
                                                       embed_dim))

        if conditional_training:
            self.physchem_embed = nn.Linear(physchem_dim, embed_dim)

        # Optional classical parameter reduction of the `embeddings` which will propagate to query/key matrices to match quantum parameter count
        if self.classical_parameter_reduction:
            self._initialize_reduced_classical_embeddings(
                vocab_size,
                ansatz_layers,
                num_qubits_per_register,
                block_size,
                physchem_dim,
            )

        # Transformer architecture components
        self.dropout = nn.Dropout(0.1)
        self.block = Transformer_Block(
            embed_dim=embed_dim,
            classical_attention=classical_attention,
            conditional_training=conditional_training,
            num_qubits=num_qubits,
            ansatz_layers=ansatz_layers,
            quantum_gradient_method=quantum_gradient_method,
            epsilon=epsilon,
            classical_parameter_reduction=self.classical_parameter_reduction,
            qpu_count=qpu_count,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def _initialize_quantum_embeddings(
        self,
        vocab_size,
        ansatz_layers,
        num_qubits_per_register,
        block_size,
        physchem_dim,
    ):
        """
        Initializes quantum parameter `embeddings`.

        Notes:
            - We use a layer of Ry gates followed by circular CNOTs. Thus, the number of parameters for a given quantum embedding is
              equal to the number of qubits representing the individual token/position/`physchem` register multiplied by the number of those ansatz layers.

            - This matrix is min/max scaled at the model's initialization to help the quantum states
              representing tokens span as much of the Hilbert space as possible so that they can be easily distinguished

            - The position embedding is initialized to 0, which gives us the identity matrix as the initial unitary preparing the positional information

        """

        self.token_embed_quantum_parameters = nn.Embedding(
            vocab_size, ansatz_layers * num_qubits_per_register)
        self._scale_quantum_parameters(
            self.token_embed_quantum_parameters.weight)

        self.position_embed_quantum_parameters = nn.Parameter(
            torch.zeros(1, block_size, ansatz_layers * num_qubits_per_register))

        if self.conditional_training:
            self.physchem_embed_quantum_parameters = nn.Linear(
                physchem_dim, ansatz_layers * num_qubits_per_register)

    def _scale_quantum_parameters(self, tensor):
        """Scales quantum parameters for better Hilbert space coverage."""
        with torch.no_grad():
            min_val, max_val = tensor.min(), tensor.max()
            scaled_weights = (tensor - min_val) / (max_val - min_val) * np.pi
            tensor.copy_(scaled_weights)

    def _initialize_reduced_classical_embeddings(
        self,
        vocab_size,
        ansatz_layers,
        num_qubits_per_register,
        block_size,
        physchem_dim,
    ):
        """Initializes classical `embeddings` with reduced dimensionality to match quantum model parameter count."""
        self.reduced_token_embed = nn.Embedding(
            vocab_size, ansatz_layers * num_qubits_per_register)
        self.reduced_position_embed = nn.Parameter(
            torch.zeros(1, block_size, ansatz_layers * num_qubits_per_register))

        if self.conditional_training:
            self.reduced_physchem_embed = nn.Linear(
                physchem_dim, ansatz_layers * num_qubits_per_register)

    def forward(
        self,
        idx: torch.Tensor,
        physchem_props: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass through the model.

        Args:
            `idx` (torch.Tensor): Tensor containing token indices of shape (B, T).
            `physchem_props` (Optional[torch.Tensor]): Tensor containing `physicochemical` properties if conditional training is used.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output `logits` and attention weights.

        Notes:
            - If we want to run the classical (`eq`) model with the same number of parameters as the quantum model,
              we need token and position embedding vectors with a reduced dimensionality.

            - Since the value matrix in the attention mechanism is computed classically in all models,
              we need to create a separate set of token/position/`physicochemical` `embeddings` with reduced dimensionality
              so they can propagate to the reduced dimensionality query/key matrices.
        """

        # Get the batch size and sequence length
        B, T = idx.size()

        if self.conditional_training:

            # If we are using conditional training, we also need to define the `physicochemical` `embeddings`
            # The `physicochemical` `embeddings` for the molecules are repeated for each token in the sequence with `unsqueeze` and expand
            physchem_embeddings = (
                self.physchem_embed(physchem_props).unsqueeze(1).expand(
                    -1, T, -1))

            # Add token, position, `physicochemical` `embeddings` together to get the combined `embeddings`
            x = self.dropout(
                self.token_embed(idx) + self.position_embed[:, :T, :] +
                physchem_embeddings)

            # If we are running the classical (`eq`) model with conditions, we do the same as above but with reduced dimensionality `embeddings`
            reduced_physchem_embeddings = (
                self.reduced_physchem_embed(physchem_props).unsqueeze(1).expand(
                    -1, T, -1) if self.classical_parameter_reduction else None)
            reduced_x = (self.dropout(
                self.reduced_token_embed(idx) +
                self.reduced_position_embed[:, :T, :] +
                reduced_physchem_embeddings)
                         if self.classical_parameter_reduction else None)

        else:
            # If we are not using conditional training, we only need to add the token and position `embeddings`
            x = self.dropout(
                self.token_embed(idx) + self.position_embed[:, :T, :])
            reduced_x = (self.dropout(
                self.reduced_token_embed(idx) +
                self.reduced_position_embed[:, :T, :])
                         if self.classical_parameter_reduction else None)

        angles = None

        # If we are using quantum `embeddings`, we need to prepare the inputs for the quantum circuits
        if not self.classical_attention:

            # Ensure that all batches share the same position embedding values
            position_embedding_angles = self.position_embed_quantum_parameters[:, :T, :].expand(
                B, T, self.position_embed_quantum_parameters.shape[-1])

            # All the angles are stacked so they can be easily restructured, prepped, and fed into the custom CUDA-Q functions
            angles = torch.stack([
                self.token_embed_quantum_parameters(idx),
                position_embedding_angles
            ])

            if self.conditional_training:

                # We scale the molecular properties between 0 and pi to help the model easily differentiate
                # between different `physicochemical` properties rather than them being concentrated in a small Hilbert subspace
                physchem_embeddings_angles = scale_to_range(
                    self.physchem_embed_quantum_parameters(
                        physchem_props).unsqueeze(1).expand(-1, T, -1),
                    0,
                    torch.pi,
                )
                angles = torch.cat(
                    [angles, physchem_embeddings_angles.unsqueeze(0)], dim=0)

        # Perform the forward pass through the transformer block
        x, attn_weight = self.block(x, angles, reduced_x)

        # Apply layer normalization
        x = self.layer_norm(x)

        # Generate the output `logits`
        logits = self.output(x)

        return logits, attn_weight


class Transformer_Block(nn.Module):
    """
    Transformer decoder layer with support for both classical and quantum attention.

    Args:
        `qpu_count` (int): Number of GPUs for quantum circuit simulations.
        embed_dim (int): Embedding dimension.
        classical_attention (bool): Use classical attention if True, else quantum attention.
        `num_qubits` (int): Number of working qubits in quantum circuits.
        ansatz_layers (int): Layers in the quantum ansatz.
        conditional_training (bool): Include `physicochemical` `embeddings` in training.
        quantum_gradient_method (`str`): Method for optimizing quantum circuits (default: '`spsa`').
        epsilon (float): Epsilon value for `SPSA` optimization.
        classical_parameter_reduction (bool): Reduce classical parameters to match quantum parameter count.

    Attributes:
        layer_norm1 (`nn.LayerNorm`): First layer normalization.
        layer_norm2 (`nn.LayerNorm`): Second layer normalization.
        attention (`nn.Module`): Attention module (classical or quantum).
        `mlp` (`nn.Sequential`): Multi-layer `perceptron` module.

    Methods:
        forward(x, angles, reduced_x): Forward pass of the transformer block.

    """

    def __init__(
        self,
        qpu_count: int,
        embed_dim: int = 64,
        classical_attention: bool = True,
        num_qubits: int = 6,
        ansatz_layers: int = 1,
        conditional_training: bool = True,
        quantum_gradient_method: str = "spsa",
        epsilon: float = 0.01,
        classical_parameter_reduction: bool = False,
    ):
        super().__init__()

        self.classical_attention = classical_attention
        self.classical_parameter_reduction = (classical_parameter_reduction
                                              if classical_attention else False)

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        """
        Notes on choosing the dimensions to ensure an equal number of parameters between the classical and quantum models:

        - The dimension of Z is (n x d). If we are trying to have each token (z_i) be equal in parameter
          count to the quantum model, embedding dimension d must be ansatz_layers*`num_qubits_per_register` 
          (this is dependent on the fact our ansatz structure is a single Ry gate on each qubit for each layer)
         
        - We choose an output dimension of classical Q and K such that the dimensions of the weight matrices to
          in the self-attention mechanism will be equal to the number of quantum parameters in U_q and U_k.
         
        - The number of parameters in U_q and U_k = `total_num_qubits`*layers, thus to achieve the same number of parameters 
          in W^Q and W^K (`eq` 3) as in U_q and U_k, W^Q and W^K must be shape (reduced_x_dim, `reduced_qk_dim`), 
          where reduced_x_dim times `reduced_qk_dim` is equal to `total_num_qubits`*layers
         
        - Solving for `reduced_qk_dim`, it is (`total_num_qubits`*layers)/(reduced_x_dim) which is equal to (`total_num_qubits`*layers)/(layers*`num_qubits_per_register`)
         
        - Then we can see that `reduced_qk_dim` = `total_num_qubits`/`num_qubits_per_register` where the total number of qubits 
          is the `num` of `num_tok_qubits`, `num_pos_qubits`, and `num_physchem_qubits` if conditional_training==True
         
        - Thus, the `reduced_qk_dim` is 2 if doing sequence-only training and 3 if doing condition-based training
        """

        num_qubits_per_register = num_qubits // (3
                                                 if conditional_training else 2)
        reduced_x_dim = (ansatz_layers * num_qubits_per_register
                         if self.classical_parameter_reduction else None)
        reduced_qk_dim = (2 + int(conditional_training)
                          if self.classical_parameter_reduction else None)

        # Initialize attention mechanism
        self.attention = (Self_Attention(
            embed_dim=embed_dim,
            reduced_x_dim=reduced_x_dim,
            reduced_qk_dim=reduced_qk_dim,
        ) if self.classical_attention else AttentionQuantumLayer(
            qpu_count=qpu_count,
            embed_dim=embed_dim,
            shift=torch.tensor(torch.pi / 2),
            ansatz_layers=ansatz_layers,
            num_qubits=num_qubits,
            conditional_training=conditional_training,
            quantum_gradient_method=quantum_gradient_method,
            epsilon=epsilon,
        ))

        # Multi-layer `perceptron`
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x, angles, reduced_x):

        y, attn_weight = self.attention(self.layer_norm1(x), angles, reduced_x)
        x = x + y
        x = x + self.mlp(self.layer_norm2(x))
        return x, attn_weight


class Self_Attention(nn.Module):
    """
    Classical self-attention module.

    Args:
        embed_dim (int): Embedding dimension.
        reduced_x_dim (Optional[int]): Dimension of reduced input representation, if applicable.
        `reduced_qk_dim` (Optional[int]): Dimension of query and key vectors in reduced form.
        n_heads (int): Number of attention heads. 1 in this work.

    Attributes:
        query (`nn.Linear`): Linear layer for query projection.
        key (`nn.Linear`): Linear layer for key projection.
        value (`nn.Linear`): Linear layer for value projection.
        projection (`nn.Linear`): Final projection layer for attention output.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        reduced_x_dim: Optional[int] = None,
        reduced_qk_dim: Optional[int] = None,
        n_heads: int = 1,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.reduced_qk_dim = reduced_qk_dim

        # Define query and key projections, considering reduced dimensions
        qk_dim = reduced_qk_dim if reduced_qk_dim else embed_dim
        self.query = nn.Linear(reduced_x_dim or embed_dim, qk_dim, bias=False)
        self.key = nn.Linear(reduced_x_dim or embed_dim, qk_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: Tensor,
        angles: Optional[Tensor] = None,
        reduced_x: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the self-attention module.

        Args:
            x (Tensor): Input tensor of shape (batch, `seq_len`, embed_dim).
            angles (Optional[Tensor]): Unused parameter (for API consistency).
            reduced_x (Optional[Tensor]): Reduced representation of `x`, if applicable.

        Returns:
            tuple[Tensor, Tensor]:
                - Output tensor of shape (batch, `seq_len`, embed_dim).
                - Attention weight matrix of shape (batch, n_heads, `seq_len`, `seq_len`).
        """

        B, T, C = x.shape
        qk_x = reduced_x if reduced_x is not None else x

        # Compute query, key, and value vectors
        q = self.query(qk_x).view(B, T, self.n_heads, -1)
        k = self.key(qk_x).view(B, T, self.n_heads, -1)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        scale_factor = 1 / np.sqrt(C // self.n_heads)
        attn_bias = torch.zeros(T, T, dtype=q.dtype, device=q.device)

        # Create a mask for the upper triangular part of the attention matrix
        temp_mask = torch.ones(T, T, dtype=torch.bool,
                               device=q.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        # Compute the attention weights
        attn_weight = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)

        # Multiply the attention weights by the value vectors
        y = torch.matmul(attn_weight, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Map the output to the embedding dimension
        return self.projection(y), attn_weight
