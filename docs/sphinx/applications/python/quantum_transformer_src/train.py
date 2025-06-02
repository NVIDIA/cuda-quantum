import csv
import logging
import os
from datetime import datetime
from typing import Optional, Tuple

import cudaq
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import random_split
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from quantum_transformer_src.transformer import Transformer_Dataset, Transformer_Model
from quantum_transformer_src.utils import get_physchem_properties

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def train_transformer(
    training_data: str,
    checkpoint_dir: str,
    checkpoint_resume_path: Optional[str] = None,
    learning_rate: float = 0.005,
    weight_decay: float = 0.1,
    batch_size: int = 256,
    epochs: int = 20,
    save_every_n_batches: int = 0,
    validation_split: float = 0.05,
    attn_type: str = "classical",
    num_qubits: int = 6,
    ansatz_layers: int = 1,
    conditional_training: bool = True,
    quantum_gradient_method: str = "spsa",
    spsa_epsilon: float = 0.01,
    sample_percentage: float = 1.0,
    seed: int = 42,
    classical_parameter_reduction: bool = False,
    device: str = "gpu",
    qpu_count: int = -1,
) -> None:
    """
    Train a transformer model with either classical or quantum attention.

    Args:
        training_data (`str`): Path to the training data CSV file.
        `checkpoint_dir` (`str`): Directory to save model checkpoints.
        checkpoint_resume_path (Optional[`str`]): Path to resume training from a checkpoint.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.
        save_every_n_batches (int): Frequency to save batch-level model checkpoints.
        validation_split (float): Fraction of data used for validation.
        attn_type (`str`): Use classical attention or quantum attention ('classical' or 'quantum').
        `num_qubits` (int): Number of working qubits in the quantum attention layer.
        ansatz_layers (int): Number of layers in the quantum ansatz.
        conditional_training (bool): Whether to train with `physicochemical` properties.
        quantum_gradient_method (`str`): Quantum gradient method ('`spsa`' or 'parameter-shift').
        `spsa_epsilon` (float): Epsilon value for `SPSA` optimization.
        sample_percentage (float): Fraction of `dataset` used for training.
        seed (int): Random seed for `reproducibility`.
        classical_parameter_reduction (bool): Ensure the number of classical parameters is equal to the number of quantum parameters.
        device (`str`): Device for training, either '`cpu`' or '`gpu`'.
        `qpu_count` (int): Number of GPUs to use (-1 = all available GPUs).

    Returns:
        None
    """

    classical_attention = True if attn_type == "classical" else False
    train_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    training_configuration = {
        "training_data": training_data,
        "checkpoint_path": checkpoint_dir,
        "checkpoint_resume_path": checkpoint_resume_path,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "save_every_n_batches": save_every_n_batches,
        "validation_split": validation_split,
        "device": device,
        "classical_attention": classical_attention,
        "num_qubits": num_qubits,
        "ansatz_layers": ansatz_layers,
        "conditional_training": conditional_training,
        "quantum_gradient_method": quantum_gradient_method,
        "spsa_epsilon": spsa_epsilon,
        "sample_percentage": sample_percentage,
        "seed": seed,
        "classical_parameter_reduction": classical_parameter_reduction,
        "qpu_count": qpu_count,
        "train_id": train_id,
    }

    # Check if there are any discrepancies in the training configuration if resuming from a checkpoint
    assert check_training_configuration(
        checkpoint_resume_path,
        training_configuration), "Training configuration mismatch."

    # Set random seeds for `reproducibility`
    set_random_seeds(seed)

    # Ensure requested parameters are valid
    validate_parameters(
        sample_percentage,
        device,
        quantum_gradient_method,
        num_qubits,
        conditional_training,
    )

    # Prepare checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory set at: {checkpoint_dir}")

    # Configure quantum target and number of GPUs to use
    qpu_count = configure_quantum_target(device, qpu_count)

    # Set device for PyTorch
    torch_device: torch.device = torch.device("cuda:0" if (
        device == "gpu" and torch.cuda.is_available()) else "cpu")
    logger.info(f"Using device: {torch_device}")

    # Ensure training data is available
    ensure_training_data(training_data)

    # Load and split `dataset`
    train_dataset, val_dataset, dataset = prepare_datasets(
        training_data, sample_percentage, validation_split, seed)

    # Save SMILES strings
    save_smiles_strings(train_dataset, val_dataset, dataset, train_id)

    # Initialize model
    model = initialize_model(
        dataset,
        torch_device,
        classical_attention,
        num_qubits,
        ansatz_layers,
        conditional_training,
        quantum_gradient_method,
        spsa_epsilon,
        classical_parameter_reduction,
        qpu_count,
        checkpoint_resume_path,
    )

    # Initialize optimizer, `scaler`, and loss function
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    loss_function = CrossEntropyLoss(ignore_index=-1)
    logger.info("Optimizer, scaler, and loss function initialized.")

    # Load checkpoint if provided, otherwise initializes the empty training metrics
    starting_epoch, start_batch, metrics, train_id = checkpoint_resume_path_function(
        checkpoint_resume_path, model, optimizer, scaler, train_id)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset,
                                                   batch_size)

    if checkpoint_resume_path:
        train_loader.load_state_dict(
            torch.load(checkpoint_resume_path,
                       map_location="cpu",
                       weights_only=False)["train_loader_state"])

    # Start training loop
    logger.info("Starting training...")
    train_model(
        training_configuration,
        model,
        optimizer,
        scaler,
        loss_function,
        train_loader,
        val_loader,
        epochs,
        starting_epoch,
        start_batch,
        metrics,
        checkpoint_dir,
        save_every_n_batches,
        torch_device,
    )


def set_random_seeds(seed: int) -> None:
    """Set random seeds for `reproducibility`."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "42"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    logger.debug(f"Random seeds set to: {seed}")
    logger.info(
        "Note reproducability between device architectures is not garunteed.")


def validate_parameters(
    sample_percentage: float,
    device: str,
    quantum_gradient_method: str,
    num_qubits: int,
    conditional_training: bool,
) -> None:
    """Validate input parameters."""
    if not (0 < sample_percentage <= 1.0):
        raise ValueError("Sample percentage must be between 0 and 1.")
    if device not in {"cpu", "gpu"}:
        raise ValueError("Device must be either 'cpu' or 'gpu'.")
    if quantum_gradient_method not in {"spsa", "parameter-shift"}:
        raise ValueError(
            "Quantum gradient method must be 'spsa' or 'parameter-shift'.")
    if (not conditional_training and
            num_qubits % 2 != 0) or (conditional_training and
                                     num_qubits % 3 != 0):
        condition = "even" if not conditional_training else "divisible by 3"
        raise ValueError(
            f"Number of qubits must be {condition} for the selected training mode."
        )
    logger.debug("All parameters validated successfully.")


def configure_quantum_target(device: str, qpu_count: int) -> int:
    """Configure the quantum target based on device availability."""
    target = "nvidia" if device == "gpu" and cudaq.has_target(
        "nvidia") else "qpp-cpu"
    cudaq.set_target(target, option="mqpu,fp32")
    effective_qpu_count = (cudaq.get_target().num_qpus()
                           if qpu_count == -1 else qpu_count)
    logger.info(
        f"Quantum target set to: {target} with QPU count: {effective_qpu_count}"
    )
    return effective_qpu_count


def ensure_training_data(training_data: str) -> None:
    """Ensure that the training data exists, download if necessary."""
    if not os.path.exists(training_data):
        if os.path.basename(training_data) == "qm9.csv":
            logger.warning(
                f"{training_data} not found. Attempting to download qm9.csv ..."
            )
            os.makedirs(os.path.dirname(training_data), exist_ok=True)
            import gdown

            file_id = "1eXIkHTIeQ0gO84fmGwW7cc9s618xsmVR"
            gdown.download(f"https://drive.google.com/uc?id={file_id}",
                           training_data,
                           quiet=True)
        else:
            logger.warning(
                f"Training data not found at: {training_data}. Set to training_data='./dataset/qm9.csv' to download the QM9 dataset."
            )
    else:
        logger.info(f"Training data found at: {training_data}")


def prepare_datasets(
    training_data: str, sample_percentage: float, validation_split: float,
    seed: int
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset,
           Transformer_Dataset]:
    """Load and split the `dataset` into training and validation sets."""
    block_size = 22 if os.path.basename(training_data) == "qm9.csv" else None
    dataset = Transformer_Dataset(data_path=training_data,
                                  block_size=block_size)

    # Sample a percentage of the `dataset`
    total_size = len(dataset)
    sample_size = int(sample_percentage * total_size)

    train_size = sample_size - int(validation_split * sample_size)
    val_size = sample_size - train_size

    # Randomly select a subset of the `dataset` to use
    sampled_dataset, _ = random_split(
        dataset,
        [sample_size, total_size - sample_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Split the sampled `dataset` into training and validation sets
    train_dataset, val_dataset = random_split(
        sampled_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    logger.info(
        f"Loaded {train_size} training samples and {val_size} validation samples "
        f"from a total of {sample_size} samples ({sample_percentage * 100:.2f}% of the canonicalized dataset)."
    )
    return train_dataset, val_dataset, dataset


def save_smiles_strings(
    train_dataset: torch.utils.data.Subset,
    val_dataset: torch.utils.data.Subset,
    dataset: Transformer_Dataset,
    train_id: str,
) -> None:
    """Save SMILES strings from training and validation `datasets` to CSV files."""

    os.makedirs("./training_splits", exist_ok=True)
    os.makedirs("./validation_splits", exist_ok=True)

    property_names = [
        "MW",
        "HBA",
        "HBD",
        "nRot",
        "nRing",
        "nHet",
        "TPSA",
        "LogP",
        "StereoCenters",
    ]
    with open(f"./training_splits/train_dataset_{train_id}.csv",
              mode="w",
              newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["SMILES"] + property_names)
        for idx in tqdm(train_dataset.indices, desc="Saving training SMILES"):
            smiles = dataset.smiles.iloc[idx]
            properties = get_physchem_properties(smiles)
            writer.writerow([smiles] + properties)
    logger.info(
        f"Training SMILES strings with properties saved to ./training_splits/train_dataset_{train_id}.csv."
    )

    with open(f"./validation_splits/val_dataset_{train_id}.csv",
              mode="w",
              newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["SMILES"] + property_names)
        for idx in tqdm(val_dataset.indices, desc="Saving validation SMILES"):
            smiles = dataset.smiles.iloc[idx]
            properties = get_physchem_properties(smiles)
            writer.writerow([smiles] + properties)
    logger.info(
        f"Validation SMILES strings with properties saved to ./validation_splits/val_dataset_{train_id}.csv."
    )


def initialize_model(
    dataset: Transformer_Dataset,
    device: torch.device,
    classical_attention: bool,
    num_qubits: int,
    ansatz_layers: int,
    conditional_training: bool,
    quantum_gradient_method: str,
    spsa_epsilon: float,
    classical_parameter_reduction: bool,
    qpu_count: int,
    checkpoint_resume_path: Optional[str],
) -> Transformer_Model:
    """Initialize the Transformer model."""
    model = Transformer_Model(
        vocab_size=len(dataset.vocab),
        embed_dim=64,
        block_size=dataset.block_size,
        classical_attention=classical_attention,
        num_qubits=num_qubits,
        ansatz_layers=ansatz_layers,
        conditional_training=conditional_training,
        quantum_gradient_method=quantum_gradient_method,
        epsilon=spsa_epsilon,
        classical_parameter_reduction=classical_parameter_reduction,
        qpu_count=qpu_count,
    ).to(device)

    logger.info("Transformer model initialized.")

    # Since the parameter structures are the same, the same seed will still
    # give different `inits` for the same weights in the hybrid network
    # Thus we choose to save the quantum state dict at `initiailization`
    # and load this into the classical model where possible
    if classical_attention and checkpoint_resume_path is None:
        quantum_model_for_init = Transformer_Model(
            vocab_size=len(dataset.vocab),
            embed_dim=64,
            block_size=dataset.block_size,
            classical_attention=False,
            conditional_training=conditional_training,
            quantum_gradient_method=quantum_gradient_method,
            epsilon=spsa_epsilon,
            qpu_count=qpu_count,
        ).to(device)

        model.load_state_dict(quantum_model_for_init.state_dict(), strict=False)
        logger.info(
            "Initialized classical model with shared weights to ensure same initialization between models"
        )
        del quantum_model_for_init
    return model


def create_data_loaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    batch_size: int,
) -> Tuple[StatefulDataLoader, StatefulDataLoader]:
    """Create training and validation data loaders."""
    train_loader = StatefulDataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=0,
    )
    val_loader = StatefulDataLoader(
        val_dataset,
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=0,
    )
    logger.info("Data loaders created.")
    return train_loader, val_loader


def check_training_configuration(checkpoint_path: Optional[str],
                                 training_configuration: dict) -> bool:
    """Check training configuration for discrepancies."""

    config_match = True
    if checkpoint_path:

        checkpoint = torch.load(checkpoint_path,
                                map_location="cpu",
                                weights_only=False)

        saved_training_configuration = checkpoint["training_configuration"]
        for key in saved_training_configuration:
            if saved_training_configuration[key] != training_configuration[key]:
                if key == "training_data":
                    logger.warning(
                        f"Different training data location detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                elif key == "checkpoint_path":
                    logger.warning(
                        f"Different checkpoint save directory detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                elif key == "checkpoint_resume_path":
                    logger.debug(
                        f"Different checkpoint resume location detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                elif key == "learning_rate":
                    logger.warning(
                        f"Different learning rate detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                elif key == "weight_decay":
                    logger.warning(
                        f"Different weight decay detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                elif key == "batch_size":
                    logger.warning(
                        f"Different batch size detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                elif key == "epochs":
                    logger.warning(
                        f"Different number of epochs detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                elif key == "save_every_n_batches":
                    logger.debug(
                        f"Different save frequency detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                elif key == "validation_split":
                    logger.warning(
                        f"Different validation split detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                elif key == "device":
                    logger.info(
                        f"Different device detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                elif key == "classical_attention":
                    logger.error(
                        f"Different attention mechanism detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                    config_match = False
                elif key == "num_qubits":
                    logger.error(
                        f"Different number of qubits detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                    config_match = False
                elif key == "ansatz_layers":
                    logger.error(
                        f"Different number of ansatz layers detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                    config_match = False
                elif key == "conditional_training":
                    logger.error(
                        f"Different embeddings detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                    config_match = False
                elif key == "quantum_gradient_method":
                    logger.warning(
                        f"Different quantum gradient method detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                elif key == "spsa_epsilon":
                    logger.warning(
                        f"Different SPSA epsilon detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                elif key == "sample_percentage":
                    logger.warning(
                        f"Different percentage of data being used detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                elif key == "seed":
                    logger.warning(
                        f"Different random seed detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                elif key == "classical_parameter_reduction":
                    logger.error(
                        f"Different classical parameter reduction detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                    config_match = False
                elif key == "qpu_count":
                    logger.debug(
                        f"Different QPU count detected for '{key}': {saved_training_configuration[key]} != {training_configuration[key]}"
                    )
                else:
                    logger.debug(
                        f"Resuming training from train_id': {saved_training_configuration[key]}"
                    )

    return config_match


def checkpoint_resume_path_function(
    checkpoint_path: Optional[str],
    model: Transformer_Model,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler | torch.amp.GradScaler,
    train_id: str,
) -> Tuple[int, int, dict, str]:
    """Resume training from a checkpoint."""
    if checkpoint_path:

        checkpoint = torch.load(checkpoint_path,
                                map_location="cpu",
                                weights_only=False)
        train_id = checkpoint["training_configuration"]["train_id"]

        torch.set_rng_state(checkpoint["cpu_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        starting_epoch = checkpoint["epoch"] + 1
        train_losses = checkpoint["training_losses"]
        val_losses = checkpoint["val_losses"]
        mid_epoch = checkpoint["mid_epoch"]
        if mid_epoch:
            # Restore running totals for partial epoch
            total_loss = checkpoint["current_epoch_loss_sum"]
            processed_batches_count = checkpoint["current_epoch_batches_done"]

            # If it's a batch checkpoint, start from same epoch, next batch
            starting_epoch = checkpoint["epoch"]
            start_batch = checkpoint["current_batch"] + 1

        else:
            total_loss = 0.0
            processed_batches_count = 0

            starting_epoch = checkpoint["epoch"] + 1
            start_batch = 0

        logger.info(
            f"Previous training losses: {train_losses}, "
            f"Previous validation losses: {val_losses}, "
            f"Starting at epoch {starting_epoch+1} and batch {start_batch+1} ")

    else:
        train_losses = []
        val_losses = []
        total_loss = 0.0
        processed_batches_count = 0

        starting_epoch = 0
        start_batch = 0

    metrics = {
        "total_loss": total_loss,
        "processed_batches_count": processed_batches_count,
        "training_losses": train_losses,
        "val_losses": val_losses,
    }

    return starting_epoch, start_batch, metrics, train_id


def train_model(
    training_configuration: dict,
    model: Transformer_Model,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler | torch.amp.GradScaler,
    loss_function: CrossEntropyLoss,
    train_loader: StatefulDataLoader,
    val_loader: StatefulDataLoader,
    epochs: int,
    starting_epoch: int,
    start_batch: int,
    metrics: dict,
    checkpoint_path: str,
    save_every_n_batches: int,
    device: torch.device,
) -> None:
    """Main training loop for the Transformer model."""
    global_batch_counter = starting_epoch * len(train_loader) + start_batch

    total_loss = metrics["total_loss"]
    processed_batches_count = metrics["processed_batches_count"]

    for epoch in range(starting_epoch, epochs):

        if start_batch == 0:
            total_loss = 0.0
            processed_batches_count = 0

        model.train()

        # Stateful `dataloader` may use lazy loading and seems to have a problem with `tqdm`
        # We trigger it to load in the batches by priming it
        if (len(metrics["training_losses"]) > 0 and start_batch == 0 and
                epoch == starting_epoch):
            for idx, batch in enumerate(train_loader, start=start_batch):
                break

        train_pbar = tqdm(
            enumerate(train_loader, start=start_batch),
            total=len(train_loader),
            desc=f"Training Epoch {epoch + 1}/{epochs}",
            initial=start_batch,
        )

        for batch_id, batch in train_pbar:
            global_batch_counter += 1
            optimizer.zero_grad()

            x, y, physchem_props = [item.to(device) for item in batch]

            with torch.autocast("cuda"):
                logits, _ = model(x, physchem_props)
                loss = loss_function(logits.view(-1, logits.size(-1)),
                                     y.view(-1))

            # if batch_id < 5:
            #    print(loss)
            #    print(x)

            scaler.scale(loss).backward()

            # Per-layer Gradient clipping
            for _, param in model.named_parameters():
                if param.grad is not None:
                    torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            processed_batches_count += 1

            avg_loss = total_loss / processed_batches_count
            train_pbar.set_description(
                f"Epoch {epoch + 1}/{epochs}; Training Loss: {avg_loss:.4f}")

            # Save batch-level checkpoint
            if save_every_n_batches > 0 and (batch_id +
                                             1) % save_every_n_batches == 0:
                save_checkpoint(
                    training_configuration,
                    model,
                    train_loader,
                    optimizer,
                    scaler,
                    metrics,
                    epoch,
                    batch_id,
                    checkpoint_path,
                    total_loss,
                    processed_batches_count,
                    mid_epoch=True,
                )

        avg_train_loss = total_loss / processed_batches_count
        metrics["training_losses"].append(avg_train_loss)
        logger.debug(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        avg_val_loss = validate_model(model, loss_function, val_loader, device,
                                      epoch, epochs)
        metrics["val_losses"].append(avg_val_loss)
        logger.debug(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")

        # Save epoch-level checkpoint
        save_checkpoint(
            training_configuration,
            model,
            train_loader,
            optimizer,
            scaler,
            metrics,
            epoch,
            0,
            checkpoint_path,
            total_loss,
            processed_batches_count,
        )


def save_checkpoint(
    training_configuration: dict,
    model: Transformer_Model,
    train_loader: StatefulDataLoader,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    metrics: dict,
    epoch: int,
    batch_id: int,
    checkpoint_path: str,
    total_loss: float,
    processed_batches_count: int,
    mid_epoch: bool = False,
) -> None:
    """Save a checkpoint at the current batch."""
    checkpoint = {
        "training_configuration": training_configuration,
        "model_state_dict": model.state_dict(),
        "training_losses": metrics["training_losses"],
        "val_losses": metrics["val_losses"],
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "grad_scaler_state_dict": scaler.state_dict(),
        "train_loader_state": train_loader.state_dict(),
        "current_batch": batch_id,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "current_epoch_loss_sum": total_loss,
        "current_epoch_batches_done": processed_batches_count,
        "mid_epoch": mid_epoch,
    }

    save_name = "most_recent_batch.pt" if mid_epoch else f"model_epoch_{epoch + 1}.pt"
    save_file = os.path.join(checkpoint_path, save_name)
    torch.save(checkpoint, save_file)
    logger.debug(
        f"Checkpoint saved for epoch {epoch + 1}, batch {batch_id + 1} at {save_file}."
    )


def validate_model(
    model: Transformer_Model,
    loss_function: CrossEntropyLoss,
    val_loader: StatefulDataLoader,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """Evaluate the model on the validation set."""
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        val_pbar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f"Validation Epoch {epoch + 1}/{total_epochs}",
        )
        for batch_id, batch in val_pbar:
            x_val, y_val, physchem_props_val = [
                item.to(device) for item in batch
            ]

            with torch.autocast("cuda"):
                val_logits, _ = model(x_val, physchem_props_val)
                val_loss = loss_function(
                    val_logits.view(-1, val_logits.size(-1)), y_val.view(-1))

            total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / (batch_id + 1)
            val_pbar.set_description(
                f"Epoch {epoch + 1}/{total_epochs}; Validation Loss: {avg_val_loss:.4f}"
            )

    return total_val_loss / len(val_loader)
