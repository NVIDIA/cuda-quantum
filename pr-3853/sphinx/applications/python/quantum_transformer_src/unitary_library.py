import cudaq


@cudaq.kernel
def unitary(qubits: cudaq.qview, theta: list[float], subsystem_size: int,
            func: int, layers: int):
    """
    This unitary represents our ansatz, which is a single layer of Ry gates followed by a series of circular entangling CNOT gates

    Args:
        qubits (`cudaq.qview`): The qubits that the ansatz will be applied to
        theta (list[float]): The list of angles that will be used for the Ry gates
        subsystem_size (int): The size of the subsystem (number of qubits spanning token/position/key/query) that the ansatz will be applied to
        `func` (int): The function that the ansatz will be applied to. -1: Apply the ansatz to all qubits
                                                                      0: Apply the ansatz to the first register only, which is used for the token `embeddings`
                                                                      1: Apply the ansatz to the second register only, which is used for the position `embeddings`
                                                                      2: Apply the ansatz to the third register only, which is used for the `physicochemical` `embeddings`
        layers (int): The number of layers of the ansatz
        conditional (bool): Whether the ansatz is conditional on the third register

    Returns:
        None
    """

    # Initialize the start and end indices for the qubits that the ansatz will be applied to
    start, end = 0, 0

    # Since passing strings to a kernel are not supported, we use an integer to represent the choice of function
    if func == -1:
        start, end = 0, qubits.size()

    elif func == 0:
        start, end = 0, subsystem_size

    elif func == 1:
        start, end = subsystem_size, 2 * subsystem_size

    elif func == 2:
        start, end = 2 * subsystem_size, 3 * subsystem_size

    for layer in range(layers):
        # Apply the first set of rotations for the initial layer
        for i in range(start, end):
            ry(theta[i - start + layer * (end - start)], qubits[i])

        # Apply circular entangling CNOT gates
        for i in range(start, end):
            if i < end - 1:
                x.ctrl(qubits[i], qubits[i + 1])
            else:
                x.ctrl(qubits[i], qubits[start])


@cudaq.kernel
def controlled_adjoint_unitary(
    control: cudaq.qubit,
    qubits: cudaq.qview,
    theta: list[float],
    subsystem_size: int,
    func: int,
    layers: int,
):
    """
    This unitary represents the adjoint of our ansatz controlled by the ancilla qubit

    Args:
        control (cudaq.qubit): The ancilla qubit that controls the application of the ansatz
        qubits (`cudaq.qview`): The qubits that the ansatz will be applied to
        theta (list[float]): The list of angles that will be used for the Ry gates
        subsystem_size (int): The size of the subsystem (number of qubits spanning token/position/key/query) that the ansatz will be applied to
        `func` (int): The function that the ansatz will be applied to. -1: Apply the ansatz to all qubits
                                                                      0: Apply the ansatz to the first register only, which is used for the token `embeddings`
                                                                      1: Apply the ansatz to the second register only, which is used for the position `embeddings`
                                                                      2: Apply the ansatz to the third register only, which is used for the `physicochemical` `embeddings`
        layers (int): The number of layers of the ansatz
        conditional (bool): Whether the ansatz is conditional on the third register

    Returns:
        None
    """

    # Initialize the start and end indices for the qubits that the ansatz will be applied to
    start, end = 0, 0

    # Since passing strings to a kernel are not supported, we use an integer to represent the choice of function
    if func == -1:
        start, end = 0, qubits.size()

    elif func == 0:
        start, end = 0, subsystem_size

    elif func == 1:
        start, end = subsystem_size, 2 * subsystem_size

    elif func == 2:
        start, end = 2 * subsystem_size, 3 * subsystem_size

    for layer in range(layers - 1, -1, -1):
        # Apply `CNOTs` in reversed order, starting with the loop closure from last to first
        if (
                end - start > 1
        ):  # Close the loop with a CNOT, ensuring more than one qubit is involved
            x.ctrl(control, qubits[end - 1], qubits[start])
            for i in range(end - 2, start - 1, -1):
                x.ctrl(control, qubits[i], qubits[i + 1])

        # Apply Ry gates in reverse order with negative angles
        for i in range(end - 1, start - 1, -1):
            # Calculate the correct angle index based on layer and qubit position
            angle_index = i - start + layer * (end - start)
            ry.ctrl(-theta[angle_index], control, qubits[i])


@cudaq.kernel
def controlled_unitary(
    control: cudaq.qubit,
    qubits: cudaq.qview,
    theta: list[float],
    subsystem_size: int,
    func: int,
    layers: int,
):
    """
    This unitary represents our ansatz controlled by an ancilla qubit

    Args:
        control (cudaq.qubit): The ancilla qubit that controls the application of the ansatz
        qubits (`cudaq.qview`): The qubits that the ansatz will be applied to
        theta (list[float]): The list of angles that will be used for the Ry gates
        subsystem_size (int): The size of the subsystem (number of qubits spanning token/position/key/query) that the ansatz will be applied to
        `func` (int): The function that the ansatz will be applied to. -1: Apply the ansatz to all qubits
                                                                      0: Apply the ansatz to the first register only, which is used for the token `embeddings`
                                                                      1: Apply the ansatz to the second register only, which is used for the position `embeddings`
                                                                      2: Apply the ansatz to the third register only, which is used for the `physicochemical` `embeddings`
        layers (int): The number of layers of the ansatz
        conditional (bool): Whether the ansatz is conditional on the third register

    Returns:
        None

    """
    start, end = 0, 0

    if func == -1:
        start, end = 0, qubits.size()

    elif func == 0:
        start, end = 0, subsystem_size

    elif func == 1:
        start, end = subsystem_size, 2 * subsystem_size

    elif func == 2:
        start, end = 2 * subsystem_size, 3 * subsystem_size

    for layer in range(layers):
        # Apply the first set of rotations for the initial layer
        for i in range(start, end):
            ry.ctrl(theta[i - start + layer * (end - start)], control,
                    qubits[i])

        # Apply circular entangling CNOT gates
        for i in range(start, end):
            if i < end - 1:
                x.ctrl(control, qubits[i], qubits[i + 1])
            else:
                x.ctrl(control, qubits[i], qubits[start])


# This quantum circuit only include token and positional `encodings` as shown in Figure 4
@cudaq.kernel
def build_sequence_only_circuit(
    token_i: list[float],
    position_i: list[float],
    query: list[float],
    token_j: list[float],
    position_j: list[float],
    key: list[float],
    ansatz_layers: list[int],
    num_working_qubits: list[int],
):
    layers = ansatz_layers[0]
    ancilla = cudaq.qubit()
    register = cudaq.qvector(num_working_qubits[0])
    subsystem_size = num_working_qubits[0] // 2

    h(ancilla)
    unitary(register, token_i, subsystem_size, 0, layers)
    unitary(register, position_i, subsystem_size, 1, layers)
    unitary(register, query, subsystem_size, -1, layers)
    controlled_adjoint_unitary(ancilla, register, query, subsystem_size, -1,
                               layers)
    controlled_adjoint_unitary(ancilla, register, position_i, subsystem_size, 1,
                               layers)
    controlled_adjoint_unitary(ancilla, register, token_i, subsystem_size, 0,
                               layers)
    controlled_unitary(ancilla, register, token_j, subsystem_size, 0, layers)
    controlled_unitary(ancilla, register, position_j, subsystem_size, 1, layers)
    controlled_unitary(ancilla, register, key, subsystem_size, -1, layers)
    h(ancilla)


# This quantum circuit includes token, positional, and `physicochemical` `embeddings` as shown in Figure 5
@cudaq.kernel
def build_physchem_embeddings_circuit(
    token_i: list[float],
    position_i: list[float],
    physchem: list[float],
    query: list[float],
    token_j: list[float],
    position_j: list[float],
    key: list[float],
    ansatz_layers: list[int],
    num_working_qubits: list[int],
):
    layers = ansatz_layers[0]
    ancilla = cudaq.qubit()
    register = cudaq.qvector(num_working_qubits[0])
    subsystem_size = num_working_qubits[0] // 3

    h(ancilla)
    unitary(register, token_i, subsystem_size, 0, layers)
    unitary(register, position_i, subsystem_size, 1, layers)
    unitary(register, physchem, subsystem_size, 2, layers)

    unitary(register, query, subsystem_size, -1, layers)
    controlled_adjoint_unitary(ancilla, register, query, subsystem_size, -1,
                               layers)

    controlled_adjoint_unitary(ancilla, register, position_i, subsystem_size, 1,
                               layers)
    controlled_adjoint_unitary(ancilla, register, token_i, subsystem_size, 0,
                               layers)

    controlled_unitary(ancilla, register, token_j, subsystem_size, 0, layers)
    controlled_unitary(ancilla, register, position_j, subsystem_size, 1, layers)

    controlled_unitary(ancilla, register, key, subsystem_size, -1, layers)
    h(ancilla)
