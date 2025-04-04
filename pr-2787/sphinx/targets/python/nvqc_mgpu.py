import argparse
import cudaq
import random

# This example assumes the NVQC API key has been set in the `NVQC_API_KEY` environment variable.
# If not, you can set the API Key environment variable in the Python script with:
# ```
# os.environ["NVQC_API_KEY"] = "<YOUR NVQC API KEY>"`
# ```


def random_bitstring(length: int):
    bitstring = ""
    for bit in range(length):
        bitstring += str(random.randint(0, 1))
    return bitstring


def oracle(kernel: cudaq.Kernel, register: cudaq.QuakeValue,
           auxillary_qubit: cudaq.QuakeValue, hidden_bitstring: str):
    """
    The inner-product oracle for the Bernstein-Vazirani algorithm.
    """
    for index, bit in enumerate(hidden_bitstring):
        if bit == "0":
            # Apply identity operation to the qubit if it's
            # in the 0-state.
            # In this case, we do nothing.
            pass
        else:
            # Otherwise, apply a `cx` gate with the current qubit as
            # the control and the auxillary qubit as the target.
            kernel.cx(control=register[index], target=auxillary_qubit)


def bernstein_vazirani(qubit_count: int):
    """
    Returns a kernel implementing the Bernstein-Vazirani algorithm
    for a random, hidden bitstring.
    """
    kernel = cudaq.make_kernel()
    # Allocate the specified number of qubits - this
    # corresponds to the length of the hidden bitstring.
    qubits = kernel.qalloc(qubit_count)
    # Allocate an extra auxillary qubit.
    auxillary_qubit = kernel.qalloc()

    # Prepare the auxillary qubit.
    kernel.h(auxillary_qubit)
    kernel.z(auxillary_qubit)

    # Place the rest of the register in a superposition state.
    kernel.h(qubits)

    # Generate a random, hidden bitstring for the oracle
    # to encode. Note: we define the bitstring here so
    # as to be able to return it for verification.
    hidden_bitstring = random_bitstring(qubit_count)

    # Query the oracle.
    oracle(kernel, qubits, auxillary_qubit, hidden_bitstring)

    # Apply another set of Hadamards to the register.
    kernel.h(qubits)

    # Apply measurement gates to just the `qubits`
    # (excludes the auxillary qubit).
    kernel.mz(qubits)
    return kernel, hidden_bitstring


# This example demonstrated GPU-accelerated simulator backends on NVQC can easily handle a large number of qubits.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python',
        description='Run a Bernstein-Vazirani algorithm using NVQC.',
        epilog=
        'For more information about CUDA-Q, see https://nvidia.github.io/cuda-quantum'
    )
    parser.add_argument('--size',
                        type=int,
                        required=False,
                        default=30,
                        help='The number of bits in the secret string.')
    parser.add_argument('--ngpus',
                        type=int,
                        required=False,
                        default=1,
                        help='The number of NVQC GPUs to run the simulation.')
    parser.add_argument('--seed',
                        type=int,
                        required=False,
                        default=0,
                        help='The random seed to generate the secret string.')
    args = parser.parse_args()

    # Depending on the number of GPUs requested, you can
    # set the size of the secret string to around 31-34 (total qubit count = string length + 1) when
    # you pass the `--ngpus` as a command line argument.
    qubit_count = args.size
    if args.seed != 0:
        random.seed(args.seed)

    cudaq.set_target("nvqc", backend="nvidia-mgpu", ngpus=args.ngpus)

    print(
        f"Running on NVQC using 'nvidia-mgpu' simulator backend with {args.ngpus} GPU(s) ..."
    )
    kernel, hidden_bitstring = bernstein_vazirani(qubit_count)
    result = cudaq.sample(kernel)

    print(f"encoded bitstring = {hidden_bitstring}")
    print(f"measured state = {result.most_probable()}")
    print(f"Were we successful? {hidden_bitstring == result.most_probable()}")
