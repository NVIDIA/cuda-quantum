# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys, os, numpy, platform, multiprocessing
from ._packages import *

# Set the multiprocessing start method to 'spawn' if not already set
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method('forkserver')

# CUDAQ_DYNLIBS must be set before any other imports that would initialize
# LinkedLibraryHolder.
if not "CUDAQ_DYNLIBS" in os.environ:
    try:
        custatevec_libs = get_library_path("custatevec-cu11")
        custatevec_path = os.path.join(custatevec_libs, "libcustatevec.so.1")

        cutensornet_libs = get_library_path("cutensornet-cu11")
        cutensornet_path = os.path.join(cutensornet_libs, "libcutensornet.so.2")

        os.environ["CUDAQ_DYNLIBS"] = f"{custatevec_path}:{cutensornet_path}"

        # The following package is only available on `x86_64` (not `aarch64`). For
        # `aarch64`, the library must be provided another way (likely with
        # LD_LIBRARY_PATH).
        if platform.processor() == "x86_64":
            cudart_libs = get_library_path("nvidia-cuda_runtime-cu11")
            cudart_path = os.path.join(cudart_libs, "libcudart.so.11.0")
            os.environ["CUDAQ_DYNLIBS"] += f":{cudart_path}"
    except:
        import importlib.util
        package_spec = importlib.util.find_spec("cuda-quantum")
        if not package_spec is None and not package_spec.loader is None:
            print("Could not find a suitable cuQuantum Python package.")
        pass

from .display import display_trace
from .kernel.kernel_decorator import kernel, PyKernelDecorator
from .kernel.kernel_builder import make_kernel, QuakeValue, PyKernel
from .kernel.ast_bridge import globalAstRegistry, globalKernelRegistry, globalRegisteredOperations
from .runtime.sample import sample
from .runtime.observe import observe
from .runtime.state import to_cupy
from .kernel.register_op import register_operation

from .mlir._mlir_libs._quakeDialects import cudaq_runtime

try:
    from qutip import Qobj, Bloch
except ImportError:
    from .visualization.bloch_visualize_err import install_qutip_request as add_to_bloch_sphere
    from .visualization.bloch_visualize_err import install_qutip_request as show
else:
    from .visualization.bloch_visualize import add_to_bloch_sphere
    from .visualization.bloch_visualize import show_bloch_sphere as show

# Add the parallel runtime types
parallel = cudaq_runtime.parallel

# Primitive Types
spin = cudaq_runtime.spin
qubit = cudaq_runtime.qubit
qvector = cudaq_runtime.qvector
qview = cudaq_runtime.qview
SpinOperator = cudaq_runtime.SpinOperator
Pauli = cudaq_runtime.Pauli
Kernel = PyKernel
Target = cudaq_runtime.Target
State = cudaq_runtime.State
pauli_word = cudaq_runtime.pauli_word
Tensor = cudaq_runtime.Tensor
SimulationPrecision = cudaq_runtime.SimulationPrecision

# to be deprecated
qreg = cudaq_runtime.qvector

# Optimizers + Gradients
optimizers = cudaq_runtime.optimizers
gradients = cudaq_runtime.gradients
OptimizationResult = cudaq_runtime.OptimizationResult

# Runtime Functions
__version__ = cudaq_runtime.__version__
initialize_cudaq = cudaq_runtime.initialize_cudaq
set_target = cudaq_runtime.set_target
reset_target = cudaq_runtime.reset_target
has_target = cudaq_runtime.has_target
get_target = cudaq_runtime.get_target
get_targets = cudaq_runtime.get_targets
set_random_seed = cudaq_runtime.set_random_seed
mpi = cudaq_runtime.mpi
num_available_gpus = cudaq_runtime.num_available_gpus
set_noise = cudaq_runtime.set_noise
unset_noise = cudaq_runtime.unset_noise

# Noise Modeling
KrausChannel = cudaq_runtime.KrausChannel
KrausOperator = cudaq_runtime.KrausOperator
NoiseModel = cudaq_runtime.NoiseModel
DepolarizationChannel = cudaq_runtime.DepolarizationChannel
AmplitudeDampingChannel = cudaq_runtime.AmplitudeDampingChannel
PhaseFlipChannel = cudaq_runtime.PhaseFlipChannel
BitFlipChannel = cudaq_runtime.BitFlipChannel

# Functions
sample_async = cudaq_runtime.sample_async
observe_async = cudaq_runtime.observe_async
get_state = cudaq_runtime.get_state
get_state_async = cudaq_runtime.get_state_async
SampleResult = cudaq_runtime.SampleResult
ObserveResult = cudaq_runtime.ObserveResult
AsyncSampleResult = cudaq_runtime.AsyncSampleResult
AsyncObserveResult = cudaq_runtime.AsyncObserveResult
AsyncStateResult = cudaq_runtime.AsyncStateResult
vqe = cudaq_runtime.vqe
draw = cudaq_runtime.draw
translate = cudaq_runtime.translate
displaySVG = display_trace.displaySVG
getSVGstring = display_trace.getSVGstring

ComplexMatrix = cudaq_runtime.ComplexMatrix

# to be deprecated
to_qir = cudaq_runtime.get_qir

testing = cudaq_runtime.testing

# target-specific
orca = cudaq_runtime.orca


def synthesize(kernel, *args):
    # Compile if necessary, no-op if already compiled
    kernel.compile()
    return PyKernelDecorator(None,
                             module=cudaq_runtime.synthesize(kernel, *args),
                             kernelName=kernel.name)


def complex():
    """
    Return the data type for the current simulation backend, 
    either `numpy.complex128` or `numpy.complex64`.
    """
    target = get_target()
    precision = target.get_precision()
    if precision == cudaq_runtime.SimulationPrecision.fp64:
        return numpy.complex128
    return numpy.complex64


def amplitudes(array_data):
    """
    Create a state array with the appropriate data type for the 
    current simulation backend target. 
    """
    return numpy.array(array_data, dtype=complex())


def __clearKernelRegistries():
    global globalKernelRegistry, globalAstRegistry, globalRegisteredOperations
    globalKernelRegistry.clear()
    globalAstRegistry.clear()
    globalRegisteredOperations.clear()


# Expose chemistry domain functions
from .domains import chemistry
from .kernels import uccsd
from .dbg import ast

initKwargs = {}

# Look for --target=<target> options
for p in sys.argv:
    split_params = p.split('=')
    if len(split_params) == 2:
        if split_params[0] in ['-target', '--target']:
            initKwargs['target'] = split_params[1]

# Look for --target <target> (with a space)
if '-target' in sys.argv:
    initKwargs['target'] = sys.argv[sys.argv.index('-target') + 1]
if '--target' in sys.argv:
    initKwargs['target'] = sys.argv[sys.argv.index('--target') + 1]
if '--target-option' in sys.argv:
    initKwargs['option'] = sys.argv[sys.argv.index('--target-option') + 1]
if '--emulate' in sys.argv:
    initKwargs['emulate'] = True
if not '--cudaq-full-stack-trace' in sys.argv:
    sys.tracebacklimit = 0

cudaq_runtime.initialize_cudaq(**initKwargs)
