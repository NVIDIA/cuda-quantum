# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import multiprocessing
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy

from ._metadata import cuda_major
from ._packages import get_library_path

# Set the multiprocessing start method to `forkserver` if not already set
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method('forkserver')


# ============================================================================ #
# CUDA Library Path Configuration
# ============================================================================ #
def _configure_cuda_library_paths() -> None:
    """    
    Sets the `CUDAQ_DYNLIBS` environment variable with paths to required
    CUDA libraries based on the detected CUDA version.
    """
    # Skip if already configured or CUDA not detected
    if "CUDAQ_DYNLIBS" in os.environ or cuda_major is None:
        return

    # Library configuration: (package_name_template, library_filename_template)
    # Common libraries
    common_libs: Dict[str, Tuple[str, str]] = {
        'cutensor': ('cutensor-cu{cuda_major}', 'libcutensor.so.2'),
        'custatevec': ('custatevec-cu{cuda_major}', 'libcustatevec.so.1'),
        'cutensornet': ('cutensornet-cu{cuda_major}', 'libcutensornet.so.2'),
        'cudensitymat': ('cudensitymat-cu{cuda_major}', 'libcudensitymat.so.0'),
    }

    # CUDA 12 specific libraries
    cuda_12_specific: Dict[str, Tuple[str, str]] = {
        'curand': ('nvidia-curand-cu{cuda_major}', 'libcurand.so.10'),
        'cudart':
            ('nvidia-cuda_runtime-cu{cuda_major}', 'libcudart.so.{cuda_major}'),
        'nvrtc':
            ('nvidia-cuda_nvrtc-cu{cuda_major}', 'libnvrtc.so.{cuda_major}'),
        'cublas': ('nvidia-cublas-cu{cuda_major}', 'libcublas.so.{cuda_major}'),
        'cublaslt':
            ('nvidia-cublas-cu{cuda_major}', 'libcublasLt.so.{cuda_major}'),
        'cusolver': ('nvidia-cusolver-cu{cuda_major}', 'libcusolver.so.11'),
        'cusolvermg': ('nvidia-cusolver-cu{cuda_major}', 'libcusolverMg.so.11'),
    }

    # CUDA 13 specific libraries
    cuda_13_specific: Dict[str, Tuple[str, str]] = {
        'curand': ('nvidia-curand', 'libcurand.so.10'),
        'cudart': ('nvidia-cuda_runtime', 'libcudart.so.{cuda_major}'),
        'nvrtc': ('nvidia-cuda_nvrtc', 'libnvrtc.so.{cuda_major}'),
        'nvrtc_builtins':
            ('nvidia-cuda_nvrtc', 'libnvrtc-builtins.so.{cuda_major}.0'),
        'cublas': ('nvidia-cublas', 'libcublas.so.{cuda_major}'),
        'cublaslt': ('nvidia-cublas', 'libcublasLt.so.{cuda_major}'),
        'cusolver': ('nvidia-cusolver', 'libcusolver.so.12'),
        'cusolvermg': ('nvidia-cusolver', 'libcusolverMg.so.12'),
    }

    # Load dependencies first
    load_order: List[str] = [
        'cudart', 'curand', 'nvrtc', 'nvrtc_builtins', 'cublaslt', 'cublas',
        'cusolver', 'cusolvermg', 'cutensor', 'custatevec', 'cutensornet',
        'cudensitymat'
    ]

    # Select library configuration based on CUDA version
    if cuda_major == 12:
        lib_config = {**common_libs, **cuda_12_specific}
    elif cuda_major == 13:
        lib_config = {**common_libs, **cuda_13_specific}
    else:
        warnings.warn(f"Unsupported CUDA version {cuda_major}.", RuntimeWarning)
        return

    # Colon-separated list of library paths for `LinkedLibraryHolder` to load
    library_paths: List[str] = []

    # Attempt to load each library
    for lib_name in load_order:
        if lib_name not in lib_config:
            continue

        package_template, lib_filename_template = lib_config[lib_name]

        try:
            # Resolve package and library names
            package_name = package_template.format(cuda_major=cuda_major)
            lib_filename = lib_filename_template.format(cuda_major=cuda_major)
            # Get library directory and construct full path
            lib_dir = get_library_path(package_name)
            lib_path = os.path.join(lib_dir, lib_filename)
            # Verify the library file exists
            if not os.path.isfile(lib_path):
                raise FileNotFoundError(f"Library file not found: {lib_path}")

            library_paths.append(lib_path)

        except Exception:
            continue

    os.environ["CUDAQ_DYNLIBS"] = ":".join(library_paths)


# CUDAQ_DYNLIBS must be set before any other imports that would initialize
# `LinkedLibraryHolder`.
try:
    _configure_cuda_library_paths()
except Exception:
    import importlib.util
    package_spec = importlib.util.find_spec(f"cuda-quantum-cu{cuda_major}")
    if not package_spec is None and not package_spec.loader is None:
        print("Could not find a suitable cuQuantum Python package.")
    pass

# ============================================================================ #
# Module Imports
# ============================================================================ #

from .display import display_trace
from .kernel.kernel_decorator import kernel, PyKernelDecorator
from .kernel.kernel_builder import (make_kernel, QuakeValue, PyKernel)
from .kernel.ast_bridge import (globalAstRegistry, globalRegisteredOperations)
from .runtime.sample import sample
from .runtime.sample import sample_async, AsyncSampleResult
from .runtime.observe import observe
from .runtime.observe import observe_async
from .runtime.run import run
from .runtime.run import run_async
from .runtime.translate import translate
from .runtime.state import (get_state, get_state_async, to_cupy)
from .runtime.draw import draw
from .runtime.unitary import get_unitary
from .runtime.resource_count import estimate_resources
from .runtime.vqe import vqe  # Removed! Use VQE from CUDA-QX
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
qubit = cudaq_runtime.qubit
qvector = cudaq_runtime.qvector
qview = cudaq_runtime.qview
Pauli = cudaq_runtime.Pauli
Kernel = PyKernel
Target = cudaq_runtime.Target
State = cudaq_runtime.State
StateMemoryView = cudaq_runtime.StateMemoryView
pauli_word = cudaq_runtime.pauli_word
Tensor = cudaq_runtime.Tensor
SimulationPrecision = cudaq_runtime.SimulationPrecision
Resources = cudaq_runtime.Resources

# to be deprecated
qreg = cudaq_runtime.qvector

# Operator API
from .operators import boson
from .operators import fermion
from .operators import spin
from .operators import custom as operators
from .operators.definitions import *
from .operators.manipulation import OperatorArithmetics
# needs to be imported, since otherwise e.g. evaluate is not defined
import cudaq.operators.expressions
from .operators.super_op import SuperOperator

# Time evolution API
from .dynamics.schedule import Schedule
from .dynamics.evolution import evolve, evolve_async
from .dynamics.integrators import *
from .dynamics.helpers import IntermediateResultSave

InitialStateType = cudaq_runtime.InitialStateType

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
register_set_target_callback = cudaq_runtime.register_set_target_callback
unregister_set_target_callback = cudaq_runtime.unregister_set_target_callback

# Noise Modeling
KrausChannel = cudaq_runtime.KrausChannel
KrausOperator = cudaq_runtime.KrausOperator
NoiseModelType = cudaq_runtime.NoiseModelType
NoiseModel = cudaq_runtime.NoiseModel
DepolarizationChannel = cudaq_runtime.DepolarizationChannel
AmplitudeDampingChannel = cudaq_runtime.AmplitudeDampingChannel
PhaseFlipChannel = cudaq_runtime.PhaseFlipChannel
BitFlipChannel = cudaq_runtime.BitFlipChannel
PhaseDamping = cudaq_runtime.PhaseDamping
ZError = cudaq_runtime.ZError
XError = cudaq_runtime.XError
YError = cudaq_runtime.YError
Pauli1 = cudaq_runtime.Pauli1
Pauli2 = cudaq_runtime.Pauli2
Depolarization1 = cudaq_runtime.Depolarization1
Depolarization2 = cudaq_runtime.Depolarization2

# Functions
SampleResult = cudaq_runtime.SampleResult
ObserveResult = cudaq_runtime.ObserveResult
AsyncObserveResult = cudaq_runtime.AsyncObserveResult
EvolveResult = cudaq_runtime.EvolveResult
AsyncEvolveResult = cudaq_runtime.AsyncEvolveResult
AsyncStateResult = cudaq_runtime.AsyncStateResult
displaySVG = display_trace.displaySVG
getSVGstring = display_trace.getSVGstring

ComplexMatrix = cudaq_runtime.ComplexMatrix

testing = cudaq_runtime.testing

# target-specific
orca = cudaq_runtime.orca

# ============================================================================ #
# Utility Functions
# ============================================================================ #


def synthesize(kernel, *args):
    return PyKernelDecorator(None,
                             module=cudaq_runtime.synthesize(kernel, *args),
                             kernelName=kernel.name,
                             decorator=kernel)


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
    global globalAstRegistry, globalRegisteredOperations
    globalAstRegistry.clear()
    globalRegisteredOperations.clear()


# Expose chemistry domain functions
from .domains import chemistry
from .kernels import uccsd
from .dbg import ast

# ============================================================================ #
# Command Line Argument Parsing
# ============================================================================ #
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
