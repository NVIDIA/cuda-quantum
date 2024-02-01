# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys, os
from ._packages import *
from .kernel.kernel_decorator import kernel, PyKernelDecorator
from .kernel.kernel_builder import make_kernel, QuakeValue, PyKernel
from .kernel.ast_bridge import globalAstRegistry, globalKernelRegistry
from .kernel.qubit_qis import adjoint, control, compute_action
from .runtime.sample import sample
from .runtime.observe import observe
from .runtime.state import get_state
from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime

global globalJIT

# Primitive Types
spin = cudaq_runtime.spin
qubit = cudaq_runtime.qubit
qvector = cudaq_runtime.qvector
qview = cudaq_runtime.qview
qlist = qvector
SpinOperator = cudaq_runtime.SpinOperator
Pauli = cudaq_runtime.Pauli
Kernel = PyKernel
Target = cudaq_runtime.Target

# to be deprecated
qreg = cudaq_runtime.qvector

# Optimizers + Gradients
optimizers = cudaq_runtime.optimizers
gradients = cudaq_runtime.gradients

# Runtime Functions
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
SampleResult = cudaq_runtime.SampleResult
ObserveResult = cudaq_runtime.ObserveResult
AsyncSampleResult = cudaq_runtime.AsyncSampleResult
AsyncObserveResult = cudaq_runtime.AsyncObserveResult

ComplexMatrix = cudaq_runtime.ComplexMatrix
to_qir = cudaq_runtime.get_qir
testing = cudaq_runtime.testing


def synthesize(kernel, *args):
    if kernel.module is None:
        raise RuntimeError("kernel must have jit=True for cudaq.synthesize")
    return PyKernelDecorator(None,
                             module=cudaq_runtime.synthesize(kernel, *args),
                             kernelName=kernel.name)


def __clearKernelRegistries():
    global globalKernelRegistry, globalAstRegistry
    globalKernelRegistry.clear()
    globalAstRegistry.clear()


def enable_jit():
    """
    Enable JIT compilation to MLIR for all cudaq.kernel functions
    """
    PyKernelDecorator.globalJIT = True


def disable_jit():
    """
    Disable JIT compilation to MLIR for all cudaq.kernel functions
    """
    PyKernelDecorator.globalJIT = False


if not "CUDAQ_DYNLIBS" in os.environ:
    try:
        custatevec_libs = get_library_path("custatevec-cu11")
        custatevec_path = os.path.join(custatevec_libs, "libcustatevec.so.1")

        cutensornet_libs = get_library_path("cutensornet-cu11")
        cutensornet_path = os.path.join(cutensornet_libs, "libcutensornet.so.2")

        os.environ["CUDAQ_DYNLIBS"] = f"{custatevec_path}:{cutensornet_path}"
    except:
        import importlib.util
        if not importlib.util.find_spec("cuda-quantum") is None:
            print("Could not find a suitable cuQuantum Python package.")
        pass

initKwargs = {}

if '-target' in sys.argv:
    initKwargs['target'] = sys.argv[sys.argv.index('-target') + 1]
if '--target' in sys.argv:
    initKwargs['target'] = sys.argv[sys.argv.index('--target') + 1]
if '--emulate' in sys.argv:
    initKwargs['emulate'] = True

cudaq_runtime.initialize_cudaq(**initKwargs)
