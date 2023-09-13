# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from concurrent.futures import ThreadPoolExecutor
import ast
import inspect
import sys
import os
import os.path
from ._packages import *

if not "CUDAQ_DYNLIBS" in os.environ:
    try:
        cublas_libs = get_library_path("nvidia-cublas-cu11")
        cublas_path = os.path.join(cublas_libs, "libcublas.so.11")
        cublasLt_path = os.path.join(cublas_libs, "libcublasLt.so.11")

        custatevec_libs = get_library_path("custatevec-cu11")
        custatevec_path = os.path.join(custatevec_libs, "libcustatevec.so.1")

        cutensornet_libs = get_library_path("cutensornet-cu11")
        cutensornet_path = os.path.join(cutensornet_libs, "libcutensornet.so.2")

        os.environ[
            "CUDAQ_DYNLIBS"] = f"{cublasLt_path}:{cublas_path}:{custatevec_path}:{cutensornet_path}"
    except:
        pass

from ._pycudaq import *
from .domains import chemistry

initKwargs = {'target': 'default'}

if '-target' in sys.argv:
    initKwargs['target'] = sys.argv[sys.argv.index('-target') + 1]

if '--target' in sys.argv:
    initKwargs['target'] = sys.argv[sys.argv.index('--target') + 1]

initialize_cudaq(**initKwargs)

# Expose global static quantum operations
h = h()
x = x()
y = y()
z = z()
s = s()
t = t()

rx = rx()
ry = ry()
rz = rz()
r1 = r1()
swap = swap()


class MidCircuitMeasurementAnalyzer(ast.NodeVisitor):
    """The `MidCircuitMeasurementAnalyzer` is a utility class searches for 
       common measurement - conditional patterns to indicate to the runtime 
       that we have a circuit with mid-circuit measurement and subsequent conditional 
       quantum operation application."""

    def __init__(self):
        self.measureResultsVars = []
        self.hasMidCircuitMeasures = False

    def visit_Assign(self, node):
        target = node.targets[0]
        if not 'func' in node.value.__dict__:
            return
        creatorFunc = node.value.func
        if 'id' in creatorFunc.__dict__ and creatorFunc.id == 'mz':
            self.measureResultsVars.append(target.id)

    def visit_If(self, node):
        condition = node.test
        if 'id' in condition.__dict__ and condition.id in self.measureResultsVars:
            self.hasMidCircuitMeasures = True


class kernel(object):
    """The `cudaq.kernel` represents the CUDA Quantum language function 
       attribute that programmers leverage to indicate the following function 
       is a CUDA Quantum kernel and should be compile and executed on 
       an available quantum coprocessor."""

    def __init__(self, function, *args, **kwargs):
        self.kernelFunction = function
        self.inputArgs = args
        self.inputKwargs = kwargs
        src = inspect.getsource(function)
        leadingSpaces = len(src) - len(src.lstrip())
        self.funcSrc = '\n'.join(
            [line[leadingSpaces:] for line in src.split('\n')])
        self.module = ast.parse(self.funcSrc)
        analyzer = MidCircuitMeasurementAnalyzer()
        analyzer.visit(self.module)
        self.metadata = {'conditionalOnMeasure': analyzer.hasMidCircuitMeasures}

    def __call__(self, *args):
        if get_target().is_remote():
            raise Exception(
                "Python kernel functions cannot run on remote QPUs yet.")

        self.kernelFunction(*args)


# Create a Global `ThreadPool`` for asynchronous tasks
threadPool = ThreadPoolExecutor()


class Future:
    '''
    Thin wrapper around a future result from a task running asynchronously.
    '''

    def __init__(self, future):
        self.f = future

    def get(self):
        return self.f.result()


def observe_async(kernel, spin_operator, *args, **kwargs):
    '''
    Compute the expected value of the `spin_operator` with respect to 
    the `kernel` asynchronously. If the kernel accepts arguments, it will 
    be evaluated with respect to `kernel(*arguments)`. When targeting a
    quantum platform with more than one QPU, the optional `qpu_id` allows
    for control over which QPU to enable. Will return a future whose results
    can be retrieved via `future.get()`.

    Args:
        kernel (:class:`Kernel`): The :class:`Kernel` to evaluate the 
            expectation value with respect to.
        spin_operator (:class:`SpinOperator`): The Hermitian spin operator to 
            calculate the expectation of.
        *arguments (Optional[Any]): The concrete values to evaluate the 
            kernel function at. Leave empty if the kernel doesn't accept any arguments.
        qpu_id (Optional[int]): The optional identification for which QPU on 
            the platform to target. Defaults to zero. Key-word only.
        shots_count (Optional[int]): The number of shots to use for QPU 
            execution. Defaults to -1 implying no shots-based sampling. Key-word only.
        noise_model (Optional[`NoiseModel`]): The optional 
            :class:`NoiseModel` to add noise to the kernel execution on the simulator.
            Defaults to an empty noise model.
    Returns:
        Future-like `cudaq.ObserveResult` : A future containing the result of the 
            call to observe.
    '''
    # For now, fall back on existing work for any kernel_builder codes
    if isinstance(kernel, Kernel) or get_target().is_remote():
        return _pycudaq.observe_async(kernel, spin_operator, *args, **kwargs)
    else:
        # For library mode, we need to fork a thread here in Python
        return Future(
            threadPool.submit(observe, kernel, spin_operator, *args, **kwargs))


def sample_async(kernel, *args, **kwargs):
    '''
    Asynchronously sample the state generated by the provided `kernel` at the given kernel 
    `arguments` over the specified number of circuit executions (`shots_count`). 
    Each argument in `arguments` provided can be a `list` or `ndarray` of arguments  
    of the specified kernel argument type, and in this case, the `sample` 
    functionality will be broadcast over all argument sets and a list of 
    `sample_result` instances will be returned.

    Args:
        kernel (:class:`Kernel`): The :class:`Kernel` to execute `shots_count`
            times on the QPU.
        *arguments (Optional[Any]): The concrete values to evaluate the kernel
            function at. Leave empty if the kernel doesn't accept any arguments. For 
            example, if the kernel takes two `float` values as input, the `sample` call 
            should be structured as `cudaq.sample(kernel, firstFloat, secondFloat)`. For 
            broadcasting of the `sample` function, the arguments should be structured as a 
            `list` or `ndarray` of argument values of the specified kernel argument type.
        shots_count (Optional[int]): The number of kernel executions on the QPU.
            Defaults to 1000. Key-word only.
        noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel`
            to add noise to the kernel execution on the simulator. Defaults to
            an empty noise model.
        qpu_id (Optional[int]): The optional identification for which QPU 
            on the platform to target. Defaults to zero. Key-word only.

    Returns:
        Future-like `SampleResult`: A dictionary containing the measurement count
            results for the :class:`Kernel`, or a list of such results in the 
            case of `sample` function broadcasting. 
    '''
    # For now, fall back on existing work for any kernel_builder codes
    if isinstance(kernel, Kernel) or get_target().is_remote():
        return _pycudaq.sample_async(kernel, *args, **kwargs)
    else:
        # For library mode, we need to fork a thread here in Python
        return Future(threadPool.submit(sample, kernel, *args, **kwargs))
