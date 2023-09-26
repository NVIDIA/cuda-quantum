# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

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
from .language.analysis import MidCircuitMeasurementAnalyzer

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


class kernel(object):
    """The `cudaq.kernel` represents the CUDA Quantum language function 
       attribute that programmers leverage to indicate the following function 
       is a CUDA Quantum kernel and should be compiled and executed on 
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
