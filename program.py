# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np

def use_builder1():
    v = [0., 1., 1., 0.] 

    # (deferred) qubit allocation from concrete state vector
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(v)
  
    cudaq.sample(kernel).dump()

def use_builder2():
    v = [0., 1., 1., 0.] 

    # kernel parameterized on input state data
    kernel, state = cudaq.make_kernel(list[float])
    qubits = kernel.qalloc(state)
  
    cudaq.sample(kernel, v).dump()

############################
# Test using builder
############################
    
# Current behavior
# TypeError: get(): incompatible function arguments. The following argument types are supported:
#     1. (cls: object, context: MlirContext, size: int = 0) -> MlirType

# Invoked with: <class 'importlib._bootstrap.VeqType'>, <cudaq.mlir._mlir_libs._site_initialize.<locals>.Context object at 0x7ff594b64c70>, [0.0, 1.0, 1.0, 0.0]
    
# use_builder1()
    
############################
# Test using builder
############################

# Current behavior

# Vector created from a list:
# [0.0, 1.0, 1.0, 0.0]
# error: 'quake.alloca' op operand #0 must be signless integer, but got '!cc.stdvec<f64>'
# RuntimeError: cudaq::builder failed to JIT compile the Quake representation.

# use_builder2()

##########################
# Test creating a kernel
##########################

# Current behavior
    
# error: 'quake.alloca' op operand #0 must be signless integer, but got '!cc.stdvec<complex<f64>>'
# RuntimeError: Failure while executing pass pipeline.

# During handling of the above exception, another exception occurred:

# RuntimeError: could not compile code for 'test'.
    
# vector of float
    
cudaq.reset_target()
cudaq.set_target('nvidia-fp64')

f = [0., 1., 1., 0.]

@cudaq.kernel 
def test_vec1(vec : list[float]):
   q = cudaq.qvector(vec)

# cudaq.sample(test_vec1, f).dump()


@cudaq.kernel
def test_vec2():
   q = cudaq.qvector(f)

# cudaq.sample(test_vec2).dump()

@cudaq.kernel
def test_vec3():
   q = cudaq.qvector(np.array(f))

# cudaq.sample(test_vec3).dump()


# vector of complex

c = [.70710678 + 0j, 0., 0., 0.70710678]

@cudaq.kernel(verbose=True)
def test_complex_vec1(vec : list[complex]):
   q = cudaq.qvector(vec)

# works!
#cudaq.sample(test_complex_vec1, c).dump()

@cudaq.kernel(verbose=True)
def test_complex_vec2():
   q = cudaq.qvector(c)
   # q = cudaq.qvector(2)

# works!
#cudaq.sample(test_complex_vec2).dump()

@cudaq.kernel(verbose=True)
def test_complex_vec3():
   q = cudaq.qvector(np.array(c))

# works!
# cudaq.sample(test_complex_vec3).dump()
   
@cudaq.kernel(verbose=True)
def test_complex_vec4():
   q = cudaq.qvector([1.0 + 0j, 0., 0., 1.])
# TODO: error: 'arith.addf' op operand #0 must be floating-point-like, but got 'complex<f64>'
# cudaq.sample(test_complex_vec4).dump()
   

# simulation scalar

cudaq.reset_target()

@cudaq.kernel(verbose=True)
def test_complex_vec5(vec : list[complex]):
      q = cudaq.qvector(np.array(c, dtype=cudaq.simulation_dtype()))

cudaq.sample(test_complex_vec4).dump()

# builder

cudaq.reset_target()
cudaq.set_target('nvidia-fp64')

def test_from_state():
    kernel, initState = cudaq.make_kernel(list[complex])
    qubits = kernel.qalloc(initState)
    print(kernel)

    # Test float64 list, casts to complex
    state = [.70710678, 0., 0., 0.70710678]
    cudaq.sample(kernel, state).dump()

# test_from_state()