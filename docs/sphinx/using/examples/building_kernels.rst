Building Kernels
================
This section will cover the most basic CUDA-Q construct, a quantum kernel.
Topics include, building kernels, initializing states, and applying gate operations.

Defining Kernels
----------------

Kernels are the building blocks of quantum algorithms in CUDA-Q. A kernel is specified by using the following syntax. `cudaq.qubit` builds a register consisting of a single qubit, while `cudaq.qvector` builds a register of :math:`N` qubits.

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin Definition]
      :end-before: [End Definition]

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/building_kernels.cpp
      :language: cpp
      :start-after: [Begin Definition]
      :end-before: [End Definition]

Inputs to kernels are defined by specifying a parameter in the kernel definition along with the appropriate type. The kernel below takes an integer to define a register of N qubits.

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin `InputDefinition`]
      :end-before: [End `InputDefinition`]

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/building_kernels.cpp
      :start-after: [Begin `InputDefinition`]
      :end-before: [End `InputDefinition`]

Initializing states
-------------------

It is often helpful to define an initial state for a kernel. There are a few ways to do this in CUDA-Q. Note, method 5 is particularly useful for cases where the state of one kernel is passed into a second kernel to prepare its initial state.

1. Passing complex vectors as parameters

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin `PassingComplexVector`]
      :end-before: [End `PassingComplexVector`]

.. tab:: C++
   
   .. literalinclude:: ../../examples/cpp/building_kernels.cpp
      :language: cpp
      :start-after: [Begin `PassingComplexVector`]
      :end-before: [End `PassingComplexVector`]

2. Capturing complex vectors

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin `CapturingComplexVector`]
      :end-before: [End `CapturingComplexVector`]

.. tab:: C++
   
   .. literalinclude:: ../../examples/cpp/building_kernels.cpp
      :language: cpp
      :start-after: [Begin `CapturingComplexVector`]
      :end-before: [End `CapturingComplexVector`]

3. Precision-agnostic API

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin `PrecisionAgnosticAPI`]
      :end-before: [End `PrecisionAgnosticAPI`]

.. tab:: C++
   
   .. literalinclude:: ../../examples/cpp/building_kernels.cpp
      :language: cpp
      :start-after: [Begin `PrecisionAgnosticAPI`]
      :end-before: [End `PrecisionAgnosticAPI`]

4. Define as CUDA-Q amplitudes

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin `CUDAQAmplitudes`]
      :end-before: [End `CUDAQAmplitudes`]

5. Pass in a state from another kernel

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin `PassingState`]
      :end-before: [End `PassingState`]

Applying Gates
--------------

After a kernel is constructed, gates can be applied to start building out a quantum circuit.
All the predefined gates in CUDA-Q can be found `here <https://nvidia.github.io/cuda-quantum/api/default_ops>`_.

Gates can be applied to all qubits in a register.

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin `AllQubits`]
      :end-before: [End `AllQubits`]

.. tab:: C++
   
   .. literalinclude:: ../../examples/cpp/building_kernels.cpp
      :language: cpp
      :start-after: [Begin `AllQubits`]
      :end-before: [End `AllQubits`]

Or, to individual qubits in a register.

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin `IndividualQubits`]
      :end-before: [End `IndividualQubits`]

.. tab:: C++
   
   .. literalinclude:: ../../examples/cpp/building_kernels.cpp
      :language: cpp
      :start-after: [Begin `IndividualQubits`]
      :end-before: [End `IndividualQubits`]

Controlled Operations
---------------------

Controlled operations are available for any gate and can be used by adding `.ctrl` to the end of any gate, followed by specification of the control qubit and the target qubit.

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin `ControlledOperations`]
      :end-before: [End `ControlledOperations`]

Multi-Controlled Operations
---------------------------

It is valid for more than one qubit to be used for multi-controlled gates. The control qubits are specified as a list.

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin `MultiControlledOperations`]
      :end-before: [End `MultiControlledOperations`]

.. tab:: C++
   
   .. literalinclude:: ../../examples/cpp/building_kernels.cpp
      :language: cpp
      :start-after: [Begin `MultiControlledOperations`]
      :end-before: [End `MultiControlledOperations`]

You can also call a controlled kernel within a kernel.

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin `ControlledKernel`]
      :end-before: [End `ControlledKernel`]

.. tab:: C++
   
   .. literalinclude:: ../../examples/cpp/building_kernels.cpp
      :language: cpp
      :start-after: [Begin `ControlledKernel`]
      :end-before: [End `ControlledKernel`]

Adjoint Operations
------------------

The adjoint of a gate can be applied by appending the gate with the `adj` designation.

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin `AdjointOperations`]
      :end-before: [End `AdjointOperations`]

.. tab:: C++
   
   .. literalinclude:: ../../examples/cpp/building_kernels.cpp
      :language: cpp
      :start-after: [Begin `AdjointOperations`]
      :end-before: [End `AdjointOperations`]

Custom Operations
-----------------

Custom gate operations can be specified using `cudaq.register_operation`. A one-dimensional `Numpy` array specifies the unitary matrix to be applied. The entries of the array read from top to bottom through the rows.

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin `CustomOperations`]
      :end-before: [End `CustomOperations`]

Building Kernels with Kernels
-----------------------------

For many complex applications, it is helpful for a kernel to call another kernel to perform a specific subroutine. The example blow shows how `kernel_A` can be called within `kernel_B` to perform CNOT operations.

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin `BuildingKernelsWithKernels`]
      :end-before: [End `BuildingKernelsWithKernels`]

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/building_kernels.cpp
      :language: cpp
      :start-after: [Begin `BuildingKernelsWithKernels`]
      :end-before: [End `BuildingKernelsWithKernels`]   

Parameterized Kernels
---------------------

It is often useful to define parameterized circuit kernels which can be used for applications like VQE.

.. tab:: Python

   .. literalinclude:: ../../examples/python/building_kernels.py
      :language: python
      :start-after: [Begin `ParameterizedKernels`]
      :end-before: [End `ParameterizedKernels`]

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/building_kernels.cpp
      :language: cpp
      :start-after: [Begin `ParameterizedKernels`]
      :end-before: [End `ParameterizedKernels`]
