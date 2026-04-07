Measuring Kernels
======================

.. tab:: Python

    .. literalinclude:: ../../examples/python/measuring_kernels.py
        :language: python
        :start-after: [Begin Docs]
        :end-before: [End Docs]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/measuring_kernels.cpp
        :language: cpp
        :start-after: [Begin Docs]
        :end-before: [End Docs]

Kernel measurement can be specified in the Z, X, or Y basis using `mz`, `mx`, and `my`. Measurement occurs in the Z basis by default.

.. tab:: Python

    .. literalinclude:: ../../examples/python/measuring_kernels.py
        :language: python
        :start-after: [Begin Sample1]
        :end-before: [End Sample1]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/measuring_kernels.cpp
        :language: cpp
        :start-after: [Begin Sample1]
        :end-before: [End Sample1]

Specific qubits or registers can be measured rather than the entire kernel.

.. tab:: Python

    .. literalinclude:: ../../examples/python/measuring_kernels.py
        :language: python
        :start-after: [Begin Sample2]
        :end-before: [End Sample2]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/measuring_kernels.cpp
        :language: cpp
        :start-after: [Begin Sample2]
        :end-before: [End Sample2]

Mid-circuit Measurement and Conditional Logic
----------------------------------------------

In certain cases, it is helpful for some operations in a quantum kernel to depend on measurement results following previous operations. This is accomplished in the following example by performing a Hadamard on qubit 0, then measuring qubit 0 and saving the result as `b0`. Then, qubit 0 can be reset and used later in the computation. In this case it is flipped to a 1. Finally, an if statement performs a Hadamard on qubit 1 if `b0` is 1.

The results show qubit 0 is one, indicating the reset worked, and qubit 1 has a 75/25 distribution, demonstrating the mid-circuit measurement worked as expected.

.. tab:: Python

    .. literalinclude:: ../../examples/python/measuring_kernels.py
        :language: python
        :start-after: [Begin Run0]
        :end-before: [End Run0]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/measuring_kernels.cpp
        :language: cpp
        :start-after: [Begin Run1]
        :end-before: [End Run1]

Output

.. tab:: Python

    .. literalinclude:: ../../examples/python/measuring_kernels.py
        :language: python
        :start-after: [Begin Run1]
        :end-before: [End Run1]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/measuring_kernels.cpp
        :language: cpp
        :start-after: [Begin Run1]
        :end-before: [End Run1]