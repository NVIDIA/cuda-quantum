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

Measurement Handles
----------------------------------------------

In CUDA-Q, :code:`mz`, :code:`mx`, and :code:`my` return a *measurement
handle* — :code:`cudaq::measure_handle` in C++ (with the alias
:code:`cudaq::measure_result`), and the :code:`measure_handle` type in
Python — rather than a classical value. Measuring a single qubit returns one
handle; measuring a :code:`qvector` returns a vector of handles. A handle
records a measurement event and defers reading its classical value, so the
same measurement can drive mid-circuit conditional logic and
quantum-error-correction declarations (see :doc:`dem_from_kernel`).

A handle is *discriminated* into its classical bit by using it in a boolean
context inside the kernel — for example the :code:`if (b0)` test in the
mid-circuit example below. To read a whole vector of handles at once,
discriminate it in bulk with :code:`to_bools` (yielding a
:code:`list[bool]` / :code:`std::vector<bool>`) or :code:`to_integer`
(packing the bits little-endian into an integer). The C++ mid-circuit example
below returns :code:`to_bools(mz(q))`; a Python kernel typed to return
:code:`list[bool]` discriminates a returned handle vector automatically.

A handle cannot cross the host-device boundary without being discriminated:
convert it to a boolean, :code:`list[bool]` / :code:`std::vector<bool>`, or
integer inside the kernel before returning it.

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
        :start-after: [Begin Run0]
        :end-before: [End Run0]

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