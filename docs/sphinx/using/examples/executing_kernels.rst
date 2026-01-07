Executing Kernels
=================

In CUDA-Q, there are 4 ways in which one can execute quantum kernels:

1. `sample`: yields measurement counts
2. `run`: yields individual return values from multiple executions
3. `observe`: yields expectation values
4. `get_state`: yields the quantum statevector of the computation

Asynchronous programming is a technique that enables your program to start a potentially long-running task and still be able to be responsive to other events while that task runs, rather than having to wait until that task has finished. Once that task has finished, your program is presented with the result. The most intensive task in the computation is the execution of the quantum kernel hence each execution function can be parallelized given access to multiple quantum processing units (multi-QPU) using: `sample_async`, `run_async`, `observe_async` and `get_state_async`.

Since multi-QPU platforms are not yet feasible, we emulate each QPU with a GPU.

Sample
------

Quantum states collapse upon measurement and hence need to be sampled many times to gather statistics. The CUDA-Q `sample` call enables this:

.. tab:: Python

    .. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin Sample]
        :end-before: [End Sample]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/executing_kernels_sample.cpp
        :language: cpp
        :start-after: [Begin Sample]
        :end-before: [End Sample]

.. literalinclude:: ../../examples/python/executing_kernels.py
        :start-after: [Begin `SampleOutput`]
        :end-before: [End `SampleOutput`]

Note that there is a subtle difference between how `sample` is executed with the target device set to a simulator or with the target device set to a QPU. In simulation mode, the quantum state is built once and then sampled :math:`s` times where :math:`s` equals the `shots_count`. In hardware execution mode, the quantum state collapses upon measurement and hence needs to be rebuilt over and over again.

There are a number of helpful tools that can be found in the `API docs <https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api>`_ to process the `Sample_Result` object produced by `sample`.

Sample Asynchronous
~~~~~~~~~~~~~~~~~~~

`sample` also supports asynchronous execution for the `sample_async arguments it accepts <https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html#cudaq.sample_async>`_. One can parallelize over various kernels, variational parameters or even distribute shots counts over multiple QPUs.

.. tab:: Python

    .. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin `SampleAsync`]
        :end-before: [End `SampleAsync`]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/executing_kernels_sample_async.cpp
        :language: cpp
        :start-after: [Begin `SampleAsync`]
        :end-before: [End `SampleAsync`]

.. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin `SampleAsyncOutput`]
        :end-before: [End `SampleAsyncOutput`]

Run
---

The `run` API executes a quantum kernel multiple times and returns each individual result. Unlike `sample`, which collects measurement statistics as counts, `run` preserves each individual return value from every execution. This is useful when you need to analyze the distribution of returned values rather than just aggregated measurement counts.

Key points about `run`:

- Requires a kernel that returns a non-void value
- Returns a list containing all individual execution results
- Supports scalar types (bool, int, float) and custom data classes as return types

.. tab:: Python

    .. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin Run]
        :end-before: [End Run]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/executing_kernels_run.cpp
        :language: cpp
        :start-after: [Begin Run]
        :end-before: [End Run]

.. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin `RunOutput`]
        :end-before: [End `RunOutput`]

Return Custom Data Types
~~~~~~~~~~~~~~~~~~~~~~~~

The `run` API also supports returning custom data types using Python's data classes. This allows returning multiple values from your quantum computation in a structured way.

.. tab:: Python

    .. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin `RunCustom`]
        :end-before: [End `RunCustom`]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/executing_kernels_run_custom.cpp
        :language: cpp
        :start-after: [Begin `RunCustom`]
        :end-before: [End `RunCustom`]

.. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin `RunCustomOutput`]
        :end-before: [End `RunCustomOutput`]

Run Asynchronous
~~~~~~~~~~~~~~~~

Similar to `sample_async`, `run` also supports asynchronous execution for the `run_async arguments it accepts <https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html#cudaq.run_async>`_.

.. tab:: Python

    .. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin `RunAsync`]
        :end-before: [End `RunAsync`]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/executing_kernels_run_async.cpp
        :language: cpp
        :start-after: [Begin `RunAsync`]
        :end-before: [End `RunAsync`]

.. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin `RunAsyncOutput`]
        :end-before: [End `RunAsyncOutput`]

.. note::
    Currently, `run` and `run_async` are supported on simulator targets and select hardware platforms.

Observe
-------

The `observe` function allows us to calculate expectation values. We must supply a spin operator in the form of a Hamiltonian, :math:`H`, from which we would like to calculate :math:`\langle\psi|H|\psi\rangle`.

.. tab:: Python

    .. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin Observe]
        :end-before: [End Observe]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/executing_kernels_observe.cpp
        :language: cpp
        :start-after: [Begin Observe]
        :end-before: [End Observe]

.. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin `ObserveOutput`]
        :end-before: [End `ObserveOutput`]

Observe Asynchronous
~~~~~~~~~~~~~~~~~~~~

`observe` can be a time intensive task. We can parallelize the execution of `observe` via the arguments it accepts.

.. tab:: Python

    .. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin `ObserveAsync`]
        :end-before: [End `ObserveAsync`]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/executing_kernels_observe_async.cpp
        :language: cpp
        :start-after: [Begin `ObserveAsync`]
        :end-before: [End `ObserveAsync`]

.. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin `ObserveAsyncOutput`]
        :end-before: [End `ObserveAsyncOutput`]

Above we parallelized the `observe` call over the `hamiltonian` parameter; however, we can parallelize over any of the arguments it accepts by just iterating over the `qpu_id`.

Get State
---------

The `get_state` function gives us access to the quantum statevector of the computation. Remember, that this is only feasible in simulation mode.

.. tab:: Python

    .. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin `GetState`]
        :end-before: [End `GetState`]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/executing_kernels_state.cpp
        :language: cpp
        :start-after: [Begin `GetState`]
        :end-before: [End `GetState`]

.. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin `GetStateOutput`]
        :end-before: [End `GetStateOutput`]

The statevector generated by the `get_state` command follows Big-endian convention for associating numbers with their binary representations, which places the least significant bit on the left. That is, for the example of a 2-bit system, we have the following translation between integers and bits:

.. math::

    \begin{matrix} 
    \text{Integer} & \text{Binary representation}\\
    & \text{least signinificant bit on left}\\
    0 = \textcolor{red}{0}*2^0 + \textcolor{blue}{0} *2^1 & \textcolor{red}{0}\textcolor{blue}{0} \\
    1 = \textcolor{red}{1}*2^0 + \textcolor{blue}{0} *2^1 & \textcolor{red}{1}\textcolor{blue}{0} \\
    2 = \textcolor{red}{0}*2^0 + \textcolor{blue}{1} *2^1 & \textcolor{red}{0}\textcolor{blue}{1} \\
    3 = \textcolor{red}{1}*2^0 + \textcolor{blue}{1} *2^1 & \textcolor{red}{1}\textcolor{blue}{1}
    \end{matrix}

Get State Asynchronous
~~~~~~~~~~~~~~~~~~~~~~~

Similar to `observe_async` above, `get_state` also supports asynchronous execution for the `arguments it accepts <https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html#cudaq.get_state_async>`_.

.. tab:: Python

    .. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin `GetStateAsync`]
        :end-before: [End `GetStateAsync`]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/executing_kernels_state_async.cpp
        :language: cpp
        :start-after: [Begin `GetStateAsync`]
        :end-before: [End `GetStateAsync`]

.. literalinclude:: ../../examples/python/executing_kernels.py
        :language: python
        :start-after: [Begin `GetStateAsyncOutput`]
        :end-before: [End `GetStateAsyncOutput`]
