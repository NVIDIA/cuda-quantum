
Density Matrix Simulators
==================================


Density Matrix 
++++++++++++++++

.. _density-matrix-cpu-backend:

Density matrix simulation is helpful for understanding the impact of noise on quantum applications. Unlike state vectors simulation which manipulates the :math:`2^n` state vector, density matrix simulations manipulate the :math:`2^n x 2^n`  density matrix which defines an ensemble of states. To learn how you can leverage the :code:`density-matrix-cpu` backend to study the impact of noise models on your applications, see the  `example here <https://nvidia.github.io/cuda-quantum/latest/examples/python/noisy_simulations.html>`__.

The `Quantum Volume notebook <https://nvidia.github.io/cuda-quantum/latest/applications/python/quantum_volume.html>`__ also demonstrates a full application that leverages the :code:`density-matrix-cpu` backend. 

To execute a program on the :code:`density-matrix-cpu` target, use the following commands:

.. tab:: Python

    .. code:: bash 

        python3 program.py [...] --target density-matrix-cpu

    The target can also be defined in the application code by calling

    .. code:: python 

        cudaq.set_target('density-matrix-cpu')

    If a target is set in the application code, this target will override the :code:`--target` command line flag given during program invocation.

.. tab:: C++

    .. code:: bash 

        nvq++ --target density-matrix-cpu program.cpp [...] -o program.x
        ./program.x


Stim 
++++++

.. _stim-backend:

This backend provides a fast simulator for circuits containing *only* Clifford
gates. Any non-Clifford gates (such as T gates and Toffoli gates) are not
supported. This simulator is based on the `Stim <https://github.com/quantumlib/Stim>`_
library.

To execute a program on the :code:`stim` target, use the following commands:

.. tab:: Python

    .. code:: bash 

        python3 program.py [...] --target stim

    The target can also be defined in the application code by calling

    .. code:: python 

        cudaq.set_target('stim')

    If a target is set in the application code, this target will override the :code:`--target` command line flag given during program invocation.

.. tab:: C++

    .. code:: bash 

        nvq++ --target stim program.cpp [...] -o program.x
        ./program.x

.. note::
    CUDA-Q currently executes kernels using a "shot-by-shot" execution approach.
    This allows for conditional gate execution (i.e. full control flow), but it
    can be slower than executing Stim a single time and generating all the shots
    from that single execution.
