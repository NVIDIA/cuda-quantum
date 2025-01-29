Photonics Simulators
==================================

The :code:`orca-photonics` target provides a state vector simulator with the :code:`Q++` library. 
The :code:`orca-photonics` target supports supports a double precision simulator that can run in multiple CPUs.

OpenMP CPU-only
++++++++++++++++++++++++++++++++++

.. _qpp-cpu-photonics-backend:

This target provides a state vector simulator based on the CPU-only, OpenMP threaded `Q++ <https://github.com/softwareqinc/qpp>`_  library.
To execute a program on the :code:`orca-photonics` target, use the following commands:

.. tab:: Python

    .. code:: bash

        python3 program.py [...] --target orca-photonics

    The target can also be defined in the application code by calling

    .. code:: python
 
        cudaq.set_target('orca-photonics')

    If a target is set in the application code, this target will override the :code:`--target` command line flag given during program invocation.

.. tab:: C++

    .. code:: bash

        nvq++ --library-mode --target orca-photonics program.cpp [...] -o program.x
