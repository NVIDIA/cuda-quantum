
Other Simulators
==================================

Stim (CPU)
++++++++++++++++++++++++++++++++++

.. _stim-backend:

This target provides a fast simulator for circuits containing *only* Clifford
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

Fermioniq
++++++++++

.. _fermioniq-backend:

`Fermioniq <https://fermioniq.com/>`__ offers a cloud-based tensor-network emulation platform, `Ava <https://www.fermioniq.com/ava/>`__, 
for the approximate simulation of large-scale quantum circuits beyond the memory limit of state vector and exact tensor network based methods. 

The level of approximation can be controlled by setting the bond dimension: larger values yield more accurate simulations at the expense 
of slower computation time. For a detailed description of Ava users are referred to the `online documentation <https://docs.fermioniq.com/>`__.

Users of CUDA-Q can access a simplified version of the full Fermioniq emulator (`Ava <https://www.fermioniq.com/ava/>`__) from either
C++ or Python. This version currently supports emulation of quantum circuits without noise, and can return measurement samples and/or 
compute expectation values of observables.

.. note::
    In order to use the Fermioniq emulator, users must provide access credentials. These can be requested by contacting info@fermioniq.com 

    The credentials must be set via two environment variables:
    `FERMIONIQ_ACCESS_TOKEN_ID` and `FERMIONIQ_ACCESS_TOKEN_SECRET`.

.. tab:: Python

    The target to which quantum kernels are submitted 
    can be controlled with the ``cudaq::set_target()`` function.

    .. code:: python

        cudaq.set_target('fermioniq')

    You will have to specify a remote configuration id for the Fermioniq backend
    during compilation.

    .. code:: python

        cudaq.set_target("fermioniq", **{
            "remote_config": remote_config_id
        })

    For a comprehensive list of all remote configurations, please contact Fermioniq directly.

    When your organization requires you to define a project id, you have to specify
    the project id during compilation.

    .. code:: python

        cudaq.set_target("fermioniq", **{
            "project_id": project_id
        })

    To specify the bond dimension, you can pass the ``bond_dim`` parameter.

    .. code:: python 

        cudaq.set_target("fermioniq", **{
            "bond_dim": 5
        })

.. tab:: C++

    To target quantum kernel code for execution in the Fermioniq backends,
    pass the flag ``--target fermioniq`` to the ``nvq++`` compiler. CUDA-Q will 
    authenticate via the Fermioniq REST API using the environment variables 
    set earlier.

    .. code:: bash

        nvq++ --target fermioniq src.cpp ...

    You will have to specify a remote configuration id for the Fermioniq backend
    during compilation.

    .. code:: bash

        nvq++ --target fermioniq --fermioniq-remote-config <remote_config_id> src.cpp ...

    For a comprehensive list of all remote configurations, please contact Fermioniq directly.

    When your organization requires you to define a project id, you have to specify
    the project id during compilation.

    .. code:: bash

        nvq++ --target fermioniq --fermioniq-project-id <project_id> src.cpp ...

    To specify the bond dimension, you can pass the ``fermioniq-bond-dim`` parameter.

    .. code:: bash

        nvq++ --target fermioniq --fermioniq-bond-dim 10 src.cpp ...

