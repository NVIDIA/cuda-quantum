Quantum Control Systems
========================

Quantum Machines
+++++++++++++++++++++

.. _quantum-machines-backend:

Quantum Machines provides a unified quantum computing control system that enables 
users to execute quantum programs on various QPU types (hardware backends) through
their Quantum Orchestration Platform (QOP) - a platform that controls different 
quantum hardware implementations through a unified software API.

For information about available hardware backends and their capabilities, 
please consult the `Quantum Machines website <https://www.quantum-machines.co/>`__.


Setting Credentials
`````````````````````````

To use Quantum Machines with CUDA-Q, you need to have an API key and access to 
the Quantum Machines server or their `QOperator` service. You can set it using 
an environment variable:

.. code-block:: bash

    export QUANTUM_MACHINES_API_KEY="<your_api_key>"

Alternatively, you can provide it directly when setting the target in your code.


Submitting
`````````````````````````

To specify which backend to use, set the `executor` parameter when configuring 
the target. The available backends depend on your specific access rights and 
set up with Quantum Machines.  By default, a mock executor is used.

.. tab:: Python

    To target Quantum Machines from Python, use the ``cudaq.set_target()`` function:

    .. code:: python

        cudaq.set_target("quantum_machines", 
                        url="https://api.quantum-machines.com", 
                        api_key="your_api_key",
                        executor="mock")

    Parameters:

    - ``url``: The URL of the Quantum Machines server
    - ``executor``: The name of the executor/backend to use (defaults to "mock")
    - ``api_key``: Your API key (optional if set via environment variable)


.. tab:: C++

    To target quantum kernel code for execution on Quantum Machines, pass the 
    flag ``--target quantum_machines`` to the ``nvq++`` compiler:

    .. code-block:: bash

        nvq++ --target quantum_machines --quantum-machines-url "https://api.quantum-machines.com" src.cpp

    You can specify additional parameters:

    - ``--quantum-machines-url``: The URL of the QOperator server
    - ``--quantum-machines-executor``: The name of the executor/backend to use (defaults to "mock")
    - ``--quantum-machines-api-key``: Your API key (if not set via environment variable)

To see a complete example, take a look at :ref:`Quantum Machines examples <quantum-machines-examples>`.
