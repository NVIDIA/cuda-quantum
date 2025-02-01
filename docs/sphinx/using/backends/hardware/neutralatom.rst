Neutral Atom
=============

Infleqtion
+++++++++++

.. _infleqtion-backend:

Infleqtion is a quantum hardware provider of gate-based neutral atom quantum computers. Their backends may be
accessed via `Superstaq <https://superstaq.infleqtion.com/>`__, a cross-platform software API from Infleqtion,
that performs low-level compilation and cross-layer optimization. To get started users can create a Superstaq
account by following `these instructions <https://superstaq.readthedocs.io/en/latest/get_started/credentials.html>`__.

For access to Infleqtion's neutral atom quantum computer, Sqale,
`pre-registration <https://www.infleqtion.com/sqale-preregistration>`__ is now open.

Setting Credentials
`````````````````````````

Programmers of CUDA-Q may access Infleqtion backends from either C++ or Python. Generate
an API key from your `Superstaq account <https://superstaq.infleqtion.com/profile>`__ and export
it as an environment variable:

.. code:: bash

  export SUPERSTAQ_API_KEY="superstaq_api_key"


Submitting
`````````````````````````

.. tab:: Python

        The target to which quantum kernels are submitted
        can be controlled with the ``cudaq::set_target()`` function.

        .. code:: python

            cudaq.set_target("infleqtion")

        By default, quantum kernel code will be submitted to Infleqtion's Sqale
        simulator.

        To specify which Infleqtion QPU to use, set the :code:`machine` parameter.

        .. code:: python

            cudaq.set_target("infleqtion", machine="cq_sqale_qpu")

        where ``cq_sqale_qpu`` is an example of a physical QPU.

        To run an ideal dry-run execution of the QPU, additionally set the ``method`` flag to ``"dry-run"``.

        .. code:: python

            cudaq.set_target("infleqtion", machine="cq_sqale_qpu", method="dry-run")

        To noisily simulate the QPU instead, set the ``method`` flag to ``"noise-sim"``.

        .. code:: python

            cudaq.set_target("infleqtion", machine="cq_sqale_qpu", method="noise-sim")

        Alternatively, to emulate the Infleqtion machine locally, without submitting through the cloud,
        you can also set the ``emulate`` flag to ``True``. This will emit any target
        specific compiler diagnostics, before running a noise free emulation.

        .. code:: python

            cudaq.set_target("infleqtion", emulate=True)

        The number of shots for a kernel execution can be set through
        the ``shots_count`` argument to ``cudaq.sample`` or ``cudaq.observe``. By default,
        the ``shots_count`` is set to 1000.

        .. code:: python

            cudaq.sample(kernel, shots_count=100)

        To see a complete example for using Infleqtion's backends, take a look at our :doc:`Python examples <../../examples/examples>`.
        Moreover, for an end-to-end application workflow example executed on the Infleqtion QPU, take a look at the 
        :doc:`Anderson Impurity Model ground state solver <../../applications>` notebook.


.. tab:: C++


        To target quantum kernel code for execution on Infleqtion's backends,
        pass the flag ``--target infleqtion`` to the ``nvq++`` compiler.

        .. code:: bash

            nvq++ --target infleqtion src.cpp

        This will take the API key and handle all authentication with, and submission to, Infleqtion's QPU 
        (or simulator). By default, quantum kernel code will be submitted to Infleqtion's Sqale
        simulator.

        To execute your kernels on a QPU, pass the ``--infleqtion-machine`` flag to the ``nvq++`` compiler
        to specify which machine to submit quantum kernels to:

        .. code:: bash

            nvq++ --target infleqtion --infleqtion-machine cq_sqale_qpu src.cpp ...

        where ``cq_sqale_qpu`` is an example of a physical QPU.

        To run an ideal dry-run execution on the QPU, additionally pass ``dry-run`` with the ``--infleqtion-method`` 
        flag to the ``nvq++`` compiler:

        .. code:: bash

            nvq++ --target infleqtion --infleqtion-machine cq_sqale_qpu --infleqtion-method dry-run src.cpp ...

        To noisily simulate the QPU instead, pass ``noise-sim`` to the ``--infleqtion-method`` flag like so:

        .. code:: bash

            nvq++ --target infleqtion --infleqtion-machine cq_sqale_qpu --infleqtion-method noise-sim src.cpp ...

        Alternatively, to emulate the Infleqtion machine locally, without submitting through the cloud,
        you can also pass the ``--emulate`` flag to ``nvq++``. This will emit any target
        specific compiler diagnostics, before running a noise free emulation.

        .. code:: bash

            nvq++ --emulate --target infleqtion src.cpp

        To see a complete example for using Infleqtion's backends, take a look at our :doc:`C++ examples <../../examples/examples>`.




QuEra Computing
++++++++++++++++


.. _quera-backend:

Setting Credentials
```````````````````

Programmers of CUDA-Q may access Aquila, QuEra's first generation of quantum
processing unit (QPU) via Amazon Braket. Hence, users must first enable Braket by 
following `these instructions <https://docs.aws.amazon.com/braket/latest/developerguide/braket-enable-overview.html>`__. 
Then set credentials using any of the documented `methods <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html>`__.
One of the simplest ways is to use `AWS CLI <https://aws.amazon.com/cli/>`__.

.. code:: bash

    aws configure

Alternatively, users can set the following environment variables.

.. code:: bash

  export AWS_DEFAULT_REGION="us-east-1"
  export AWS_ACCESS_KEY_ID="<key_id>"
  export AWS_SECRET_ACCESS_KEY="<access_key>"
  export AWS_SESSION_TOKEN="<token>"

Submission from Python
`````````````````````````

The target to which quantum kernels are submitted 
can be controlled with the ``cudaq::set_target()`` function.

.. code:: python

    cudaq.set_target('quera')

By default, analog Hamiltonian will be submitted to the Aquila system.

Aquila is a "field programmable qubit array" operated as an analog 
Hamiltonian simulator on a user-configurable architecture, executing 
programmable coherent quantum dynamics on up to 256 neutral-atom qubits.
Refer to QuEra's `whitepaper <https://cdn.prod.website-files.com/643b94c382e84463a9e52264/648f5bf4d19795aaf36204f7_Whitepaper%20June%2023.pdf>`__ for details.

Due to the nature of the underlying hardware, this target only supports the 
``evolve`` and ``evolve_async`` APIs.
The `hamiltonian` must be an `Operator` of the type `RydbergHamiltonian`. Only 
other parameters supported are `schedule` (mandatory) and `shots_count` (optional).

For example,

.. code:: python

    evolution_result = evolve(RydbergHamiltonian(atom_sites=register,
                                                 amplitude=omega,
                                                 phase=phi,
                                                 delta_global=delta),
                               schedule=schedule)

The number of shots for a kernel execution can be set through the ``shots_count``
argument to ``evolve`` or ``evolve_async``. By default, the ``shots_count`` is 
set to 100.

.. code:: python 

    cudaq.evolve(RydbergHamiltonian(...), schedule=s, shots_count=1000)

To see a complete example for using QuEra's backend, take a look at our :doc:`Python examples <../../examples/hardware_providers>`.
