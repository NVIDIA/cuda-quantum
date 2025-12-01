Neutral Atom
=============

Infleqtion
+++++++++++

.. _infleqtion-backend:

Infleqtion is a quantum hardware provider of gate-based neutral atom quantum computers. Their backends may be
accessed via `Superstaq <https://superstaq.infleqtion.com/>`__, a cross-platform software API from Infleqtion,
that performs low-level compilation and cross-layer optimization. To get started users can create a Superstaq
account by following `these instructions <https://superstaq.readthedocs.io/en/latest/get_started/credentials.html>`__.

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
        can be controlled with the ``cudaq.set_target()`` function.

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


To see a complete example, take a look at :ref:`Infleqtion examples <infleqtion-examples>`.
Moreover, for an end-to-end application workflow example executed on the Infleqtion QPU, take a look at the
:doc:`Anderson Impurity Model ground state solver <../../../applications/python/logical_aim_sqale>` notebook.

.. note:: 

        In local emulation mode (``emulate`` flag set to ``True``), the program will be executed on the :ref:`default simulator <default-simulator>`.
        The environment variable ``CUDAQ_DEFAULT_SIMULATOR`` can be used to change the emulation simulator. 
        
        For example, the simulation floating point accuracy and/or the simulation capabilities (e.g., maximum number of qubits, supported quantum gates),
        depend on the selected simulator.  
        
        Any environment variables must be set prior to setting the target or running "`import cudaq`".

Pasqal
++++++++++++++++

Pasqal is a quantum computing hardware company that builds quantum processors from ordered neutral atoms in 2D and 3D
arrays to bring a practical quantum advantage to its customers and address real-world problems.
The currently available Pasqal QPUs are analog quantum computers, and one, named Fresnel, is available through our cloud
portal.

In order to access Pasqal's devices you need an account for `Pasqal's cloud platform <https://portal.pasqal.cloud>`__
and an active project. Please see our `cloud documentation <https://docs.pasqal.cloud/cloud/>`__ for more details if needed.

Although a different SDK, `Pasqal's Pulser library <https://pulser.readthedocs.io/en/latest/>`__, is a good
resource for getting started with analog neutral atom quantum computing.
For support you can also join the `Pasqal Community <https://community.pasqal.com/>`__.


.. _pasqal-backend:

Setting Credentials
```````````````````

An authentication token for the session must be obtained from Pasqal's cloud platform.
For example from Python one can use the `pasqal-cloud package <https://github.com/pasqal-io/pasqal-cloud>`__ as below:

.. code:: python

    from pasqal_cloud import SDK
    import os

    sdk = SDK(
        username=os.environ.get['PASQAL_USERNAME'],
        password=os.environ.get('PASQAL_PASSWORD', None)
    )

    token = sdk.user_token()

    os.environ['PASQAL_AUTH_TOKEN'] = str(token)
    os.environ['PASQAL_PROJECT_ID'] = 'your project id'

Alternatively, users can set the following environment variables directly.

.. code:: bash

  export PASQAL_AUTH_TOKEN=<>
  export PASQAL_PROJECT_ID=<>


Submitting
`````````````````````````
.. tab:: Python

        The target to which quantum kernels are submitted 
        can be controlled with the ``cudaq.set_target()`` function.

        .. code:: python

            cudaq.set_target('pasqal')


        This accepts an optional argument, ``machine``, which is used in the cloud platform to
        select the corresponding Pasqal QPU or emulator to execute on.
        See the `Pasqal cloud portal <https://portal.pasqal.cloud/>`__ for an up to date list.
        The default value is ``EMU_MPS`` which is an open-source tensor network emulator based on the
        Matrix Product State formalism running in Pasqal's cloud platform. You can see the
        documentation for the publicly accessible emulator `here <https://pasqal-io.github.io/emulators/latest/emu_mps/>`__.

        To target the QPU use the FRESNEL machine name. Note that there are restrictions
        regarding the values of the pulses as well as the register layout. We invite you to
        consult our `documentation <https://docs.pasqal.com/cloud/fresnel-job>`__. Note that
        the CUDA-Q integration currently only works with `arbitrary layouts <https://docs.pasqal.com/cloud/fresnel-job/#arbitrary-layouts>`__
        which are implemented with automatic calibration for less than 30 qubits. For jobs
        larger than 30 qubits please use the `atom_sites` to define the layout, and use the
        `atom_filling` to select sites as filled or not filled in order to define the register.

        Due to the nature of the underlying hardware, this target only supports the 
        ``evolve`` and ``evolve_async`` APIs.
        The `hamiltonian` must be an `Operator` of the type `RydbergHamiltonian`. The only
        other supported parameters are `schedule` (mandatory) and `shots_count` (optional).

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


.. tab:: C++

        To target quantum kernel code for execution on Pasqal QPU or simulators,
        pass the flag ``--target pasqal`` to the ``nvq++`` compiler.

        .. code:: bash

            nvq++ --target pasqal src.cpp
        
        You can also pass the flag ``--pasqal-machine`` to select the corresponding Pasqal QPU or emulator to execute on.
        See the `Pasqal cloud portal <https://portal.pasqal.cloud/>`__ for an up to date list.
        The default value is ``EMU_MPS`` which is an open-source tensor network emulator based on the
        Matrix Product State formalism running in Pasqal's cloud platform. You can see the
        documentation for the publicly accessible emulator `here <https://pasqal-io.github.io/emulators/latest/emu_mps/>`__.

        .. code:: bash

            nvq++ --target pasqal --pasqal-machine EMU_FREE src.cpp

        To target the QPU use the FRESNEL machine name. Note that there are restrictions
        regarding the values of the pulses as well as the register layout. We invite you to
        consult our `documentation <https://docs.pasqal.com/cloud/fresnel-job>`__. Note that
        the CUDA-Q integration currently only works with `arbitrary layouts <https://docs.pasqal.com/cloud/fresnel-job/#arbitrary-layouts>`__
        which are implemented with automatic calibration for less than 30 qubits. For jobs
        larger than 30 qubits please use the `atom_sites` to define the layout, and use the
        `atom_filling` to select sites as filled or not filled in order to define the register.
        
        Due to the nature of the underlying hardware, this target only supports the 
        ``evolve`` and ``evolve_async`` APIs.
        The `hamiltonian` must be of the type `rydberg_hamiltonian`. Only 
        other parameters supported are `schedule` (mandatory) and `shots_count` (optional).

        For example,

        .. code:: cpp

            auto evolution_result = cudaq::evolve(
                cudaq::rydberg_hamiltonian(register_sites, omega, phi, delta),
                schedule);

        The number of shots for a kernel execution can be set through the ``shots_count``
        argument to ``evolve`` or ``evolve_async``. By default, the ``shots_count`` is 
        set to 100.

        .. code:: cpp

            auto evolution_result = cudaq::evolve(cudaq::rydberg_hamiltonian(...), schedule, 1000);


To see a complete example, take a look at :ref:`Pasqal examples <pasqal-examples>`.


.. note:: 

    Local emulation via ``emulate`` flag is not yet supported on the `pasqal` target.


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

About Aquila
`````````````````````````

Aquila is a "field programmable qubit array" operated as an analog 
Hamiltonian simulator on a user-configurable architecture, executing 
programmable coherent quantum dynamics on up to 256 neutral-atom qubits.
Refer to QuEra's `whitepaper <https://cdn.prod.website-files.com/643b94c382e84463a9e52264/648f5bf4d19795aaf36204f7_Whitepaper%20June%2023.pdf>`__ for details.

Submitting
`````````````````````````
.. tab:: Python

        The target to which quantum kernels are submitted
        can be controlled with the ``cudaq.set_target()`` function.

        .. code:: python

            cudaq.set_target('quera')

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


.. tab:: C++

        To target quantum kernel code for execution on QuEra's Aquila,
        pass the flag ``--target quera`` to the ``nvq++`` compiler.

        .. code:: bash

            nvq++ --target quera src.cpp
        
        Due to the nature of the underlying hardware, this target only supports the 
        ``evolve`` and ``evolve_async`` APIs.
        The `hamiltonian` must be of the type `rydberg_hamiltonian`. Only 
        other parameters supported are `schedule` (mandatory) and `shots_count` (optional).

        For example,

        .. code:: cpp

            auto evolution_result = cudaq::evolve(
                cudaq::rydberg_hamiltonian(register_sites, omega, phi, delta),
                schedule);

        The number of shots for a kernel execution can be set through the ``shots_count``
        argument to ``evolve`` or ``evolve_async``. By default, the ``shots_count`` is 
        set to 100.

        .. code:: cpp

            auto evolution_result = cudaq::evolve(cudaq::rydberg_hamiltonian(...), schedule, 1000);

To see a complete example, take a look at :ref:`QuEra Computing examples <quera-examples>`.

.. note:: 

    Local emulation via ``emulate`` flag is not yet supported on the `quera` target.
